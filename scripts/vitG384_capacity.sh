#!/usr/bin/env bash
# ViT-G (2B) @ 384 surgical CPT — single long capacity job, 12h walltime.
# 2B sibling of vitg384_capacity.sh (1B). Runs continuously (no self-resubmit,
# no EXIT_AFTER_CKPT); swaps in for the debug-scaling chain via the shared LOCK
# once a large allocation lands. See vitG384_chain_debugscaling.sh header and
# memory vitG-2b-swap for rationale.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_capacity.sh
#
#PBS -N vitG_cap
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=16
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -o pipefail  # NOT set -e: module load/venv activate can return nonzero
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
# FIXED-SHAPE config (§4f fix): cleandata reintroduces per-step mask VA churn (churn
# source #2). fixedshape.yaml is IDENTICAL to cleandata EXCEPT it pins num_keep_enc/pred,
# so it keeps the exact training schedule (epochs/ipe/LR) but removes the churn. This is
# the measured env-diff baseline — do NOT revert to cleandata for a real run.
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitG384_fixedshape.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitG384_fixedshape.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_vitG384_fixedshape/vitG384_n16g12_weak
PARAMS=$CKPT_DIR/params-pretrain.yaml
LOCK=$CKPT_DIR/.training.lock
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

if [[ ! -f "$PARAMS" ]]; then
  echo "Staging runtime cfg -> $PARAMS"
  $PY $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
  cp $RUNTIME_CFG $PARAMS
fi

NUM_EPOCHS=$($PY -c "import yaml; print(yaml.safe_load(open('$PARAMS'))['optimization']['epochs'])")
CURRENT_EPOCH=0
if [[ -f $CKPT_DIR/latest.pth.tar ]]; then
  CURRENT_EPOCH=$($PY -c "import torch; print(torch.load('$CKPT_DIR/latest.pth.tar', map_location='cpu', weights_only=False).get('epoch', 0))" 2>/dev/null || echo 0)
fi
echo "progress: epoch $CURRENT_EPOCH / $NUM_EPOCHS"
START_EP=$CURRENT_EPOCH   # captured for the self-healing resubmit progress-guard at exit
if (( CURRENT_EPOCH >= NUM_EPOCHS )); then
  echo "ViT-G CPT complete (epoch >= num_epochs). capacity job stops."
  rm -f "$LOCK"
  exit 0
fi

# Single long job (no self-resubmit). It honors the shared LOCK so it will not
# collide with a debug-scaling chain slice; once it starts, the chain drains.
if [[ -f "$LOCK" ]]; then
  HOLDER=$(cat "$LOCK" 2>/dev/null || echo "")
  if [[ -n "$HOLDER" ]] && qstat "$HOLDER" 2>/dev/null | awk 'NR>2{print $5}' | grep -q '^R$'; then
    echo "LOCK held by running job $HOLDER — another slice is training. Skipping."
    echo "JOB END (skipped): $(date)"
    exit 0
  else
    echo "stale lock from $HOLDER (not running) — taking over."
  fi
fi
echo "$PBS_JOBID" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd $ROOT
# HSDP requires torch 2.13 (native FSDP1 + xccl). Load frameworks FIRST, then the
# pt213 venv on top (module gives oneCCL/MPI, venv gives torch 2.13).
module load frameworks
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
# HSDP TRANSPORT (validated: launcher=none + ofi; pmix/mpi DEADLOCKS FSDP subgroup
# collectives at iter0 — 2n A/B job 8643134 ofi=60 clean iters vs mpi=hang, 16n
# confirmed job 8643156). With launcher=none the train mpiexec drops --pmi=pmix.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_KVS_IFACE=hsn0
export CCL_OP_SYNC=1
# CCL_WORKER_COUNT=1: the PROVEN base. The only 16n run to reach 74 iters (env-diff
# 8643398) used workers=1; the workers=4 A/B (8643434) failed its first-iter test.
# For an unattended launch, proven > theoretical. (workers=4 remains a daytime A/B to
# retry against the §4g host-stalls once someone can babysit it.)
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export WDS_LOCAL_SLICING=1
# --- HSDP: shard params/grads/opt across the 12 tiles (57.6GB DDP -> ~22GB/tile),
# removing the DDP L0-headroom wedge. bf16 comm hook + DDP bucket are DDP-only and
# not used under HSDP (FSDP MixedPrecision handles reduce dtype). ---
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=12
export FSDP_SHARDING=shard_grad_op   # _HYBRID_SHARD_ZERO2
# §4f/§4e: the three "insurance" flags below were FALSIFIED by the env-diff run
# (8643398) — they were not the accumulator, memory is flat with/without them, and
# PRISM's production launcher is grep-clean of all three. Keeping them means NOT
# matching the measured baseline. They are UNSET here (not exported).
unset PYTORCH_ALLOC_CONF
unset FI_MR_CACHE_MONITOR
unset CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD
# TRUE grad accumulation (§4f/§4g fabric lever): fetch N loader batches/step, one
# inter-node collective per step instead of N (fewer host-side-stall opportunities).
# 1n memory smoke (8643455) PASSED: no_sync full-grad fits with 10.3GiB free L0, loss
# sane. Default 1 (=proven plain base) unless the 16n verify (8643462) confirms it
# flattens the fabric spikes — then launch with VJEPA_TRUE_ACCUM=2 in the qsub env.
# Effective global batch scales Nx; LR unchanged (memory true-accum-lr-decision).
export VJEPA_TRUE_ACCUM=${VJEPA_TRUE_ACCUM:-1}
# NOTE: capacity job runs CONTINUOUSLY — do NOT set VJEPA_EXIT_AFTER_CKPT (that's
# only for the 1h debug-scaling chain slices).
if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"
echo "HSDP ENVS: VJEPA_DIST_STRATEGY=$VJEPA_DIST_STRATEGY FSDP_SHARDING=$FSDP_SHARDING TRUE_ACCUM=$VJEPA_TRUE_ACCUM WORKER_COUNT=$CCL_WORKER_COUNT transport=none/ofi"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT (per-node disjoint) ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS \
        --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

# ---- STALL WATCHDOG (background): kill a hung training so the successor can take over.
# The §4g fabric stalls are recoverable multi-minute spikes; the DDP wedge hit 328s and
# recovered. Only a TRUE hang (no new CSV rows for a long time) should trigger a kill.
# 1800s (30min) >> any observed recoverable spike (max ~544s) but still bounds a real hang.
CSV_WATCH="$CKPT_DIR/log_r0.csv"
STALL_DEADLINE=1800
FIRST_ITER_DEADLINE=1200   # staging+wrap+load+first iter (loader fill is slow at 16n)
(
    start=$(date +%s); last_rows=-1; last_change=$start
    while true; do
        sleep 60
        pgrep -f "app.main_dist_aurora" >/dev/null 2>&1 || exit 0
        now=$(date +%s)
        rows=0; [ -f "$CSV_WATCH" ] && rows=$(($(wc -l < "$CSV_WATCH" 2>/dev/null || echo 1)-1))
        if [ "$rows" -gt 0 ]; then
            if [ "$rows" -ne "$last_rows" ]; then last_rows=$rows; last_change=$now; fi
            if [ $((now-last_change)) -gt $STALL_DEADLINE ]; then
                echo "WATCHDOG: STALL — no new iters for ${STALL_DEADLINE}s at row $rows. Killing for resubmit." >&2
                pkill -9 -f "app.main_dist_aurora"; exit 1
            fi
        elif [ $((now-start)) -gt $FIRST_ITER_DEADLINE ]; then
            echo "WATCHDOG: NO FIRST ITER within ${FIRST_ITER_DEADLINE}s. Killing for resubmit." >&2
            pkill -9 -f "app.main_dist_aurora"; exit 1
        fi
    done
) &
WATCHDOG_PID=$!

mpiexec -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
TRAIN_RC=$?
kill $WATCHDOG_PID 2>/dev/null
echo "JOB END: $(date) (train rc=$TRAIN_RC)"

# ---- SELF-HEALING RESUBMIT: if training did not finish all epochs, resubmit a successor
# (which auto-resumes from latest.pth.tar, train.py:375). Skips if the run is complete or a
# successor is already queued. Mirrors the chain launcher's resilience but for the 12h job,
# so an intrinsic-fabric-stall hang (watchdog-killed) does not end the training campaign.
release_lock() { rm -f "$LOCK"; }
CUR_EP=0
[[ -f $CKPT_DIR/latest.pth.tar ]] && CUR_EP=$($PY -c "import torch;print(torch.load('$CKPT_DIR/latest.pth.tar',map_location='cpu',weights_only=False).get('epoch',0))" 2>/dev/null || echo 0)
# Resubmit-storm guard: count CONSECUTIVE runs that made no epoch progress. A startup
# crash loop (e.g. the loader-fill hang seen 3x) must not churn the queue forever.
FAILF=$CKPT_DIR/.consecutive_noprogress
FAILS=$(cat "$FAILF" 2>/dev/null || echo 0)
if (( CUR_EP > START_EP )); then FAILS=0; else FAILS=$((FAILS+1)); fi
echo "$FAILS" > "$FAILF"
echo "resubmit-guard: start_ep=$START_EP cur_ep=$CUR_EP consecutive_noprogress=$FAILS"
if (( CUR_EP >= NUM_EPOCHS )); then
  echo "campaign complete (epoch $CUR_EP >= $NUM_EPOCHS) — no resubmit."
  release_lock
elif (( FAILS >= 4 )); then
  echo "STORM GUARD: $FAILS consecutive runs made no progress — STOPPING resubmit. Needs a human."
  release_lock
else
  QUEUED=$(qstat -u "$USER" 2>/dev/null | grep -c "vitG_cap")
  if (( QUEUED > 1 )); then
    echo "successor already queued ($QUEUED vitG_cap jobs) — no resubmit."
  else
    release_lock  # let the successor take the lock cleanly
    NEXT=$(qsub "$ROOT/scripts/vitG384_capacity.sh" 2>&1) && echo "RESUBMITTED successor: $NEXT" || echo "resubmit failed: $NEXT"
  fi
fi
