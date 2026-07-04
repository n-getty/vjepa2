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
# CCL_WORKER_COUNT=4 (§4g): matches PRISM production. The env-diff run showed
# intermittent multi-minute HOST-SIDE collective stalls (iter-ms 365s w/ gpu-ms 21s)
# — a single progress-engine worker (=1) is a plausible cause; PRISM uses 4 (8 -> EINVAL).
# Under test in job 8643434; this launcher adopts the value pending that A/B's confirmation.
export CCL_WORKER_COUNT=4
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
echo "HSDP ENVS: VJEPA_DIST_STRATEGY=$VJEPA_DIST_STRATEGY FSDP_SHARDING=$FSDP_SHARDING transport=none/ofi"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT (per-node disjoint) ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS \
        --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

mpiexec -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"
