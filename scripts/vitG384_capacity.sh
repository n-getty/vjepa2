#!/usr/bin/env bash
# ViT-G (2B) @ 384 surgical CPT — capacity job, 6h walltime (was 12h; hard cap
# per user: NO 12h jobs while recipe is unresolved). Walltime is set IN-SCRIPT so
# every self-resubmit successor also inherits 6h — a CLI -l override would not.
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
#PBS -l walltime=06:00:00
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

# KILL-SWITCH: if this sentinel exists, do NOT run (and thus do not resubmit). Lets us stop
# the self-healing chain from launching any NEW 12h job while keeping the current one running.
# Any successor 8644288 might qsub will hit this and exit in seconds (no 12h consumed).
if [[ -f "$CKPT_DIR/.no_relaunch" ]]; then
  echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID — .no_relaunch sentinel present, EXITING (no new 12h job)."
  exit 0
fi

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

if [[ ! -f "$PARAMS" ]]; then
  echo "Staging runtime cfg -> $PARAMS"
  $PY $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale > /dev/null
  cp $RUNTIME_CFG $PARAMS
fi
# CRITICAL (fixed 2026-07-04 07:40): force the trainer's output `folder` to EQUAL $CKPT_DIR.
# The runtime cfg's folder was the OLD cleandata dir, so training wrote CSVs+checkpoints there
# while the watchdog / auto-resume / storm-guard all watched $CKPT_DIR — they never saw progress,
# so the watchdog KILLED HEALTHY TRAINING JOBS at the 1200s "no first iter" deadline (8643547
# was at iter 60+, loss 0.33, when killed). Patch folder so all four agree. Idempotent.
$PY - "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
p, folder = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(p))
if d.get("folder") != folder:
    print(f"PATCHING folder: {d.get('folder')} -> {folder}")
    d["folder"] = folder
    yaml.safe_dump(d, open(p, "w"), sort_keys=False)
else:
    print(f"folder already correct: {folder}")
PY

NUM_EPOCHS=$($PY -c "import yaml; print(yaml.safe_load(open('$PARAMS'))['optimization']['epochs'])")
CURRENT_EPOCH=0
if [[ -f $CKPT_DIR/latest.pth.tar ]]; then
  CURRENT_EPOCH=$($PY -c "import torch; print(torch.load('$CKPT_DIR/latest.pth.tar', map_location='cpu', weights_only=False).get('epoch', 0))" 2>/dev/null || echo 0)
fi
echo "progress: epoch $CURRENT_EPOCH / $NUM_EPOCHS"
START_EP=$CURRENT_EPOCH   # captured for the self-healing resubmit progress-guard at exit
# Row count in log_r0.csv at start — the trainer APPENDS across resumes, so iters-THIS-RUN =
# rows_at_end - rows_at_start. Lets the resubmit policy tell a pre-first-iter crash (0 new rows)
# from a job that trained then failed. (CKPT_DIR == config folder after the patch above.)
ROWS_AT_START=$(awk -F, '$2~/^[0-9]+$/{n++} END{print n+0}' "$CKPT_DIR/log_r0.csv" 2>/dev/null || echo 0)
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
# oneCCL SYCL persistent-temp-buffer collectives (env-gated, DEFAULT OFF). Sibling torchtune's
# investigation (torchtune/scratch/bug_draft_feedback.txt, docs/bugs/ccl_ipc_handle_cache.md)
# fingers our class of Mode-B hang as oneCCL **stale Level-Zero IPC handle accumulation** in GPU
# collectives (signature #3). These knobs route allreduce/reduce-scatter/allgather through
# PERSISTENT TEMP BUFFERS instead of L0 IPC — the single most promising untested workaround for
# that leak. Kept OFF by default (proven recipe unchanged); flip VJEPA_CCL_TMP_BUF=1 to A/B it
# against the hang rate. Matches our own [[vitG-2b-allreduce-spikes]] (backward-phase spikes +
# L0 external-mem growth the 1B never hit).
if [[ "${VJEPA_CCL_TMP_BUF:-0}" == "1" ]]; then
  export CCL_SYCL_ALLREDUCE_TMP_BUF=1
  export CCL_SYCL_REDUCE_SCATTER_TMP_BUF=1
  export CCL_SYCL_ALLGATHERV_TMP_BUF=1
  export CCL_SYCL_ALLGATHER_TMP_BUF=1
  export CCL_SYCL_BROADCAST_TMP_BUF=1
  echo "CCL TMP_BUF workaround ON (persistent temp buffers, bypass L0 IPC handles)"
fi
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
# MODE-A FIX (2026-07-04): the shm-unmap startup crash that killed ~60% of jobs is the
# DataLoader worker mp/file-backed-shm handoff being fragile at 192-rank scale (NOT /dev/shm
# exhaustion — verified 4K/504G empty). num_workers=0 (main-process loading) eliminates it:
# 16n modeA_diag 8644227 clean, 5.6-6.9s/iter, data ~1.3s overlapped = negligible cost
# (WebDataset streaming is I/O-light). Overridable via qsub -v but default 0 for survivability.
export VJEPA_NUM_WORKERS=${VJEPA_NUM_WORKERS:-0}
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
# HANG FORENSICS: per-rank in-process watchdog (train.py). 600s > max observed recoverable
# spike (~544s) so it fires ONLY on a true hang, and dumps the stuck rank's stack BEFORE the
# 1800s shell-watchdog kill — giving us the exact blocked collective/line at the hang moment.
# Also arms the SIGABRT all-thread dump the shell watchdog triggers. Diagnostic only, no
# training-behavior change. See memory vitG-2b-allreduce-spikes / the hang-forensics work.
export VJEPA_ITER_WATCHDOG_S=${VJEPA_ITER_WATCHDOG_S:-600}
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

# ---- PRE-LAUNCH SHM HYGIENE + evidence (Mode-A mitigation): a prior/killed job on a node can
# leave /dev/shm torch/psm segments that starve the next job's DataLoader workers (the shm-unmap
# crash). Clean our own patterns per node + log df/ipcs so exhaustion is visible if it recurs.
echo "=== per-node /dev/shm BEFORE (df used / shm-file count) ==="
mpiexec -n 16 -ppn 1 --cpu-bind none bash -c \
  'echo "[$(hostname)] $(df -h /dev/shm 2>/dev/null|awk "NR==2{print \$3\"/\"\$2}") files=$(ls /dev/shm 2>/dev/null|wc -l)"' 2>&1 | grep -viE "warn" | sort | head -20
mpiexec -n 16 -ppn 1 --cpu-bind none bash -c \
  'rm -f /dev/shm/torch_* /dev/shm/*psm* /dev/shm/sem.* 2>/dev/null; true' 2>&1 | grep -viE "warn" | head -2
echo "--- shm hygiene done ---"

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
DIAG_DIR="$CKPT_DIR/hang_diag"; mkdir -p "$DIAG_DIR"
# HANG FORENSICS (added 2026-07-05, after user challenge: we were blind-killing hangs with
# ZERO evidence, unlike AGPT/torchtitan + PRISM which instrument the stuck rank). Before the
# pkill we now: (1) SIGUSR1 every rank's python so the ARMED faulthandler (VJEPA_ITER_WATCHDOG_S
# registers SIGUSR1 -> dump-all-threads-and-continue) prints ALL-THREAD STACKS to the job stderr
# -> shows exactly which collective/line each rank is blocked on; (2) snapshot the PBS_NODEFILE so
# we can attribute the hang to specific physical nodes across incidents (bad-node tracking, like
# AGPT bad_nodes.txt). mpiexec fan-out of the SIGUSR1 so ALL 16 nodes' ranks dump, not just node 0.
# (SIGUSR1 not SIGABRT: faulthandler cannot register SIGABRT on this build, and SIGUSR1 dumps
# without aborting so the stacks flush cleanly before the hard pkill -9.)
capture_hang_forensics() {
    local tag="$1"
    local stamp; stamp=$(date +%Y%m%d_%H%M%S)
    echo "WATCHDOG: capturing hang forensics ($tag) -> $DIAG_DIR/hang_${stamp}_*" >&2
    # nodefile snapshot for node attribution
    [ -f "${PBS_NODEFILE:-}" ] && cp "$PBS_NODEFILE" "$DIAG_DIR/hang_${stamp}_nodefile.txt" 2>/dev/null
    echo "jobid=$PBS_JOBID last_csv_row=$(tail -1 "$CSV_WATCH" 2>/dev/null)" > "$DIAG_DIR/hang_${stamp}_info.txt"
    # SIGUSR1 all ranks on all nodes -> faulthandler dumps per-rank all-thread stack to stderr
    # (dump-and-continue). Give it ~20s to flush before the hard kill. Best-effort (|| true):
    # never let diag block the kill.
    mpiexec -n 16 -ppn 1 --cpu-bind none bash -c 'pkill -USR1 -f app.main_dist_aurora 2>/dev/null; true' >/dev/null 2>&1 || true
    sleep 20
}
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
                echo "WATCHDOG: STALL — no new iters for ${STALL_DEADLINE}s at row $rows. Capturing forensics then killing." >&2
                capture_hang_forensics "stall_row${rows}"
                pkill -9 -f "app.main_dist_aurora"; exit 1
            fi
        elif [ $((now-start)) -gt $FIRST_ITER_DEADLINE ]; then
            echo "WATCHDOG: NO FIRST ITER within ${FIRST_ITER_DEADLINE}s. Capturing forensics then killing." >&2
            capture_hang_forensics "no_first_iter"
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
# HARDENED RESUBMIT POLICY (2026-07-04, after the crash-loop incident). Two counters:
#  - ITERS THIS RUN: did this job reach iter 0 at all? A pre-first-iter crash (Mode A shm
#    startup crash) must NEVER auto-resubmit — that was the reckless 12h-churn (7 jobs died
#    in ~5min without training, ~2300 node-hrs wasted). No iters => STOP, needs a human.
#  - EPOCH PROGRESS: only reset the guard when an epoch actually banked.
ROWS_AT_END=$(awk -F, '$2~/^[0-9]+$/{n++} END{print n+0}' "$CKPT_DIR/log_r0.csv" 2>/dev/null || echo 0)
ITERS_THIS_RUN=$((ROWS_AT_END - ROWS_AT_START))
FAILF=$CKPT_DIR/.consecutive_noprogress
FAILS=$(cat "$FAILF" 2>/dev/null || echo 0)
if (( CUR_EP > START_EP )); then FAILS=0; else FAILS=$((FAILS+1)); fi
echo "$FAILS" > "$FAILF"
echo "resubmit-policy: start_ep=$START_EP cur_ep=$CUR_EP iters_this_run=$ITERS_THIS_RUN consecutive_noprogress=$FAILS"
if (( CUR_EP >= NUM_EPOCHS )); then
  echo "campaign complete (epoch $CUR_EP >= $NUM_EPOCHS) — no resubmit."
  release_lock
elif (( ITERS_THIS_RUN == 0 )); then
  # Pre-first-iter crash (Mode A). Do NOT resubmit — a job that can't even start must not
  # churn 12h reservations. Requires human diagnosis (run scripts/modeA_diag.sh).
  echo "PRE-FIRST-ITER CRASH (0 iters logged) — NOT resubmitting. Mode A; diagnose with scripts/modeA_diag.sh before relaunch."
  release_lock
elif (( FAILS >= 5 )); then
  # Reached iters but hung/failed repeatedly without banking an epoch. Cap at 5 (not 40) —
  # cheap retries are fine but 5 consecutive no-epoch runs = something wrong, needs a human.
  echo "RESUBMIT CAP: $FAILS consecutive runs reached iters but banked no epoch — STOPPING. Investigate (Mode B hang?)."
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
