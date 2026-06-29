#!/usr/bin/env bash
# Self-resubmitting ViT-g @ 384 surgical CPT chain, 1h walltime on debug-scaling.
#
# Runs the OPTIMAL run: undistilled Meta ViT-g @ native 384, context loss on
# (lambda 0.5, ramp rescaled to the run), predictor warm-started, with the
# bf16 DDP comm hook + bucket50 (-32% iter-time, validated 2026-06-26). See
# configs/vitg16_surg_vid_webdataset_single4/vitg384_cresume_const15.yaml and memory
# optimal-cpt-vitg384 / vitg-optimization-plan for the full rationale.
#
# Strategy: debug-scaling backfills 1h jobs fast and is a SEPARATE per-user slot,
# so this chain makes continuous progress 1h at a time while a capacity job (up
# to 16h, but ≤128 nodes cluster-wide so it can wait) is queued separately and
# swaps in when it lands. BOTH point at the same CKPT_DIR and resume from
# latest.pth.tar (verified chain-safe: train.py:540 overrides the Meta init with
# latest if present). The LOCK guard prevents two slices training the same ckpt.
#
# Submit first instance:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitg384_cresume_const15_chain.sh
#
#PBS -N vitg_cr
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/vitg384_cresume_const15.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitg16_surg_vid_webdataset_single4/vitg384_cresume_const15.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
# prepare_runtime_config.py rewrites the YAML's folder: field by appending the
# topology suffix (_n16g12_weak), and the trainer writes latest.pth.tar THERE.
# CKPT_DIR must match that suffixed folder or the chain's progress/completion
# check + lock + PARAMS look in the wrong place (latest.pth.tar never found ->
# chain thinks epoch 0 forever). The base config's folder is .../vitg384.
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_vitg384_cresume/vitg384_cr15_n16g12_weak
PARAMS=$CKPT_DIR/params-pretrain.yaml
SELF=$ROOT/scripts/vitg384_cresume_const15_chain.sh
LOCK=$CKPT_DIR/.training.lock
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

# Stage the runtime config into the ckpt dir as the authoritative params file.
# (Topology is already 16x12 in the base cfg, so this is effectively a copy, but
# keep the mechanism so every chained slice + the capacity job read one file.)
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
  echo "ViT-g CPT complete (epoch >= num_epochs). chain stops."
  rm -f "$LOCK"
  exit 0
fi

# Keep the chain alive: resubmit a successor unless one is already queued, OR a
# capacity job has STARTED running (the swap-over: once capacity is live it owns
# the run via the shared LOCK, so the debug-scaling chain should drain itself
# instead of churning no-op slices behind it). While a capacity job is only
# queued (could wait hours), the chain keeps making progress.
# NOTE: do NOT use `-W depend=afterany:$PBS_JOBID` — on Aurora PBS a walltime
# kill (-29) leaves the dependent stuck in unreleasable system-hold. The LOCK
# guard already serializes, so submit dependency-free; the successor backfills.
# qstat -u columns: 3=queue, 4=jobname, 10=state. Match jobname (col 4) — the
# queue col truncates to "debug-s*" so matching it is unreliable.
CAP_RUNNING=$(qstat -u $USER 2>/dev/null | awk '$4=="vitg_crcap" && $10=="R"' | wc -l)
DS_QUEUED=$(qstat -u $USER 2>/dev/null | awk '$4=="vitg_cr" && $10=="Q"' | wc -l)
if (( CAP_RUNNING >= 1 )); then
  echo "swap-over: capacity job running -> debug-scaling chain draining (no resubmit)"
elif (( DS_QUEUED >= 1 )); then
  echo "skip resubmit: $DS_QUEUED debug-scaling job already queued"
else
  # NB: resubmit MUST be non-fatal. Under `set -e`, a failed qsub (e.g. shared-
  # account "would exceed per-user Q limit") would otherwise abort THIS slice
  # before training even starts (observed: job 8574149 Exit 38). The watchdog/
  # orchestrator re-arms the chain if this resubmit is skipped, so failing soft
  # here is safe.
  NEXT_JOB=$(qsub $SELF 2>&1) || NEXT_JOB="(resubmit failed, watchdog will re-arm: $NEXT_JOB)"
  echo "Chained next job: $NEXT_JOB"
fi

# Lock guard: never train two slices against the same latest.pth.tar. This also
# protects against a capacity job and a debug-scaling slice colliding (both honor
# this lock since they share CKPT_DIR).
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
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export WDS_LOCAL_SLICING=1
# --- ViT-g optimizations (validated 2026-06-26: -32% iter-time, no loss cost) ---
export VJEPA_BF16_COMM=1
export VJEPA_DDP_BUCKET_MB=50
export VJEPA_EXIT_AFTER_CKPT=1  # 1h slice: exit after the 1 epoch we get, dont waste partial next
if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"
echo "OPT ENVS: VJEPA_BF16_COMM=$VJEPA_BF16_COMM VJEPA_DDP_BUCKET_MB=$VJEPA_DDP_BUCKET_MB"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT (per-node disjoint) ---"
mpiexec -n 16 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS \
        --local-root $LOCAL_DATA_ROOT \
        --num-nodes 16 --local-world-size 12 --workers 8
echo "--- staging complete ---"

mpiexec --pmi=pmix -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"
