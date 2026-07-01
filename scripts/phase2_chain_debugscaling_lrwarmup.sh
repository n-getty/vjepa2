#!/usr/bin/env bash
# Self-resubmitting phase-2 training loop, 1h walltime on debug-scaling.
#
# Continues the v2 (LR-WARMUP) experiment run: phase2_main_n16g12_weak_lrwarmup.
# This run differs from v1 ONLY in the optimizer LR schedule:
#   start_lr=5e-5, warmup=30 (epochs)  -> 30-epoch linear ramp to lr=5.25e-4,
#   then flat. (v1 is flat 5.25e-4 from step 0.)
# Everything else — data, masks, lambda_progressive ramp (iter 15k-30k =
# epoch 150-300), EMA, model — is identical to v1.
#
# debug-scaling is max_run=1/user, so only ONE debug-scaling job runs at a
# time PER USER. A LOCK GUARD prevents two chains training this SAME ckpt dir
# concurrently (e.g. a capacity backfill + this one) — would corrupt
# latest.pth.tar. Whichever job holds the lock trains; the other skips its
# training slice but keeps its own chain alive by resubmitting.
#
# Submit first instance with:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/phase2_chain_debugscaling_lrwarmup.sh
#
#PBS -N phase2_ds_lrwarmup
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PARAMS=/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak_lrwarmup/params-pretrain.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitl16_surg_vid_webdataset_single4/video-segments-256px-16f.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak_lrwarmup
SELF=$ROOT/scripts/phase2_chain_debugscaling_lrwarmup.sh
LOCK=$CKPT_DIR/.training.lock

mkdir -p $CKPT_DIR

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

# Pre-flight: stage runtime cfg into folder if missing (first invocation).
if [[ ! -f "$PARAMS" ]]; then
  echo "Staging runtime cfg -> $PARAMS"
  $PY $ROOT/scripts/prepare_runtime_config.py \
    $ROOT/configs/vitl16_surg_vid_webdataset_single4/video-segments-256px-16f.yaml \
    --root $ROOT --num-gpus 12 --num-nodes 16 --weak-scale \
    --folder-base /flare/ModCon/ngetty/checkpoints/surg_2_1_v1 > /dev/null
  cp $RUNTIME_CFG $PARAMS
fi

# Check current progress.
NUM_EPOCHS=$($PY -c "import yaml; print(yaml.safe_load(open('$PARAMS'))['optimization']['epochs'])")
CURRENT_EPOCH=0
if [[ -f $CKPT_DIR/latest.pth.tar ]]; then
  CURRENT_EPOCH=$($PY -c "import torch; print(torch.load('$CKPT_DIR/latest.pth.tar', map_location='cpu', weights_only=False).get('epoch', 0))" 2>/dev/null || echo 0)
fi
echo "progress: epoch $CURRENT_EPOCH / $NUM_EPOCHS"

if (( CURRENT_EPOCH >= NUM_EPOCHS )); then
  echo "phase 2 complete (epoch >= num_epochs). chain stops."
  rm -f "$LOCK"
  exit 0
fi

# Self-resubmit BEFORE training starts so chain survives crash/walltime.
# Respect debug-scaling max_queued — only resubmit if we don't already have
# one queued (max_run=1 means at most 1 R + some Q; keep just 1 in reserve).
DS_QUEUED=$(qstat -u $USER 2>/dev/null | awk '$3=="debug-sca" && $10=="Q"' | wc -l)
if (( DS_QUEUED >= 1 )); then
  echo "skip resubmit: $DS_QUEUED debug-scaling job already queued"
else
  NEXT_JOB=$(qsub -W depend=afterany:$PBS_JOBID $SELF)
  echo "Chained next job: $NEXT_JOB"
fi

# --- LOCK GUARD ------------------------------------------------------------
# Prevent concurrent training of this run by another chain (e.g. the capacity
# 8h chain targeting the same CKPT_DIR). The lock stores the holder's JOBID;
# if that job is still running (R), we skip training but keep our chain alive.
if [[ -f "$LOCK" ]]; then
  HOLDER=$(cat "$LOCK" 2>/dev/null || echo "")
  if [[ -n "$HOLDER" ]] && qstat "$HOLDER" 2>/dev/null | awk 'NR>2{print $5}' | grep -q '^R$'; then
    echo "LOCK held by running job $HOLDER — another chain is training v1."
    echo "Skipping training this slice to avoid latest.pth.tar collision."
    echo "JOB END (skipped): $(date)"
    exit 0
  else
    echo "stale lock from $HOLDER (not running) — taking over."
  fi
fi
echo "$PBS_JOBID" > "$LOCK"
# Release lock on any exit so a stale lock never blocks the chain forever.
trap 'rm -f "$LOCK"' EXIT

# Run training
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

# Slice WebDataset shards by local rank/world_size against the per-node /tmp copy.
export WDS_LOCAL_SLICING=1

if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29503
export WORLD_SIZE=192

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

# --- Per-node WebDataset shard staging --------------------------------------
# Each node stages only the union of shards its 12 local ranks will read, onto
# /tmp, so dataloader workers read from local tmpfs instead of contending on
# flare. See scripts/stage_node_shards.py.
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
