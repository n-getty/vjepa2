#!/usr/bin/env bash
# Self-resubmitting phase-2 training loop, 8h walltime on capacity queue.
#
# Companion to scripts/phase2_chain_loop.sh (1h debug-scaling). Use this when:
#  - The 24h capacity job is stuck behind the 128-node queue cap,
#  - You want a more backfill-friendly walltime,
#  - Or you want a parallel chain so whichever slot frees first wins.
#
# Submit first instance with:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/phase2_chain_8h_capacity.sh
#
# Each invocation:
#  1. Self-resubmits with afterany dependency BEFORE training (chain survives crashes).
#  2. Checks latest.pth.tar epoch; exits if training is already complete.
#  3. Runs one 8h slice of phase-2 training.
#
#PBS -N phase2_lrwarmup_8h
#PBS -A AuroraGPT
#PBS -q capacity
#PBS -l select=16
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PARAMS=/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak_lrwarmup/params-pretrain.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitl16_surg_vid_webdataset_single4/video-segments-256px-16f.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak_lrwarmup
SELF=$ROOT/scripts/phase2_chain_8h_lrwarmup.sh

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
  exit 0
fi

# Self-resubmit BEFORE training starts so chain survives even on crash.
# afterany ensures next runs whether we exit 0 (walltime) or non-zero (crash).
# Respect capacity queue's max_queued=5 limit — skip resubmit if we're at 4 queued.
QUEUED_CT=$(qstat -u $USER 2>/dev/null | awk '$3=="capacity" && $10=="Q"' | wc -l)
if (( QUEUED_CT >= 4 )); then
  echo "skip resubmit: $QUEUED_CT capacity jobs already queued (max_queued=5)"
else
  NEXT_JOB=$(qsub -W depend=afterany:$PBS_JOBID $SELF)
  echo "Chained next job: $NEXT_JOB"
fi

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

# Tell WebDataset loader to slice by local rank/world_size, not global.
# Each node's /tmp holds exactly the union of its 12 local ranks' shards,
# so disjoint local slicing gives full coverage with no cross-node opens.
export WDS_LOCAL_SLICING=1

if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29502
export WORLD_SIZE=192

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

# --- Per-node WebDataset shard staging --------------------------------------
# Stage each node's disjoint slice of shards onto /tmp, so dataloader workers
# read from local tmpfs instead of contending on flare. The loader at
# src/datasets/webdataset.py:_make_stream computes urls[rank::world_size];
# the stager picks the union of those slices for each node.
# Free space check first: 16-node weak-scale slice is ~41 GiB/node out of
# /tmp's typical ~250 GiB headroom on Aurora.
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
