#!/usr/bin/env bash
# Self-resubmitting phase-2 training loop for debug-scaling 1h walltime.
#
# At start of every job: read latest.pth.tar epoch. If < num_epochs target,
# resubmit ourselves AT THE START so the chain continues even if this job's
# training crashes. Then train. Saved snapshots accumulate every save_every_freq.
#
# Submit first instance with:
#   qsub -A AuroraGPT -q debug-scaling -l select=16 -l walltime=01:00:00 \
#        -l filesystems=home:flare scripts/phase2_chain_loop.sh
#
#PBS -N phase2_chain
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PARAMS=/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak/params-pretrain.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/n16g12_weak/configs/vitl16_surg_vid_webdataset_single4/video-segments-256px-16f.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak

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
NEXT_JOB=$(qsub -A AuroraGPT -q debug-scaling \
    -l select=16 -l walltime=01:00:00 \
    -l filesystems=home:flare \
    -W depend=afterany:$PBS_JOBID \
    $ROOT/scripts/phase2_chain_loop.sh)
echo "Chained next job: $NEXT_JOB"

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

if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=192

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

mpiexec --pmi=pmix -n 192 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS

echo "JOB END: $(date)"
