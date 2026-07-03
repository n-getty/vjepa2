#!/usr/bin/env bash
# 1-NODE SMOKE of the 2B ViT-G @ 384 config on the debug queue (1h, 1 node).
# Purpose: validate PEAK GPU MEMORY at bs2/fpcs16/384 on a single 64GB PVC tile
# (the open risk for the 2B swap), confirm on-device checkpoint load (0 missing
# keys), and get a steady-state iter time. NOT a training run — no chain, no
# resubmit, no checkpoint saving.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/vitG384_smoke_debug.sh
#
#PBS -N vitG_smoke
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/g12_weak/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SMOKE_vitG384/smoke_n1g12
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR /flare/ModCon/ngetty/logs

echo "JOB START: $(date) PBS_JOBID=$PBS_JOBID"

# Stage a 1-node/12-GPU runtime config (rewrites folder: with the _n1g12 suffix).
$PY $ROOT/scripts/prepare_runtime_config.py \
    $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 1 --weak-scale > /dev/null
cp $RUNTIME_CFG $PARAMS
echo "staged runtime cfg -> $PARAMS"

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
# Match the real run's optimization stack so the memory number is representative.
export VJEPA_BF16_COMM=1
export VJEPA_DDP_BUCKET_MB=50

if [[ -f "${PBS_NODEFILE:-}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=12
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

export LOCAL_DATA_ROOT=/tmp/vjepa_data/${PBS_JOBID%%.*}
echo "--- staging shards to $LOCAL_DATA_ROOT ---"
mpiexec -n 1 -ppn 1 --cpu-bind none \
    python $ROOT/scripts/stage_node_shards.py \
        --params $PARAMS \
        --local-root $LOCAL_DATA_ROOT \
        --num-nodes 1 --local-world-size 12 --workers 8
echo "--- staging complete ---"

# Background xpu-smi sampler: record peak per-tile memory during the run.
MEMLOG=/flare/ModCon/ngetty/logs/vitG_smoke_${PBS_JOBID%%.*}_mem.log
( while true; do
    echo "=== $(date +%s) ==="
    xpu-smi dump -d -1 -m 18 2>/dev/null || xpu-smi stats -d 0 2>/dev/null | grep -i "memory used" || true
    sleep 10
  done ) > "$MEMLOG" 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT
echo "xpu-smi memory sampler -> $MEMLOG (pid $SMI_PID)"

mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT
echo "JOB END: $(date)"
