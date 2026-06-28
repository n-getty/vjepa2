#!/usr/bin/env bash
# Augmented-head probe: LIVE encoder (no cache) + training=True augmentation on
# the train split. 4 nodes x 12 tiles. Trains the ASFormer head only (encoder
# frozen) but with per-epoch augmentation -> better-regularized head than the
# deterministic cached probe. Usage: qsub -v TAG=e19_aug scripts/run_aug_probe_capacity.sh
#PBS -N vitg_augprobe
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=4
#PBS -l walltime=01:30:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
TAG="${TAG:?must qsub -v TAG=}"
PRB=$ROOT/configs/heads/sarrarp50/full_aug_384/${TAG}_probe.yaml
cd $ROOT && module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_OP_SYNC=1 CCL_WORKER_COUNT=1 FI_PROVIDER=cxi
export CCL_KVS_CONNECTION_TIMEOUT=600 CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216
export TMPDIR=/tmp OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128" https_proxy="http://proxy.alcf.anl.gov:3128"
if [[ -f "${PBS_NODEFILE:-}" ]]; then MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); else MASTER_ADDR=$(hostname); fi
export MASTER_ADDR MASTER_PORT=29610 WORLD_SIZE=48
echo "[augprobe $TAG] $(date) cfg=$PRB"
mpiexec --pmi=pmix -n 48 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode --fname "$PRB" --params_path "$PRB"
echo "[augprobe $TAG] DONE $(date)"
