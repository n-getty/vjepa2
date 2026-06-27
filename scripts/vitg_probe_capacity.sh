#!/usr/bin/env bash
# ViT-g @ 384 cached asformer probe as a SINGLE-NODE capacity batch job.
# Runs export (build 384 feature cache, world_size=12 -> 1 node) then probe.
# Usage: qsub -v TAG=metaraw scripts/vitg_probe_capacity.sh
#   TAG selects configs/heads/sarrarp50/full_cached_384/${TAG}_{export,probe}.yaml
# Default TAG=metaraw (the Meta-raw ViT-g downstream baseline anchor).
#
# The 384 cache is ~3.2x the 256 cache (~490G for full metaraw); disk has TBs free.
# Export is inference-only (no DDP grad) so the bigger ViT-g fits 12 tiles fine.
#
#PBS -N vitg_probe
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
TAG="${TAG:-metaraw}"
CFGD=$ROOT/configs/heads/sarrarp50/full_cached_384
EXP=$CFGD/${TAG}_export.yaml
PRB=$CFGD/${TAG}_probe.yaml
CACHE=/flare/ModCon/ngetty/surg_2_1_v2_final/probes/full_cache_384/$TAG
cd $ROOT && module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_OP_SYNC=1 CCL_WORKER_COUNT=1 FI_PROVIDER=cxi
export TMPDIR=/tmp OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
if [[ -f "${PBS_NODEFILE:-}" ]]; then MASTER_ADDR=$(head -n1 "$PBS_NODEFILE"); else MASTER_ADDR=$(hostname); fi
export MASTER_ADDR MASTER_PORT=29600 WORLD_SIZE=12

run(){ mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
        python -m app.main_dist_aurora --train_mode --fname "$1" --params_path "$1"; }

echo "[$TAG] EXPORT $(date)"; run "$EXP"
# Gate: require 12/12 manifests in both splits before probing.
for sp in train val; do
  nm=$(ls $CACHE/$sp/rank_*/manifest.json 2>/dev/null | wc -l)
  echo "[$TAG] $sp manifests: $nm/12"
  [ "$nm" -ne 12 ] && { echo "[$TAG] CACHE INCOMPLETE -> abort probe"; exit 1; }
done
echo "[$TAG] PROBE $(date)"; run "$PRB"
echo "[$TAG] DONE $(date)"
