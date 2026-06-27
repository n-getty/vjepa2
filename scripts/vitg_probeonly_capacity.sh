#!/usr/bin/env bash
# ViT-g @ 384 cached asformer PROBE-ONLY (cache already exported), single node.
# Use when the export cache exists and only the head-training probe must (re)run
# — e.g. the first metaraw run hit the 2h walltime mid-probe at epoch 13.
# Usage: qsub -v TAG=metaraw scripts/vitg_probeonly_capacity.sh
#
# 384 probe is ~8min/epoch (bigger encoder + 3.2x tokens vs 256); patience=6 so
# convergence is ~epoch 19-22 = up to ~3h of head training. Give 4h walltime.
#
#PBS -N vitg_prbonly
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
TAG="${TAG:-metaraw}"
CFGD=$ROOT/configs/heads/sarrarp50/full_cached_384
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
export MASTER_ADDR MASTER_PORT=29601 WORLD_SIZE=12

# Gate: cache must be complete (12/12 both splits) before probing.
for sp in train val; do
  nm=$(ls $CACHE/$sp/rank_*/manifest.json 2>/dev/null | wc -l)
  echo "[$TAG] $sp manifests: $nm/12"
  [ "$nm" -ne 12 ] && { echo "[$TAG] CACHE INCOMPLETE -> abort (run vitg_probe_capacity.sh to export first)"; exit 1; }
done
echo "[$TAG] PROBE-ONLY $(date)"
mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode --fname "$PRB" --params_path "$PRB"
echo "[$TAG] DONE $(date)"
