#!/usr/bin/env bash
# Probe-only (cache already exists): run ONE checkpoint's 256px probe on a single
# node (12 ranks). Usage: bash run_256_probeonly_onnode.sh <tag> <master_port>
set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
TAG="${1:?tag}"; PORT="${2:-29630}"
CFGD=$ROOT/configs/heads/sarrarp50/full_cached_256
PRB=$CFGD/${TAG}_probe.yaml
CACHE=/flare/ModCon/ngetty/surg_2_1_v2_final/probes/full_cache_256/$TAG
cd $ROOT && module load frameworks >/dev/null 2>&1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_OP_SYNC=1 CCL_WORKER_COUNT=1 FI_PROVIDER=cxi
export TMPDIR=/tmp OMP_NUM_THREADS=16
export MASTER_ADDR=$(hostname) MASTER_PORT=$PORT WORLD_SIZE=12
for sp in train val; do
  nm=$(ls $CACHE/$sp/rank_*/manifest.json 2>/dev/null | wc -l)
  [ "$nm" -ne 12 ] && { echo "[$TAG] CACHE INCOMPLETE ($sp $nm/12) -> abort"; exit 1; }
done
echo "[$TAG] PROBE-ONLY $(date)"
mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 python -m app.main_dist_aurora --train_mode --fname "$PRB" --params_path "$PRB"
echo "[$TAG] DONE $(date)"
