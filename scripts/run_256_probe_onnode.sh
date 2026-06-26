#!/usr/bin/env bash
# Run ONE checkpoint's 256px export->probe on a SINGLE node (12 ranks), driven
# interactively on a held node. Usage (ON the node, abs path):
#   bash /lus/.../scripts/run_256_probe_onnode.sh <tag> <master_port>
# tag in: v3_e19 metaraw v3_e9 ; uses configs/heads/sarrarp50/full_cached_256/
set -uo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
TAG="${1:?tag}"; PORT="${2:-29620}"
CFGD=$ROOT/configs/heads/sarrarp50/full_cached_256
EXP=$CFGD/${TAG}_export.yaml; PRB=$CFGD/${TAG}_probe.yaml
CACHE=/flare/ModCon/ngetty/surg_2_1_v2_final/probes/full_cache_256/$TAG
cd $ROOT && module load frameworks >/dev/null 2>&1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_OP_SYNC=1 CCL_WORKER_COUNT=1 FI_PROVIDER=cxi
export TMPDIR=/tmp OMP_NUM_THREADS=16
export MASTER_ADDR=$(hostname) MASTER_PORT=$PORT WORLD_SIZE=12
run(){ mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 python -m app.main_dist_aurora --train_mode --fname "$1" --params_path "$1"; }
echo "[$TAG] EXPORT $(date)"; run "$EXP"
# verify cache complete (12/12 both splits) before probe
for sp in train val; do
  nm=$(ls $CACHE/$sp/rank_*/manifest.json 2>/dev/null | wc -l)
  echo "[$TAG] $sp manifests: $nm/12"
  [ "$nm" -ne 12 ] && { echo "[$TAG] CACHE INCOMPLETE -> abort probe"; exit 1; }
done
echo "[$TAG] PROBE $(date)"; run "$PRB"
echo "[$TAG] DONE $(date)"
