#!/usr/bin/env bash
# Two SAR feature-cache EXPORTS in parallel on a 2-node debug job, 1 per node:
#   node0 -> cd64f_e5 (cooldown epoch5)   node1 -> cr34 (cresume epoch34)
# Export is inference-only (~19 min each), so it fits the debug 1h cap with margin.
# The long 20-epoch PROBE (~2.3h) runs separately on capacity reading these caches
# (qsub -v TAG=cd64f_e5 scripts/vitg_probe_capacity.sh ; same for cr34).
#
# Submit:
#   qsub scripts/vitg_export2_debug.sh
#
#PBS -N vitg_export2
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
CFGD=$ROOT/configs/heads/sarrarp50/full_cached_384
cd $ROOT && module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_OP_SYNC=1 CCL_WORKER_COUNT=1 FI_PROVIDER=cxi
export TMPDIR=/tmp OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"

# Split the 2-node allocation into two single-node host lists.
mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
echo "nodes: ${NODES[*]}"
if (( ${#NODES[@]} < 2 )); then echo "ERROR: need 2 nodes, got ${#NODES[@]}"; exit 1; fi
N0=${NODES[0]}; N1=${NODES[1]}

# Each export is its own 12-rank single-node MPI world pinned to one host.
# MASTER differs per task so the two CCL worlds don't collide.
#
# CRITICAL: the trainer derives world_size via src/utils/distributed.py's PALS
# fallback = PALS_LOCAL_SIZE (12) * (#nodes in PBS_NODEFILE). Under a 2-node PBS
# job that yields 24, so each 12-rank export's rendezvous waits forever for 24
# clients (observed: "12/24 clients joined" -> 301s timeout -> SIGTERM/rc=143).
# Fix: hand each export a PRIVATE single-host nodefile so the fallback computes
# 12*1 = 12. (mpiexec --hosts limits the launch; the nodefile fixes the count.)
export_on() {  # $1=host  $2=cfg  $3=masterport
  local host="$1" cfg="$2" port="$3"
  local nf="/tmp/nodefile_${PBS_JOBID%%.*}_${host%%.*}"
  echo "$host" > "$nf"
  PBS_NODEFILE="$nf" MASTER_ADDR="$host" MASTER_PORT="$port" WORLD_SIZE=12 \
  mpiexec --pmi=pmix -n 12 -ppn 12 --hosts "$host" --cpu-bind depth --depth 16 \
      python -m app.main_dist_aurora --train_mode \
          --fname "$cfg" --params_path "$cfg"
}

echo "[export2] START $(date)"
export_on "$N0" "$CFGD/cd64f_e5_export.yaml" 29611 > /flare/ModCon/ngetty/logs/export2_cd64f_e5.$PBS_JOBID.log 2>&1 &
PID0=$!
export_on "$N1" "$CFGD/cr34_export.yaml"     29612 > /flare/ModCon/ngetty/logs/export2_cr34.$PBS_JOBID.log 2>&1 &
PID1=$!
echo "[export2] cd64f_e5 pid=$PID0 (node $N0) | cr34 pid=$PID1 (node $N1)"
wait $PID0; R0=$?
wait $PID1; R1=$?
echo "[export2] cd64f_e5 rc=$R0 | cr34 rc=$R1"

# Gate: report manifest completeness per tag.
for TAG in cd64f_e5 cr34; do
  C=/flare/ModCon/ngetty/surg_2_1_v2_final/probes/full_cache_384/$TAG
  for sp in train val; do
    nm=$(ls $C/$sp/rank_*/manifest.json 2>/dev/null | wc -l)
    echo "[export2] $TAG $sp manifests: $nm/12"
  done
done
echo "[export2] DONE $(date)"
