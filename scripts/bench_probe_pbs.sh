#!/usr/bin/env bash
#PBS -N probe_bench
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -eo pipefail
ROOT=${PBS_O_WORKDIR:-/lus/flare/projects/ModCon/ngetty/vjepa2}
cd "$ROOT"
# NOTE: must NOT use `set -u` here — `module load frameworks` references unbound
# vars (ZSH_EVAL_CONTEXT) and dies under nounset. The working chains use -eo.
module load frameworks 2>/dev/null || module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_KVS_CONNECTION_TIMEOUT=600 CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1 CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216 FI_PROVIDER=cxi
export TMPDIR=/tmp OMP_NUM_THREADS=16
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
MA=$(head -n1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || hostname)
export MASTER_ADDR=$MA MASTER_PORT=29655 WORLD_SIZE=24
echo "=== BENCH FULL-DATA (sdpa on/off x batch) $(date) ==="
BS_LIST="2 4 8" SDPA_LIST="true false" BENCH_EPOCHS=2 bash "$ROOT/scripts/bench_probe_speed.sh" full metaraw
echo "=== BENCH done $(date) ==="
