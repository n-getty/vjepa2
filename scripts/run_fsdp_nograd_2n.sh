#!/usr/bin/env bash
#PBS -N fsdp_ng_probe2n
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=2
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2; cd $ROOT
module load frameworks; export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_KVS_CONNECTION_TIMEOUT=600 CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216 FI_PROVIDER=cxi TMPDIR=/tmp OMP_NUM_THREADS=16
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95 FI_MR_CACHE_MONITOR=disabled CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export MASTER_ADDR=$(head -n1 $PBS_NODEFILE) MASTER_PORT=29711 WORLD_SIZE=24 LOCAL_WORLD_SIZE=12
echo "=== FSDP fwd+bwd probe (isolate first-step hang) ==="
timeout 300 mpiexec --pmi=pmix -n 24 -ppn 12 --cpu-bind depth --depth 16 \
    python $ROOT/scripts/fsdp_nograd_probe.py 2>&1 \
  | grep -viE "UserWarning|warnings.warn|comm_dev_uuids|CCL_WARN" \
  | grep -iE "r0 |wrapped|FORWARD|BACKWARD|PASSED|Error|Traceback|mesh" | tail -25
echo "probe rc=${PIPESTATUS[0]}"
