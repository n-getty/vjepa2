#!/usr/bin/env bash
# Run EMA probe on a held node. Usage:
#   bash scripts/run_ema_probe.sh <jobid> <variant>
# variant: V1|V2|V3|V4|noddp
set -euo pipefail
jobid=${1:?jobid required}
variant=${2:-V1}

host=$(qstat -f $jobid 2>&1 | awk '/exec_vnode/' | head -1 | tr -d '()' | cut -d= -f2 | tr '+' '\n' | head -1 | cut -d: -f1 | tr -d ' ')
[[ -z "$host" ]] && { echo "no host"; exit 2; }

ssh -o BatchMode=yes "$host" bash -lc "'
source /etc/profile >/dev/null 2>&1
module -q load frameworks
export PBS_NODEFILE=\$(ls /var/spool/pbs/aux/${jobid}* 2>/dev/null | head -1)
export MASTER_ADDR=\$(head -n1 \"\$PBS_NODEFILE\")
export MASTER_PORT=29505
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
export MPIR_CVAR_CH4_OFI_MAX_NICS=1
export OMP_NUM_THREADS=16
export TMPDIR=/tmp
export PYTHONFAULTHANDLER=1
cd /lus/flare/projects/ModCon/ngetty/vjepa2
mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \\
    python scripts/probe_ema_xpu.py $variant 2>&1 | grep -E \"r0|init dist|variant|param|Expected|If target\"
'"
