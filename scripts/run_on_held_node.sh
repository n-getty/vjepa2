#!/usr/bin/env bash
# Helper: ssh into the held node, set up env, run a 12-rank mpiexec probe.
# Usage:  bash scripts/run_on_held_node.sh <host> <mode>
#  e.g.   bash scripts/run_on_held_node.sh x4717c5s5b0n0 D
set -euo pipefail
host=${1:-$(qstat -f $(qstat -u $USER 2>/dev/null | awk '/vjepa_hold/{print $1}' | head -1 | cut -d. -f1) 2>/dev/null | awk '/exec_vnode/{print $3}' | tr -d '()' | cut -d: -f1)}
mode=${2:-B}
[[ -z "$host" ]] && { echo "no host"; exit 2; }

ssh -o BatchMode=yes "$host" bash -lc "'
source /etc/profile >/dev/null 2>&1
module -q load frameworks
# locate PBS_NODEFILE for the held job (lost across ssh)
nodefile=\$(ls /var/spool/pbs/aux/8526933* 2>/dev/null | head -1)
export PBS_NODEFILE=\${nodefile:-}
export MASTER_ADDR=\$(head -n1 \"\$PBS_NODEFILE\")
export MASTER_PORT=29501
echo \"PBS_NODEFILE=\$PBS_NODEFILE MASTER_ADDR=\$MASTER_ADDR\"
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
export OMP_NUM_THREADS=16
export TMPDIR=/tmp
export PYTHONFAULTHANDLER=1
cd /lus/flare/projects/ModCon/ngetty/vjepa2
mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python scripts/probe_ddp_xpu.py --mode $mode
'"
