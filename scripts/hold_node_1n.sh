#!/usr/bin/env bash
# Interactive-style NODE HOLD: reserves 1 debug node for ~1h and idles, so we can
# `mpiexec` test commands INTO it repeatedly (fast edit->run) instead of requeuing
# a fresh batch job per bug. Find the assigned node in the .OU log, then exec into it.
#
# Submit:  qsub scripts/hold_node_1n.sh
# Then:    tail the .OU for "HOLD NODE READY: <host> jobid=<id>", and run
#          commands via:  mpiexec --pmi=pmix -n 12 -ppn 12 ... (from a login shell,
#          PBS routes to the held allocation when you pass -l select against jobid),
#          OR simpler: ssh to <host> is not allowed; instead use `qsub`-less
#          `mpiexec` bound to this job by running the driver here. We instead poll a
#          COMMAND FILE so we can inject work without a second qsub.
#
#PBS -N hold1n
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -o pipefail
# NOTE: deliberately NOT `set -u` — `module load frameworks` references unset
# vars (ZSH_EVAL_CONTEXT) and would abort the hold immediately under -u.
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
CMD_DIR=/flare/ModCon/ngetty/logs/hold_${PBS_JOBID%%.*}
mkdir -p "$CMD_DIR"
HOST=$(head -n1 "$PBS_NODEFILE")
echo "HOLD NODE READY: $HOST jobid=$PBS_JOBID"
echo "CMD_DIR=$CMD_DIR  (write run_N.sh here; output -> run_N.out; touch STOP to exit)"

cd $ROOT
module load frameworks
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
# Aurora/CCL env so injected mpiexec commands inherit a sane environment.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1 CCL_WORKER_COUNT=1 CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi TMPDIR=/tmp OMP_NUM_THREADS=16
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95 FI_MR_CACHE_MONITOR=disabled
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export MASTER_ADDR=$HOST MASTER_PORT=29500
export http_proxy="http://proxy.alcf.anl.gov:3128" https_proxy="http://proxy.alcf.anl.gov:3128"

# Poll loop: run any run_*.sh dropped into CMD_DIR (once each). Each runs in the
# SERIALIZED: one injected run at a time; output -> run_N.out, a run_N.done
# marker is written with the rc when it finishes. Hang protection comes from the
# injected script's own `timeout`, NOT from backgrounding -- see the loop below.
seen=""
end=$(( $(date +%s) + 3500 ))
while [ "$(date +%s)" -lt "$end" ]; do
    [ -f "$CMD_DIR/STOP" ] && { echo "STOP seen, exiting hold."; break; }
    for c in "$CMD_DIR"/run_*.sh; do
        [ -e "$c" ] || continue
        case " $seen " in *" $c "*) continue;; esac
        seen="$seen $c"
        # SERIALIZE -- do not background. Distributed runs contend for the
        # rendezvous port AND for all tiles on the node, so two concurrent
        # injections produce garbage timings (or EADDRINUSE) rather than two
        # results. Per-run MASTER_PORT so a leftover socket cannot poison the
        # next. Same fix as 4b99560 on hold_nodes_daos.sh -- this copy never
        # got it.
        port=$(( 29500 + RANDOM % 500 ))
        echo "=== LAUNCH $c @ $(date)  MASTER_PORT=$port ==="
        MASTER_PORT=$port bash "$c" > "${c%.sh}.out" 2>&1
        echo "rc=$?" > "${c%.sh}.done"
        echo "=== DONE $(basename $c) rc=$(cat ${c%.sh}.done) @ $(date) ==="
    done
    sleep 5
done
echo "HOLD END: $(date) — waiting for any in-flight runs"
wait
echo "HOLD FULLY DONE: $(date)"
