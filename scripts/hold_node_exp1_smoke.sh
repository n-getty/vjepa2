#!/usr/bin/env bash
# Interactive-style NODE HOLD for the Exp-1 head-capacity smoke test (frozen-probe
# diagnosis plan, Step 1). trip_exp1_attn_deep.yaml (PooledFeatureMultiTaskClassifier,
# num_probe_blocks=4) has NEVER run in this repo -- 3 full self-attention blocks over
# N=4608 tokens at D=1664 is quadratic in N, so we measure per-epoch cost on a hold
# before committing 3 seeds x 4 arms to a capacity-queue batch job.
#
# Submit:  qsub scripts/hold_node_exp1_smoke.sh
# Then:    tail the .OU for "HOLD NODE READY: <host> jobid=<id>", drop a run_N.sh
#          into CMD_DIR, tail run_N.out. Same command-file pattern as hold_node_1n.sh.
#
#PBS -N hold_exp1_smoke
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -o pipefail
# Deliberately NOT `set -u` -- module load references unset vars.
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
CMD_DIR=/flare/ModCon/ngetty/logs/hold_${PBS_JOBID%%.*}
mkdir -p "$CMD_DIR"
HOST=$(head -n1 "$PBS_NODEFILE")
echo "HOLD NODE READY: $HOST jobid=$PBS_JOBID"
echo "CMD_DIR=$CMD_DIR  (write run_N.sh here; output -> run_N.out; touch STOP to exit)"

cd $ROOT && module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_OP_SYNC=1 CCL_WORKER_COUNT=1 FI_PROVIDER=cxi
export TMPDIR=/tmp OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export MASTER_ADDR=$HOST

seen=""
end=$(( $(date +%s) + 3400 ))
while [ "$(date +%s)" -lt "$end" ]; do
    [ -f "$CMD_DIR/STOP" ] && { echo "STOP seen, exiting hold."; break; }
    for c in "$CMD_DIR"/run_*.sh; do
        [ -e "$c" ] || continue
        case " $seen " in *" $c "*) continue;; esac
        seen="$seen $c"
        # SERIALIZE -- do not background. Distributed runs contend for the
        # rendezvous port AND for all 12 tiles, so two concurrent injections
        # produce garbage timings (or EADDRINUSE) rather than two results.
        # Give each run its own MASTER_PORT so a leftover socket from the
        # previous run cannot poison the next; hang protection belongs in the
        # injected script's own `timeout`, not in backgrounding.
        # Same fix as 4b99560 on hold_nodes_daos.sh -- this copy never got it,
        # and job 8775574 launched two profiling arms in the same second.
        port=$(( 29500 + RANDOM % 500 ))
        echo "=== LAUNCH $c @ $(date)  MASTER_PORT=$port ==="
        MASTER_PORT=$port bash "$c" > "${c%.sh}.out" 2>&1
        echo "rc=$?" > "${c%.sh}.done"
        echo "=== DONE $(basename $c) rc=$(cat ${c%.sh}.done) @ $(date) ==="
    done
    sleep 5
done
echo "HOLD END: $(date) -- waiting for any in-flight runs"
wait
echo "HOLD FULLY DONE: $(date)"
