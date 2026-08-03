#!/bin/bash
# INTERACTIVE HOLD for HSDP/DAOS ablations — inject commands instead of requeuing.
#
# WHY
# ---
# Every ablation so far cost a fresh queue cycle to discover a one-line bug: an
# empty CCL_KVS_MODE that oneCCL rejects as an invalid enum (job 8731004, killed
# at iter 0), a `python: command not found` from loading only the daos module
# (job 8730443), a stale shared CSV read as this run's result. None of those
# needed 64 nodes or an hour to find. Hold a small allocation, drop scripts into
# CMD_DIR, read the output, fix, repeat — then spend the big block on a config
# that has actually executed.
#
# DIFFERS FROM scripts/hold_node_1n.sh: that one exports the DDP transport
# (CCL_PROCESS_LAUNCHER=pmix, CCL_ATL_TRANSPORT=mpi, CCL_KVS_MODE=mpi) plus the
# three "insurance" flags that findings 4e falsified. Both are wrong for this
# work: HSDP needs launcher=none + ofi with oneCCL's own KVS, and reusing the DDP
# env here would silently test a different configuration than the one we ship.
#
# USAGE
#   qsub -l select=2 scripts/hold_nodes_daos.sh        # cheap env/smoke checks
#   qsub -l select=64 scripts/hold_nodes_daos.sh       # real ablations
#   # then, from a login shell:
#   D=/flare/ModCon/ngetty/logs/hold_daos_<jobid>
#   cat > $D/run_1.sh <<'EOF'
#   mpiexec -n $WORLD -ppn 12 --cpu-bind depth --depth 16 --no-vni \
#       python -m app.main_dist_aurora --train_mode --fname ... --local_data_root $DAOS_MNT
#   EOF
#   # watch $D/run_1.out ; $D/run_1.done holds the rc ; touch $D/STOP to release
#
# Injected scripts inherit the full HSDP+DAOS env below, plus $WORLD, $DAOS_MNT,
# $MODELS_MNT, $NNODES, $PPN, $CMD_DIR.
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N holddaos
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=2
# 30 min, not 60: debug-scaling rejected a 2-node 60-min request outright
# ("Insufficient amount of resource: at_queue", jobs 8731089/8731106 died with no
# log at all) while the identical script at 30 min was accepted immediately. A
# 64-node 30-min job also ran fine, so the limit tracks WALLTIME here, not size.
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-vjepa_surg_wds}
MODELS_CONT=${DAOS_MODELS_CONT:-vjepa_models}
export DAOS_MNT=/tmp/${POOL}/${CONT}
export MODELS_MNT=/tmp/${POOL}/${MODELS_CONT}
export PPN=12
export NNODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l)
export WORLD=$(( NNODES * PPN ))
export CMD_DIR=/flare/ModCon/ngetty/logs/hold_daos_${PBS_JOBID%%.*}
mkdir -p "$CMD_DIR"

cd $ROOT
module use /soft/modulefiles
module load frameworks     # NOT just daos -- daos alone leaves no python on PATH
module load daos
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
# HSDP transport: oneCCL brings up its OWN KVS over CXI. Do NOT use the DDP
# pmix/mpi block here, and note mpiexec must omit --pmi=pmix to match.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_KVS_IFACE=hsn0
# UNSET, never empty: oneCCL rejects '' for this enum (killed job 8731004).
unset CCL_KVS_MODE CCL_KVS_USE_MPI_RANKS
export CCL_OP_SYNC=1
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-error}
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=${CCL_ALLREDUCE:-ring}
export CCL_CHUNK_SIZE=${CCL_CHUNK_SIZE:-16777216}
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export OMP_NUM_THREADS=16
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=$PPN
export FSDP_SHARDING=shard_grad_op
export VJEPA_NUM_WORKERS=${VJEPA_NUM_WORKERS:-0}
export TORCH_DIST_TIMEOUT_SECONDS=${TORCH_DIST_TIMEOUT_SECONDS:-3600}
export WDS_LOCAL_SLICING=0
# Falsified by findings 4e (env-diff 8643398): l0-free/l0-ext flat +-5 MiB over
# 74 iters with these removed. 5a re-adds them; trust the experiment.
unset CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD FI_MR_CACHE_MONITOR PYTORCH_ALLOC_CONF
unset LD_PRELOAD

export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=29500
export WORLD_SIZE=$WORLD

launch-dfuse.sh ${POOL}:${MODELS_CONT} || echo "WARN: launch-dfuse models failed"
launch-dfuse.sh ${POOL}:${CONT}        || echo "WARN: launch-dfuse corpus failed"
timeout 60 ls "$DAOS_MNT" >/dev/null 2>&1 && echo "DAOS corpus mounted: $DAOS_MNT" \
  || echo "WARN: $DAOS_MNT unresponsive"
timeout 60 ls "$MODELS_MNT" >/dev/null 2>&1 && echo "DAOS models mounted: $MODELS_MNT" \
  || echo "WARN: $MODELS_MNT unresponsive"

# Self-test the env that has bitten us, so a broken hold is obvious immediately
# rather than after an injected job fails 20 minutes later.
echo "=== env self-test ==="
command -v python >/dev/null && echo "  python: $(python -V 2>&1)" || echo "  python: *** MISSING ***"
echo "  CCL_KVS_MODE is $([ -z "${CCL_KVS_MODE+x}" ] && echo 'UNSET (correct)' || echo "SET='${CCL_KVS_MODE}' (WRONG)")"
echo "  nodes=$NNODES world=$WORLD master=$MASTER_ADDR"

echo
echo "HOLD READY: ${NNODES} nodes, ${WORLD} ranks, jobid=$PBS_JOBID"
echo "CMD_DIR=$CMD_DIR"
echo "  drop run_*.sh there; output -> run_*.out, rc -> run_*.done; touch STOP to release"

# Poll loop. Each injected script runs in the BACKGROUND so a hung command never
# blocks the hold or the queue behind it.
seen=""
end=$(( $(date +%s) + ${HOLD_SECONDS:-1700} ))
while [ "$(date +%s)" -lt "$end" ]; do
    [ -f "$CMD_DIR/STOP" ] && { echo "STOP seen, releasing hold."; break; }
    for c in "$CMD_DIR"/run_*.sh; do
        [ -e "$c" ] || continue
        case " $seen " in *" $c "*) continue;; esac
        seen="$seen $c"
        echo "=== LAUNCH $(basename $c) @ $(date) ==="
        ( bash "$c" > "${c%.sh}.out" 2>&1; echo "rc=$?" > "${c%.sh}.done" ) &
    done
    sleep 5
done
echo "HOLD END: $(date) — waiting for in-flight runs"
wait
echo "HOLD FULLY DONE: $(date)"
