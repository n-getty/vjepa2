#!/bin/bash
# BARRIER-INSTRUMENTED SCALING LADDER -- localize the multi-node efficiency loss.
#
# WHY THIS EXISTS
# ---------------
# We have no scaling curve. What we had was a table of four runs at four node
# counts, and on 2026-08-06 its baseline row was found to be invalid: the row
# labelled "1n" was `LRAB_vitG384_v2/arm_B_lr6e5` from
# scripts/vitG384_v2_lr_ab_32n.sh, which is `#PBS -l select=32` running
# `WORLD_SIZE=192 mpiexec -n 192 -ppn 12`. Sixteen nodes, not one. Worse, only
# 12 of its 192 rank CSVs were written, so "max-over-ranks" covered 6% of ranks
# -- the exact sampling error scripts/scaling_efficiency.py:7-31 exists to
# prevent -- and its mpiexec was Killed. Every efficiency percentage computed
# against that row is void.
#
# Two further variables were uncontrolled across the surviving rows:
#
#   num_workers   aurora_hsdp_env.sh forces VJEPA_NUM_WORKERS=0, overriding the
#                 config's 2, and the RESOLVED value was never logged -- it is
#                 unrecoverable from a finished run's own artifacts. Recovered
#                 from launchers: the "fast dataload" run was nw=2, all the slow
#                 ones nw=0. docs/vitG_2B_HSDP_findings.md:25-29 already measured
#                 the pair on this model: nw=2 -> dataload 0.00 s at p50/p90/max;
#                 nw=0 -> p50 0.96 s. That IS the ~1 s "dataload at scale". But
#                 nw=0 only means decode runs INLINE instead of prefetched, so
#                 the cost moved into a visible column. Whether it is also NET
#                 lost time is what the nw arm below settles.
#   run length    30-iteration shakeouts vs multi-thousand-iteration production,
#                 with warmup still descending at iteration 50 at 256n.
#
# NOT the axis: staged-vs-DAOS. All three historical runs read /tmp with
# WDS_LOCAL_SLICING=1. This ladder holds DAOS constant and does not test it.
#
# DESIGN
#   - Every rung in ONE allocation, back to back, same nodes, same fabric hour.
#     ccl_knob_sweep.sh:37-42 records that across separate jobs the fabric-hour
#     term swamps the effect being measured. Concurrent rungs would be faster but
#     would inject cross-rung fabric traffic into a study ABOUT fabric behaviour.
#   - A rung is `<nodes>` or `<nodes>:nw<N>`, so num_workers is a first-class arm
#     rather than a hidden constant. nw>0 is a HAZARD arm: train.py:207-211
#     forces 0 under HSDP because forking DataLoader workers after
#     init_device_mesh can inherit broken xccl state and deadlock on the first
#     batch (PRISM's documented failure mode). Run it LAST; a hang is a result.
#   - VJEPA_SCALE_PROBE=1 everywhere. target_encoder is HSDP-wrapped, so the
#     iteration's first collective is an FSDP all-gather INSIDE forward: rank
#     skew from dataload is paid as apparent forward time. The probe's pre-step
#     barrier moves it into its own column (max-over-ranks barrier-ms = skew).
#   - No checkpoint load. Every rank reads the 22.8 GB .pt independently;
#     measured process-start -> iter-0 is 5m26s at 64n and 8m57s at 256n, which
#     would eat the 1 h cap in startup alone. Random init has identical shapes
#     and FLOPs. *** LOSS FROM THIS LADDER IS MEANINGLESS. Do not report it. ***
#   - Per-rung rc is captured and logged but NEVER fatal, and each rung gets its
#     own stall watchdog. One hung rung must not cost the whole allocation.
#
# USAGE (debug-scaling caps at 1 h and allows ONE job per user, running OR
# queued, so these are three sequential submissions -- not concurrent):
#
#   L0: qsub -l select=16 -v VJEPA_LADDER_RUNGS="16 16:probe0" scripts/scaling_ladder.sh
#   L1: qsub -l select=16 -v VJEPA_LADDER_RUNGS="1 2 4 8 16"   scripts/scaling_ladder.sh
#   L2: qsub -l select=32 -v VJEPA_LADDER_RUNGS="16 32"        scripts/scaling_ladder.sh
#   L3: qsub -l select=64 -v VJEPA_LADDER_RUNGS="16 64 16:nw2" scripts/scaling_ladder.sh
#
# L0 is the PROBE-OVERHEAD CHECK and gates the rest: if probe-ON and probe-OFF
# do not have overlapping IQRs, the ladder is measuring its own barrier and the
# design needs revising before the L1-L3 debug-scaling slots are spent.
#
# A rung is <nodes>[:nw<N>][:probe<0|1>], fields in any order. Unknown fields are
# rejected, not ignored -- a typo'd arm that silently ran the default config would
# be indistinguishable from a real null result.
#
# 16n repeats in all three as a CROSS-JOB ANCHOR. It is the only thing that can
# detect fabric drift between allocations; if the anchor moves by more than its
# IQR between jobs, cross-job comparisons are void and only within-job rungs are
# usable. Analyze with: scripts/scaling_efficiency.py --ladder <OUTROOT>
#
# NOTE: no `set -u` (memory set-u-module-load-trap: Lmod init trips it).
#
#PBS -N scaleladder
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-vjepa_surg_wds}
MODELS_CONT=${DAOS_MODELS_CONT:-vjepa_models}
DAOS_MNT=/tmp/${POOL}/${CONT}
MODELS_MNT=/tmp/${POOL}/${MODELS_CONT}
PPN=12
NNODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l); NNODES=${NNODES:-16}

# Rungs, in order. Default is the L1 set; override with -v VJEPA_LADDER_RUNGS.
RUNGS=${VJEPA_LADDER_RUNGS:-"1 2 4 8 16"}
# Per rung. 80 iterations gives ~60 post-warmup at the small rungs. This is a
# COMPROMISE and a known weakness: the "200 iterations past warmup" bar cannot be
# met at 64n inside a 1 h cap (200 x 23 s = 77 min for that rung alone). It is
# enough for the trend column to say whether the warmup decay has COMPLETED, not
# enough to guarantee it has. If the 64n trend is still falling at iteration 80,
# the follow-up is a dedicated long 64n rung -- not a reinterpretation of this one.
LADDER_IPE=${VJEPA_LADDER_IPE:-80}
# Per-rung override, e.g. "64=60 16:nw2=40", so the expensive rungs can be
# shortened without shortening the cheap ones.
LADDER_IPE_OVERRIDES=${VJEPA_LADDER_IPE_OVERRIDES:-""}
# lbA (bs=2), not lbA8 (bs=1): lbA is the promoted production config
# (memory aurora-throughput-recipe), and a scaling ladder that does not measure
# the config we actually run answers a question nobody asked. bs=2 fits at 1n
# (memory ckpt-and-bs-are-independent), so the same config runs at every rung.
CFG_NAME=${VJEPA_CFG_NAME:-vitG384_lbA}
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
TAG="${PBS_JOBID%%.*}"; TAG="${TAG:-manual}"
OUTROOT=${VJEPA_LADDER_OUTROOT:-/flare/ModCon/ngetty/checkpoints/scaling_ladder/${TAG}}
JOBTMP=$(mktemp -d /tmp/ladder.XXXXXX)
# Distinct ports per rung: consecutive mpiexec worlds in one allocation on the
# same head node would otherwise collide on the rendezvous socket while the
# previous world's TIME_WAIT entries are still held.
PORT_BASE=${VJEPA_LADDER_PORT_BASE:-29700}
# Per-rung deadlines. Startup here is rendezvous only (no 22.8 GB checkpoint),
# so FIRST_ITER is far tighter than the 1800 s the 256n production launcher uses.
STALL_DEADLINE=${VJEPA_LADDER_STALL_DEADLINE:-420}
FIRST_ITER_DEADLINE=${VJEPA_LADDER_FIRST_ITER_DEADLINE:-900}
mkdir -p "$OUTROOT" /flare/ModCon/ngetty/logs

echo "JOB START $(date) PBS_JOBID=$PBS_JOBID  allocation=${NNODES}n x ${PPN}"
echo "rungs: $RUNGS   ipe=$LADDER_IPE (overrides: '${LADDER_IPE_OVERRIDES}')"
echo "outroot: $OUTROOT"

cd $ROOT
module use /soft/modulefiles
module load frameworks
module load daos
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
# The measured HSDP/DAOS recipe. Sourcing it is what holds the CCL transport, FI
# block, WDS_LOCAL_SLICING=0 and VJEPA_NUM_WORKERS=0 fixed across every rung --
# the ladder's whole value is that only ONE variable moves per comparison.
# Overrides go BELOW the source.
source $ROOT/scripts/lib/aurora_hsdp_env.sh
export LOCAL_WORLD_SIZE=$PPN
export VJEPA_TRUE_ACCUM=${VJEPA_TRUE_ACCUM:-1}
export VJEPA_SCALE_PROBE=1          # the point of this job
export VJEPA_ITER_WATCHDOG_S=${VJEPA_ITER_WATCHDOG_S:-600}
# The global-batch assertion in vitG384_256n_daos.sh hard-fails unless
# world*bs*accum matches the config's derived value, which only 256n satisfies.
# Correct to skip HERE and only here: this job measures s/iter, not schedule
# fidelity, and it does not train anything worth keeping.
export VJEPA_SKIP_GB_CHECK=1

timeout 600 launch-dfuse.sh ${POOL}:${MODELS_CONT} || { echo "FATAL: launch-dfuse models"; exit 1; }
timeout 600 launch-dfuse.sh ${POOL}:${CONT}        || { echo "FATAL: launch-dfuse corpus"; exit 1; }
timeout 60 ls "$DAOS_MNT"   >/dev/null 2>&1 || { echo "FATAL: $DAOS_MNT unresponsive"; exit 1; }
timeout 60 ls "$MODELS_MNT" >/dev/null 2>&1 || { echo "FATAL: $MODELS_MNT unresponsive"; exit 1; }
echo "DAOS mounted (corpus + models)"

# --weak-scale keeps per-rank batch FIXED as nodes grow, which is the only
# meaningful setting for a per-tile scaling curve: every rung must do identical
# per-rank work or the throughput comparison is between different workloads.
# Called once for the allocation -- under weak scaling batch_size does not depend
# on node count, and the config's `nodes:` key is read only on the submit path
# (app/main_dist_aurora.py:284), never under --train_mode.
$PY $ROOT/scripts/prepare_runtime_config.py \
  $BASE_CFG --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale > /dev/null || exit 1
RUNTIME_CFG=$ROOT/.runtime_configs/n${NNODES}g${PPN}_weak/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
[ -r "$RUNTIME_CFG" ] || { echo "FATAL: runtime cfg missing: $RUNTIME_CFG"; exit 1; }

# Per-rung stall watchdog. Same shape as vitG384_256n_daos.sh:297-331, but scoped
# to ONE rung: it must not outlive its rung or it would kill the next one, so it
# is started before mpiexec and killed after.
#
# Returns the pid in the GLOBAL `RUNG_WD_PID`, never on stdout. This is not
# style. `pid=$(start_rung_watchdog ...)` -- the obvious spelling, and what this
# function did through job 8739712 -- silently disarms the watchdog: command
# substitution reads until the pipe closes, and the backgrounded subshell holds
# that pipe open for as long as it lives, so the caller blocks until the subshell
# EXITS. At rung start no `app.main_dist_aurora` is running yet, so the first
# pgrep below fails 30 s later, the subshell exits 0, and the substitution hands
# back the pid of a corpse. Job 8739712 ran both rungs completely unwatched and
# burned 25 min of its slot on a hang nothing reaped. Assign from $! instead.
#
# The grace period exists for the same reason: mpiexec needs time to start python
# on every node, and a watchdog whose liveness test is "is the trainer running"
# must not run that test before the trainer can possibly exist.
RUNG_WD_PID=""
WD_START_GRACE=${VJEPA_LADDER_WD_START_GRACE:-180}
start_rung_watchdog () {
    local csv="$1" tag="$2" diag="$3"
    mkdir -p "$diag"
    (
        start=$(date +%s); last_rows=-1; last_change=$start
        sleep $WD_START_GRACE
        while true; do
            sleep 30
            pgrep -f "app.main_dist_aurora" >/dev/null 2>&1 || exit 0
            now=$(date +%s); rows=0
            [ -f "$csv" ] && rows=$(awk -F, '$2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$csv" 2>/dev/null)
            if [ "${rows:-0}" -gt 0 ]; then
                if [ "$rows" -ne "$last_rows" ]; then last_rows=$rows; last_change=$now; fi
                if [ $((now-last_change)) -gt $STALL_DEADLINE ]; then
                    echo "WATCHDOG[$tag]: STALL -- no new iters for ${STALL_DEADLINE}s at row $rows." >&2
                    echo "jobid=$PBS_JOBID rung=$tag last_csv_row=$(tail -1 "$csv" 2>/dev/null)" \
                        > "$diag/stall_info.txt"
                    # SIGUSR1 -> the armed faulthandler dumps each rank's stacks and
                    # CONTINUES, so we learn which collective is blocked before the kill.
                    mpiexec -n $NNODES -ppn 1 --cpu-bind none --no-vni \
                        bash -c 'pkill -USR1 -f app.main_dist_aurora 2>/dev/null; true' >/dev/null 2>&1 || true
                    sleep 20
                    pkill -9 -f "app.main_dist_aurora"; exit 1
                fi
            elif [ $((now-start)) -gt $FIRST_ITER_DEADLINE ]; then
                echo "WATCHDOG[$tag]: NO FIRST ITER within ${FIRST_ITER_DEADLINE}s." >&2
                echo "jobid=$PBS_JOBID rung=$tag no_first_iter" > "$diag/nofirstiter_info.txt"
                mpiexec -n $NNODES -ppn 1 --cpu-bind none --no-vni \
                    bash -c 'pkill -USR1 -f app.main_dist_aurora 2>/dev/null; true' >/dev/null 2>&1 || true
                sleep 20
                pkill -9 -f "app.main_dist_aurora"; exit 1
            fi
        done
    ) &
    RUNG_WD_PID=$!
}

run_rung () {
    local spec=$1
    # spec = <nodes>[:nw<N>][:probe0|:probe1]  -- colon-separated, order-free.
    # probe0 exists for the PROBE-OVERHEAD CHECK that gates this whole study: the
    # barrier is supposed to be free (it sits where ranks already synchronize at
    # forward's first FSDP all-gather), but if it is not, the ladder is measuring
    # the probe. Run `16 16:probe0` in one allocation and require overlapping IQRs.
    local R="${spec%%:*}" nw="$VJEPA_NUM_WORKERS" probe="$VJEPA_SCALE_PROBE"
    local rest="${spec#*:}" fld
    if [ "$rest" != "$spec" ]; then
        local IFS=:
        for fld in $rest; do
            case "$fld" in
                nw*)    nw="${fld#nw}" ;;
                probe*) probe="${fld#probe}" ;;
                *) echo "  rung $spec: unknown field '$fld'"; return 1 ;;
            esac
        done
    fi
    local name="n${R}_nw${nw}"
    [ "$probe" = "1" ] || name="${name}_probe${probe}"
    if [ "$R" -gt "$NNODES" ]; then
        echo "SKIP rung $spec: needs $R nodes, allocation has $NNODES"; return 0
    fi
    local W=$(( R * PPN ))
    local dir="$OUTROOT/$name"
    local nf="$JOBTMP/nodefile_${name}"
    local params="$dir/params.yaml"
    mkdir -p "$dir"

    # Private nodefile: the first R nodes of the allocation. Same pattern as
    # scaling/fanout_pbs.py:63-86. Paired with an explicit WORLD_SIZE, which
    # src/utils/distributed.py:146-158 gives PRECEDENCE over PMI's SIZE -- that
    # is what lets a sub-world of the allocation bootstrap correctly, and
    # app/vjepa_2_1/hsdp.py:52-93 then derives num_nodes = world_size /
    # local_world_size, so the (replicate, shard) mesh is right for the rung.
    sort -u "$PBS_NODEFILE" | sed -n "1,${R}p" > "$nf"
    local got=$(wc -l < "$nf")
    [ "$got" -eq "$R" ] || { echo "SKIP rung $spec: nodefile has $got of $R"; return 0; }

    local ipe=$LADDER_IPE
    for ov in $LADDER_IPE_OVERRIDES; do
        [ "${ov%%=*}" = "$spec" ] && ipe="${ov##*=}"
    done

    cp "$RUNTIME_CFG" "$params" || return 1
    $PY - "$params" "$dir" "$ipe" <<'PY'
import sys, yaml
p, d, ipe = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = yaml.safe_load(open(p))
c["folder"] = d
c["optimization"]["ipe"] = ipe
c["optimization"]["epochs"] = 1
meta = c.setdefault("meta", {})
# No checkpoint: 22.8 GB read per rank costs 5-9 min of a 60 min slot and this
# job measures s/iter, not learning. Random init is FLOP-identical.
meta["load_checkpoint"] = False
meta["pretrain_checkpoint"] = None
meta["read_checkpoint"] = None
# save_every_freq high enough that no rung writes a checkpoint: 3072-rank
# checkpoint writes are minutes of the slot and nothing here is worth keeping.
meta["save_every_freq"] = 10**6
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
PY
    [ $? -eq 0 ] || { echo "  rung $spec: config rewrite FAILED"; return 1; }

    echo "===== RUNG $name : ${R}n x ${PPN} = ${W} ranks, ipe=$ipe, num_workers=$nw, scale_probe=$probe ====="
    # Sets RUNG_WD_PID -- see the note on start_rung_watchdog for why this must
    # not be a command substitution. Verify it armed: a silently-dead watchdog is
    # exactly the failure that cost job 8739712 half its slot, and it is
    # invisible unless something checks.
    start_rung_watchdog "$dir/log_r0.csv" "$name" "$dir/hang_diag"
    local wd_pid=$RUNG_WD_PID
    if kill -0 "$wd_pid" 2>/dev/null; then
        echo "  watchdog armed pid=$wd_pid (grace ${WD_START_GRACE}s, stall ${STALL_DEADLINE}s, first-iter ${FIRST_ITER_DEADLINE}s)"
    else
        echo "  WARNING: watchdog for $name did NOT arm -- rung runs unwatched" >&2
    fi
    local t0=$(date +%s)
    # --no-vni: DAOS RPCs fail NA_HOSTUNREACH without it.
    # -o/-e per rank: never funnel 3072 ranks' stdout through the head node,
    #   which also serves the rendezvous store and DAOS keepalives (job 8730678).
    # No --pmi=pmix: that is the DDP transport; HSDP needs launcher=none + ofi.
    PBS_NODEFILE="$nf" \
    MASTER_ADDR=$(head -n1 "$nf") \
    MASTER_PORT=$(( PORT_BASE + R )) \
    WORLD_SIZE=$W \
    VJEPA_NUM_WORKERS=$nw \
    VJEPA_SCALE_PROBE=$probe \
    mpiexec -n $W -ppn $PPN --hostfile "$nf" --cpu-bind depth --depth 16 --no-vni \
        -o "$dir/rank.%r.out" -e "$dir/rank.%r.err" \
        python -m app.main_dist_aurora --train_mode \
            --fname "$params" --params_path "$params" \
            --local_data_root "$DAOS_MNT"
    local rc=$?
    kill "$wd_pid" 2>/dev/null
    local rows=0
    [ -f "$dir/log_r0.csv" ] && rows=$(awk -F, '$2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$dir/log_r0.csv")
    local csvs; csvs=$(ls "$dir"/log_r*.csv 2>/dev/null | wc -l)
    # rc is REPORTED, never fatal: one hung rung must not cost the allocation.
    # Judge a rung on CSV rows and rank coverage, not on exit code -- a watchdog
    # kill gives rc!=0 with perfectly good iterations already on disk.
    echo "  rung $name rc=$rc in $(( $(date +%s) - t0 ))s : $rows rows on r0, $csvs/$W rank CSVs"
    if [ "$csvs" -ne "$W" ]; then
        echo "  WARNING rung $name: $csvs of $W rank CSVs. max-over-ranks on a"
        echo "          partial set understates the max -- this is exactly how the"
        echo "          old 'arm_B 1n' baseline came to be wrong. Analysis will flag it."
    fi
    return 0
}

for spec in $RUNGS; do
    run_rung "$spec"
done

echo
echo "LADDER COMPLETE $(date). Rungs in $OUTROOT:"
ls -1 "$OUTROOT" 2>/dev/null | sed 's/^/  /'
echo
echo "Analyze:  $PY $ROOT/scripts/scaling_efficiency.py --ladder $OUTROOT --clips-per-rank 2"
rm -rf "$JOBTMP"
