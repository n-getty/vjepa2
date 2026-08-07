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
#   - A rung is `<nodes>` plus optional `:nw<N>` / `:cfg<NAME>` / `:omp<N>`
#     fields (full grammar below), so num_workers is a first-class arm
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
#   L4: qsub -q debug -l select=1 \
#         -v VJEPA_LADDER_RUNGS="1:prof1 1:nw2:prof1" scripts/scaling_ladder.sh
#   L5: qsub -l select=64 -v VJEPA_LADDER_RUNGS="1:nw2 64:nw2 64 64:nw2:cfgvitG384_lbA_g16" \
#         scripts/scaling_ladder.sh
#
# L5 carries three separate questions in one allocation, in the order that makes
# each one answerable:
#   - `1:nw2` is the RE-ANCHOR. The ladder's 100% reference does not reproduce
#     (memory 1n-anchor-does-not-reproduce: two identical 1n nw=0 rungs differ 2x
#     with an identical compute floor). Any efficiency-vs-1n figure needs its 1n
#     measured in the SAME allocation as the rung it divides; cross-job 8n agreed
#     to 7.8% but cross-job 1n does not. Cheap, and it goes first so a later hang
#     cannot cost it.
#   - `64:nw2` is the HAZARD ARM and the last gate on nw>0 as a default. The
#     xccl-fork deadlock (train.py:207-211) is O(ranks), so the 1n and 8n passes
#     do not transfer (memory scale-dependent-results-dont-transfer). A HANG IS A
#     RESULT here, not a failure -- it is the answer to whether nw>0 can ship.
#   - `64` (nw=0) is its in-allocation control, so the nw pair is same-nodes,
#     same-fabric-hour. It runs AFTER the hazard arm on purpose: if the hazard
#     arm hangs it burns its watchdog deadline, and the control is the cheaper
#     thing to lose.
#   - `64:nw2:cfg...g16` is the CORPUS arm, paired against `64:nw2` on the same
#     nodes. It is last because it is the only rung whose result is optional.
#
# L4 is the TAIL rung pair and needs only ONE node -- the >ceiling dataload
# draws are present at 1n, so node count buys nothing here and the cheap queue
# is the right place for it. It is a PAIR because the two profile columns are
# not equally trustworthy at both worker settings (see _PerSampleDecode): at
# nw=0 `decode` is clean but `gap` also contains the training step, so nw=0
# alone can only confirm a codec tail, never exclude one. The nw=2 rung is what
# makes `gap` a storage measurement. Read them in that order.
#
# L0 is the PROBE-OVERHEAD CHECK and gates the rest: if probe-ON and probe-OFF
# do not have overlapping IQRs, the ladder is measuring its own barrier and the
# design needs revising before the L1-L3 debug-scaling slots are spent.
#
# L6 is the STAGED-vs-DAOS arm for the dataload tail, and it needs THREE arms,
# not two, plus a closing anchor:
#
#   L6: qsub -q debug -l select=2 -v \
#         VJEPA_LADDER_RUNGS="2:nw2 2:nw2:store staged:cap24 2:nw2:cap24 2:nw2" \
#         scripts/scaling_ladder.sh
#         (write the store field WITHOUT the space -- `store staged` above is a
#          line-wrap artifact of this comment; the real spec is `2:nw2:storestaged:cap24`)
#
# cap24 is sized, not guessed. Measured against vitG384_lbA's mean shard size:
# cap=0 puts 2317 GiB/node at N=2 (~90 min at 0.43 GB/s/node, longer than the
# slot); cap=50 puts 254 GiB on a tmpfs that is RAM and already loses ~690 GiB
# to unattributed host memory over a run; cap=24 puts 122 GiB and stages in
# ~4.7 min at EVERY node count (the 24 floor binds from 2n up). 24 is also
# exactly 12 local ranks x 2 DataLoader workers -- the worker-independence
# floor, below which a worker would have to share a shard.
#
#   arm 1  daos-full      the production path, and the OPENING anchor
#   arm 2  staged-capped  local tmpfs, so the DAOS agent and the NIC are bypassed
#   arm 3  daos-capped    THE CONTROL. Same shard window as arm 2, over DAOS.
#   arm 4  daos-full      the CLOSING anchor -- the noise floor
#                         ([[wallclock-kill-deletes-the-closing-anchor]])
#
# Arm 3 is not optional. The staged arm must be capped (see the staging block in
# run_rung for why the uncapped window is ~90 min of copying at N=2), so arms 1
# and 2 differ in BOTH the storage path and the working-set size. A staged win
# over arm 1 alone cannot separate "DAOS is slow" from "the working set now fits
# in page cache". Arms 2-vs-3 isolate the path; arms 3-vs-1 isolate the window.
#
# CONFOUND TO REPORT EITHER WAY: a capped arm reads a fixed subset of every
# source, so its sampling diversity is not the production one. Harmless for
# s/iter (loss from this ladder is meaningless anyway) but it must be stated.
#
# A rung is
# <nodes>[:nw<N>][:pf<N>][:node<K>][:probe<0|1>][:prof<0|1>][:cfg<NAME>][:omp<N>][:store<daos|staged>][:cap<N>],
# fields in any order. Unknown fields are rejected, not ignored -- a typo'd arm
# that silently ran the default config would be indistinguishable from a real
# null result.
#
# `:store<daos|staged>` picks the storage path. `staged` copies this rung's
# shard window from DAOS to each node's /tmp before the rung and removes it
# after (tmpfs is RAM; a leftover window is memory the next rung loses).
#
# `:cap<N>` caps the shards/source/node that a rung reads, via
# src/datasets/shard_window.py -- the same function the stager uses, so a capped
# DAOS rung and a staged rung of the same cap read an IDENTICAL shard set
# (tests/datasets/test_shard_cap.py pins that agreement). Either field forces
# WDS_LOCAL_SLICING=1, because both give each node a different subset.
# NEVER use cap on a training run: it sees a fixed prefix of every source.
#
# `:pf<N>` sets VJEPA_PREFETCH_FACTOR -- batches each worker queues ahead
# (webdataset.py:993, default 2). It exists to test the one mechanism the tail
# analysis names but has never varied: nw=2 does not remove the tail, it makes
# it RARER (5.35x median vs 1.96x wall at 8n), because "a stall deeper than the
# prefetch queue stalls the step no matter who is reading". Queue depth is that
# depth. Prediction if the story is right: deeper queue absorbs deeper stalls,
# so total wall improves while the median barely moves. A null says the tail
# stalls are longer than any affordable queue, which is itself informative and
# closes the lever. It is IGNORED at nw=0 (DataLoader rejects the kwarg with no
# workers) -- pf without nw>0 is rejected rather than silently dropped.
# COST: queue depth is memory. Each worker holds pf batches of decoded clips, so
# node RAM scales nw x pf x bs, on a node whose /tmp is already RAM.
#
# `:node<K>` starts the rung's private nodefile at allocation node K instead of
# the head (default 0). Without it every rung takes the FIRST R nodes, so two 1n
# rungs in one allocation always land on the same node and "is this effect
# node-local?" is a question the launcher cannot ask. The only cross-node
# evidence today comes from separate allocations, where node is confounded with
# fabric-hour and run length. `1:node0 1:node1` in one job varies node alone.
# A rung whose window runs off the end of the allocation is SKIPPED, never
# clamped back to the head -- a node1 rung silently rerun on node 0 would be a
# same-node repeat wearing a cross-node name.
#
# `:omp<N>` sets OMP_NUM_THREADS and the mpiexec --depth together (see run_rung).
# It exists for the CPU-oversubscription arm: the default 16 threads x 12 ranks
# is 192 threads on 104 cores, and decode is CPU work.
#
# `:cfg<NAME>` names a config under configs/vitg16_surg_vid_webdataset_single4/
# and exists for CORPUS arms (e.g. the g16 re-encode), which cannot be expressed
# as an env knob. It only tags the rung dir when it differs from the job's
# config, so every existing rung name is unchanged. Remember that a cfg arm may
# move more than the corpus -- vitG384_lbA_g16 also downscales sitl_2026 to
# short-side 512 -- so read the config header before attributing its delta.
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
# Read AFTER the source, so an omp-less rung uses whatever the recipe resolved
# rather than a second, independently-drifting default. Captured into its own
# variable because a rung's `:omp<N>` field overwrites OMP_NUM_THREADS in the
# rung's env -- comparing against the live variable would then make every rung
# look like the default and suppress the _omp<N> dir tag.
#
# Do NOT write this as ${OMP_NUM_THREADS:-16}. PBS exports OMP_NUM_THREADS=208
# (104 cores x 2 HT) into every job script, so a `:-` default silently never
# fires; that is the bug this whole line exists to have caught.
LADDER_OMP_DEFAULT=$OMP_NUM_THREADS
# Prefetch queue depth. Resolved and EXPORTED here rather than left to the
# loader's internal default, for the same reason as omp above: a rung's `:pf<N>`
# field must be comparable against a value this script knows, not against one
# buried in webdataset.py that could drift. The 2 mirrors that loader default
# (src/datasets/webdataset.py:993) and tests/test_ladder_rung_spec.py pins the
# two together, so a change on either side fails a test instead of silently
# retagging every rung.
export VJEPA_PREFETCH_FACTOR=${VJEPA_PREFETCH_FACTOR:-2}
LADDER_PF_DEFAULT=$VJEPA_PREFETCH_FACTOR
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

# Stage the imported packages to node-local /tmp, once per allocation, before
# any rung. Job 8742027 lost a rung to Python import blocking on /lus/flare
# (12 concordant stacks in importlib get_data), and the HEALTHY import is
# 39-53 s of every rung on top of that. Measured on a login node, warm cache:
# 9.5 s -> 4.4 s, with all 2777 modules resolving from /tmp and none falling
# back. Cold compute nodes should show more, not less.
#
# VJEPA_LADDER_STAGE_VENV=0 disables it -- and staging must be A/B-able, because
# /tmp is RAM ([[aurora-tmp-is-tmpfs]]) and 4.5 GB/node of it competes with page
# cache, which the dataload-tail work has shown is not a free resource. If a
# ladder result ever moves when this flag does, that is a finding, not noise.
if [ "${VJEPA_LADDER_STAGE_VENV:-1}" = "1" ]; then
    export VJEPA_VENV_LOCAL=${VJEPA_VENV_LOCAL:-/tmp/vjepa_venv}
    export VJEPA_REPO_ROOT=$ROOT
    # Regenerate the package list from THIS venv rather than shipping a stale
    # one: it takes ~10 s and a manifest that has drifted from the environment
    # is exactly how a partial stage happens.
    #
    # `python`, NOT $PY. $PY is the frameworks interpreter and resolves a
    # DIFFERENT site-packages than the activated venv the ranks import from;
    # a manifest generated against the wrong tree names packages that do not
    # exist under $SITE, and rsync would then stage a subset without saying so.
    # The closure must be captured by the interpreter that will run the trainer.
    #
    # Through the ENVIRONMENT, not a file. $JOBTMP is /tmp -- node-local tmpfs
    # ([[aurora-tmp-is-tmpfs]]) -- so a manifest written there is invisible to
    # every node but the head one. Job 8742102 did exactly that: node 0 staged
    # 22933 files from the real closure while node 1 fell back to the built-in
    # guess and staged 19462, and the only sign was one line reading
    # "no manifest" in a log that otherwise looked clean. The two nodes of an A/B
    # ran with different staged trees.
    #
    # Lustre would fix the visibility and reintroduce the dependency this whole
    # change exists to remove. 156 package names is ~2 KB, well inside any
    # argv/env limit, so the env carries it with no filesystem involved at all.
    if python $ROOT/scripts/gen_import_closure.py --out "$JOBTMP/pkgs.txt" 2>/dev/null \
       && [ -s "$JOBTMP/pkgs.txt" ]; then
        export VJEPA_VENV_PKGS=$(tr '\n' ' ' < "$JOBTMP/pkgs.txt")
        echo "  import closure: $(wc -l < "$JOBTMP/pkgs.txt") packages (via env, ${#VJEPA_VENV_PKGS} bytes)"
    else
        echo "  closure generation failed -- stager will use its built-in package list"
    fi
    t_venv=$(date +%s)
    # -ppn 1: one stager per node. NOT fatal on failure -- the stager itself
    # exits 0 without publishing if anything goes wrong, and the ranks fall back
    # to the Lustre venv, so the worst case is the status quo.
    #
    # HARD DEADLINE, and it is not belt-and-braces. This pass reads 4.6 GB over
    # 22933 files off the SAME filesystem whose stalls it exists to avoid; a
    # stager wedged in the Lustre read path would burn the allocation exactly as
    # the import did. The timeout kills the mpiexec, not a D-state child -- but
    # the child holds only its own node's temp dir, which is never published
    # without the .complete marker, so an abandoned stage is inert.
    #
    # 900 s is deliberately generous against a MEASURED 392 s on a busy login
    # node. That measurement is also why staging can be a net LOSS at few rungs:
    # 392 s once against ~45 s of import per rung. It pays for itself past ~9
    # rungs on the mean, and immediately against the 900 s+ pathological import.
    # Report the real number every job rather than assume the login-node one.
    timeout 900 mpiexec -n "$NNODES" -ppn 1 --hostfile "$PBS_NODEFILE" \
        --cpu-bind none --no-vni \
        bash $ROOT/scripts/stage_venv_local.sh 2>&1 | tail -20
    _stage_rc=${PIPESTATUS[0]}
    [ "$_stage_rc" = "124" ] && echo "  venv staging TIMED OUT at 900s -- ranks will use Lustre"
    echo "venv staging pass done in $(( $(date +%s) - t_venv ))s (rc=$_stage_rc)"
else
    echo "venv staging DISABLED (VJEPA_LADDER_STAGE_VENV=0) -- imports read /lus/flare"
fi

# --weak-scale keeps per-rank batch FIXED as nodes grow, which is the only
# meaningful setting for a per-tile scaling curve: every rung must do identical
# per-rank work or the throughput comparison is between different workloads.
# Called once for the allocation -- under weak scaling batch_size does not depend
# on node count, and the config's `nodes:` key is read only on the submit path
# (app/main_dist_aurora.py:284), never under --train_mode.
# TAKE THE PATH FROM THE SCRIPT -- do not re-derive it here. The script prints
# its output path on stdout (prepare_runtime_config.py:172) and owns the naming
# rule, which has a special case this launcher does not: at num_nodes==1 the
# directory is `g12_weak`, NOT `n1g12_weak` (:117-124). Job 8740602 -- the 1-node
# L4 tail rung -- died at `FATAL: runtime cfg missing` on exactly that, having
# already spent the queue wait and the DAOS mount. Every earlier rung set started
# at >=2 nodes, so the hardcoded spelling was right by accident for L0-L3 and the
# one allocation size that exercises the special case is the cheap-queue job.
# Two copies of a naming rule is one copy too many.
RUNTIME_CFG=$($PY $ROOT/scripts/prepare_runtime_config.py \
  $BASE_CFG --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale | tail -n1) || exit 1
[ -r "$RUNTIME_CFG" ] || { echo "FATAL: runtime cfg missing: '$RUNTIME_CFG'"; exit 1; }
echo "runtime cfg: $RUNTIME_CFG"

# Per-rung config override (`:cfg<NAME>`), memoized. Exists for CORPUS arms: the
# g16 re-encode is a change to the data, not to a knob, so it can only be A/B'd
# by swapping the config -- and a corpus A/B across two jobs is not a corpus A/B,
# it is a fabric-hour measurement (scripts/ccl_knob_sweep.sh:37-42). Same rule
# that makes every other arm here serial-within-one-allocation.
#
# Memoized because prepare_runtime_config.py writes a file and the answer depends
# only on (name, NNODES, PPN); calling it per rung would rewrite the same path
# while a previous rung's params.yaml was copied from it. Copy-at-rung-start is
# what makes that safe today, but the cache removes the hazard entirely.
declare -A _RTCFG_CACHE
_RTCFG_CACHE[$CFG_NAME]=$RUNTIME_CFG
runtime_cfg_for () {
    local n=$1
    if [ -z "${_RTCFG_CACHE[$n]+x}" ]; then
        local base=$ROOT/configs/vitg16_surg_vid_webdataset_single4/${n}.yaml
        [ -r "$base" ] || { echo "FATAL: no such config: $base" >&2; return 1; }
        local out
        out=$($PY $ROOT/scripts/prepare_runtime_config.py \
              $base --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale | tail -n1) \
            || { echo "FATAL: prepare_runtime_config failed for $n" >&2; return 1; }
        [ -r "$out" ] || { echo "FATAL: runtime cfg missing for $n: '$out'" >&2; return 1; }
        _RTCFG_CACHE[$n]=$out
        echo "runtime cfg [$n]: $out" >&2
    fi
    printf '%s' "${_RTCFG_CACHE[$n]}"
}

# Publish a locally-built file to Lustre without ever truncating a Lustre
# object, and without letting a wedged one cost the allocation.
#
# Two defences, because they cover different failures:
#
#  1. WRITE-ONCE + rename(). The destination is written exactly once, at a name
#     that has never existed, so no O_TRUNC is ever issued against Lustre --
#     and O_TRUNC is the syscall that hung job 8742027's launcher in
#     uninterruptible D state on `osc_io_setattr_end` for the remainder of its
#     slot. rename() within a directory is an MDS metadata operation; it does
#     not resize the object it replaces.
#
#  2. A DEADLINE. Write-once removes the known failure, not all failures --
#     Lustre can stall a plain write too. So the copy runs in the
#     background and is polled. If it outlives the deadline the caller is told
#     and the rung is skipped, so the job loses one rung instead of every
#     remaining one. The stuck child is deliberately NOT killed: D state does
#     not take signals, `kill -9` would only mislead the log into claiming a
#     reap that did not happen ([[no-lazy-cause-labels]]). It exits with the job.
#
# Unique-per-attempt, not per-rung: a retry after a stalled attempt must not
# reuse the name the stalled writer still holds open.
_PUBLISH_SEQ=0
PUBLISH_TIMEOUT_S=${VJEPA_LADDER_PUBLISH_TIMEOUT_S:-120}
publish_atomic () {
    local src=$1 dst=$2
    _PUBLISH_SEQ=$(( _PUBLISH_SEQ + 1 ))
    local stage="${dst}.new.$$.${_PUBLISH_SEQ}"
    local errf="$JOBTMP/publish_${_PUBLISH_SEQ}.err"
    # O_CREAT|O_EXCL semantics come free: the name has never existed.
    #
    # The child's stdio is redirected AWAY from the launcher's, and that is not
    # tidiness. A backgrounded child inherits the parent's stdout, and a child
    # stuck in D state holds it for the life of the job -- so anything reading
    # this script's output through a pipe would block on a writer that already
    # timed out, turning a skipped rung back into a hung ladder. Same fd-lifetime
    # trap as [[watchdog-disarmed-by-command-substitution]], the other direction.
    ( cp "$src" "$stage" && mv -f "$stage" "$dst" ) > /dev/null 2>"$errf" &
    local pid=$!
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$waited" -ge "$PUBLISH_TIMEOUT_S" ]; then
            echo "  PUBLISH STALLED: $dst did not appear in ${PUBLISH_TIMEOUT_S}s (pid $pid)"
            echo "    state=$(awk '{print $3}' /proc/$pid/stat 2>/dev/null) wchan=$(cat /proc/$pid/wchan 2>/dev/null)"
            echo "    Not killing it: D state ignores signals. Skipping this rung."
            return 1
        fi
        sleep 2; waited=$(( waited + 2 ))
    done
    wait "$pid"
    local rc=$?
    [ "$rc" -eq 0 ] || { echo "  PUBLISH FAILED rc=$rc: $(head -3 "$errf" 2>/dev/null)"; return "$rc"; }
    return 0
}

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
# Reaping is CLUSTER-WIDE, and that asymmetry is the point. The watchdog runs on
# the head node only, so a bare `pkill -9` reaps 12 of a 2-node rung's 24 ranks
# and leaves the rest holding XPU tiles, host memory and their half of the CCL
# world. mpiexec then returns -- the launcher sees its ranks die -- and the NEXT
# rung starts on nodes that still have the previous rung's orphans on them.
#
# Job 8741955 is what that looks like: rungs 1 and 2 clean (rc=0, 80 rows each),
# rung 3 no-first-iter, rung 4 no-first-iter -- and rung 4 is the CLOSING ANCHOR,
# byte-identical in config to rung 1 which had just succeeded. Positional, not
# treatment-linked, which is the signature of state carried between rungs rather
# than of the knob under test. Losing the closing anchor costs the whole
# allocation ([[wallclock-kill-deletes-the-closing-anchor]]): with no measured
# drift there is nothing to price the treatment against.
#
# The SIGUSR1 above already fans out via mpiexec; the kill must use the same
# fanout or the forensics reach ranks the reaper cannot. Kept as a function so
# the two call sites cannot drift apart.
reap_rung_everywhere () {
    # Scope to the RUNG's nodefile, not the allocation's. A `1:node1` sub-world
    # rung occupies one node of many, and reaping the allocation would kill a
    # concurrent rung if concurrency is ever added -- but more immediately, -n
    # must match the hostfile's line count or mpiexec oversubscribes.
    local _n; _n=$(wc -l < "$RUNG_NODEFILE" 2>/dev/null) || _n=0
    if [ "${_n:-0}" -gt 0 ]; then
        mpiexec -n "$_n" -ppn 1 --hostfile "$RUNG_NODEFILE" --cpu-bind none --no-vni \
            bash -c "pkill -9 -f 'app[.]main_dist_aurora' 2>/dev/null; true" >/dev/null 2>&1 || true
    fi
    # Belt and braces: if that mpiexec itself cannot launch (the failure mode
    # that most often coincides with a hang), still reap what is local.
    #
    # `app[.]main_dist_aurora` throughout, not `app.main_dist_aurora`. pkill -f
    # matches the FULL command line, and the remote reaper's own `bash -c
    # "pkill -f app.main_dist_aurora ..."` contains that literal string -- so
    # the plain pattern makes the reaper a candidate for its own -9. The bracket
    # is the standard escape: it matches the same processes but not the pattern
    # itself. Harmless-looking, and it is the difference between a reaper that
    # kills the orphans and one that kills itself first.
    pkill -9 -f 'app[.]main_dist_aurora' 2>/dev/null || true
}

RUNG_WD_PID=""
RUNG_NODEFILE=""
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
                    mpiexec -n $(wc -l < "$RUNG_NODEFILE") -ppn 1 \
                        --hostfile "$RUNG_NODEFILE" --cpu-bind none --no-vni \
                        bash -c "pkill -USR1 -f 'app[.]main_dist_aurora' 2>/dev/null; true" >/dev/null 2>&1 || true
                    sleep 20
                    reap_rung_everywhere; exit 1
                fi
            elif [ $((now-start)) -gt $FIRST_ITER_DEADLINE ]; then
                echo "WATCHDOG[$tag]: NO FIRST ITER within ${FIRST_ITER_DEADLINE}s." >&2
                echo "jobid=$PBS_JOBID rung=$tag no_first_iter" > "$diag/nofirstiter_info.txt"
                mpiexec -n $(wc -l < "$RUNG_NODEFILE") -ppn 1 \
                    --hostfile "$RUNG_NODEFILE" --cpu-bind none --no-vni \
                    bash -c "pkill -USR1 -f 'app[.]main_dist_aurora' 2>/dev/null; true" >/dev/null 2>&1 || true
                sleep 20
                reap_rung_everywhere; exit 1
            fi
        done
    ) &
    RUNG_WD_PID=$!
}

run_rung () {
    local spec=$1
    # spec = <nodes>[:nw<N>][:probe0|:probe1][:prof<0|1>][:cfg<NAME>][:omp<N>]
    # -- colon-separated, order-free.
    # probe0 exists for the PROBE-OVERHEAD CHECK that gates this whole study: the
    # barrier is supposed to be free (it sits where ranks already synchronize at
    # forward's first FSDP all-gather), but if it is not, the ladder is measuring
    # the probe. Run `16 16:probe0` in one allocation and require overlapping IQRs.
    local R="${spec%%:*}" nw="$VJEPA_NUM_WORKERS" probe="$VJEPA_SCALE_PROBE" prof=0
    local cfg="$CFG_NAME" omp="$LADDER_OMP_DEFAULT" store=daos cap=0
    local pf="$LADDER_PF_DEFAULT" node=0
    local rest="${spec#*:}" fld
    if [ "$rest" != "$spec" ]; then
        # Split on ':' by SUBSTITUTION, not by setting IFS.
        #
        # `local IFS=:` here is scoped to the whole FUNCTION, not to this if-block
        # -- and it silently corrupted the ipe-override loop 60 lines below, which
        # iterates `for ov in $LADDER_IPE_OVERRIDES` over entries like "1:nw2=40".
        # With IFS still ':' that entry splits into "1" and "nw2=40", so its key
        # never matches the spec and the override is DROPPED. Job 8741170's 1n
        # re-anchor rung was budgeted at ipe=40 and silently ran at 30. It fails
        # only for specs containing a colon, i.e. exactly the rungs that carry a
        # per-rung field, and it fails quietly -- the banner prints the wrong ipe
        # as though it were intended.
        local _saved_ifs=$IFS
        IFS=:
        set -- $rest
        IFS=$_saved_ifs
        for fld in "$@"; do
            case "$fld" in
                nw*)    nw="${fld#nw}" ;;
                pf*)    pf="${fld#pf}" ;;
                # Must be matched before nothing in particular -- `node1` cannot
                # collide with `nw*`, which needs a literal 'w' -- but keep it
                # adjacent to pf so the two placement fields read together.
                node*)  node="${fld#node}" ;;
                probe*) probe="${fld#probe}" ;;
                prof*)  prof="${fld#prof}" ;;
                cfg*)   cfg="${fld#cfg}" ;;
                omp*)   omp="${fld#omp}" ;;
                store*) store="${fld#store}" ;;
                cap*)   cap="${fld#cap}" ;;
                *) echo "  rung $spec: unknown field '$fld'"; return 1 ;;
            esac
        done
    fi
    local name="n${R}_nw${nw}"
    # Only tag the dir when the rung departs from the job's config, so existing
    # rung-dir names (and scripts/scaling_efficiency.py --ladder discovery, which
    # infers ranks from the n<R> prefix) are unchanged for every non-corpus arm.
    [ "$cfg" = "$CFG_NAME" ] || name="${name}_${cfg}"
    [ "$probe" = "1" ] || name="${name}_probe${probe}"
    # omp<N>: threads per rank AND the mpiexec --depth, which must move together
    # -- --depth is the CPU span each rank is bound to, so lowering OMP alone
    # leaves the binding unchanged and lowering --depth alone oversubscribes the
    # narrower span even harder. Tagged only when it departs from the default so
    # existing rung-dir names are unchanged.
    [ "$omp" = "$LADDER_OMP_DEFAULT" ] || name="${name}_omp${omp}"
    # pf<N>: prefetch queue depth. Reject rather than silently drop at nw=0 --
    # DataLoader takes no prefetch_factor without workers (webdataset.py:993
    # makes the kwarg conditional for exactly that reason), so a `1:nw0:pf4`
    # rung would run as a plain nw0 rung under a name promising otherwise, and
    # would then read as a null for the lever.
    if [ "$pf" != "$LADDER_PF_DEFAULT" ]; then
        if [ "$nw" = "0" ]; then
            echo "  rung $spec: pf${pf} requires nw>0 (DataLoader ignores"
            echo "          prefetch_factor at num_workers=0); refusing to run an"
            echo "          arm whose name would not describe what it ran."
            return 1
        fi
        name="${name}_pf${pf}"
    fi
    # node<K>: which node of the allocation the rung starts on. Tagged whenever
    # it is not the head, so `1:node0 1:node1` produces two distinct dirs rather
    # than the _rep2 auto-suffix -- a cross-node pair must be legible as one at
    # analysis time, not inferred from submission order after the fact.
    [ "$node" = "0" ] || name="${name}_node${node}"
    # store/cap: the two fields of the staged-vs-DAOS arm. Both tag the dir so
    # a capped arm can never be mistaken for a full-corpus one at analysis time
    # -- a capped run reads a fixed subset of every source, so its loss and its
    # sampling diversity are not comparable to an uncapped run's even though its
    # s/iter is the number being compared.
    [ "$store" = "daos" ] || name="${name}_${store}"
    [ "$cap" = "0" ] || name="${name}_cap${cap}"
    # prof<0|1>: per-source decode/gap profiling (src/datasets/webdataset.py).
    # This exists because the tail has outgrown the explanation we had for it.
    # The sparse-keyframe finding reproduces the BODY of the per-rank dataload
    # distribution -- mixture-predicted p50 1.38 s vs observed 1.20 s, mean 2.66
    # vs 3.10 -- but not the tail that actually drives the order statistic:
    # 8.7% of samples exceed 10.74 s, which is 2x the slowest per-clip decode
    # ever measured offline (lapgyn6_events 5.37 s) and therefore UNREACHABLE by
    # any mixture of measured decode costs at bs=2. The excess is present at ONE
    # node, so it is not fabric. decode_ms vs gap_ms separates the two live
    # candidates: CPU/codec cost we mis-measured offline (lands in decode) versus
    # a storage-side stall offline benchmarks structurally cannot see (lands in
    # gap). Off by default and capped to the first VJEPA_DECODE_PROFILE_RANKS
    # ranks (default 12 = one node) -- not rank 0 alone, because a >ceiling event
    # hits a median of 2 of 12 ranks, so rank-0 gating sits out ~5 iterations in
    # 6 and the rung could come back clean while the tail happened two ranks over.
    # The CAP, not the rank-0 identity, is the log-funnel safety property.
    #
    # The emission period must be set against the rung length, not left at its
    # default: a rung is ipe x bs samples per rank (80 x 2 = 160 here), so the
    # default period of 200 would emit NOTHING and be indistinguishable from a
    # clean result. Hence the explicit value below.
    [ "$prof" = "1" ] && name="${name}_prof"
    # Repeat suffix -- LAST, after every other suffix, so it dedups the final
    # name. Two rungs with identical specs are now a necessary experiment: job
    # 8741386 found backward degrades WITHIN a rung (16n: 1.68 -> 7.43 s over 50
    # iters) and the NEXT rung starts clean, so "restart clears it" has to be
    # tested with two IDENTICAL rungs back to back -- otherwise node count and
    # restart change together and the observation is confounded. Without this
    # they would share one output dir and the second would resume into the
    # first's CSVs, the collision class of [[train-mode-ignores-folder-flag]].
    if [ -e "$OUTROOT/$name" ]; then
        local _base="$name" _rep=2
        while [ -e "$OUTROOT/${_base}_rep${_rep}" ]; do _rep=$(( _rep + 1 )); done
        name="${_base}_rep${_rep}"
    fi
    # --- END RUNG SPEC PARSING ---
    # tests/test_ladder_rung_spec.py extracts everything above this marker and
    # runs it against synthetic specs. Keep field parsing and dir naming above
    # it; anything needing a real allocation goes below.

    # Offset included: a `1:node5` rung in a 2-node allocation needs 6 nodes,
    # not 1, and saying so here beats failing at the nodefile line with a
    # message about line counts.
    if [ $(( node + R )) -gt "$NNODES" ]; then
        echo "SKIP rung $spec: needs $(( node + R )) nodes (offset $node + $R),"
        echo "         allocation has $NNODES"; return 0
    fi
    local W=$(( R * PPN ))
    local dir="$OUTROOT/$name"
    local nf="$JOBTMP/nodefile_${name}"
    local params="$dir/params.yaml"
    mkdir -p "$dir"

    # Private nodefile: R nodes of the allocation starting at node index $node
    # (default 0 = the head, the historical behaviour). Same pattern as
    # scaling/fanout_pbs.py:63-86. Paired with an explicit WORLD_SIZE, which
    # src/utils/distributed.py:146-158 gives PRECEDENCE over PMI's SIZE -- that
    # is what lets a sub-world of the allocation bootstrap correctly, and
    # app/vjepa_2_1/hsdp.py:52-93 then derives num_nodes = world_size /
    # local_world_size, so the (replicate, shard) mesh is right for the rung.
    #
    # The offset exists because without it two 1n rungs in one allocation ALWAYS
    # land on the same node, which makes "is this effect node-local?" a question
    # the launcher cannot express. The only cross-node evidence available today
    # comes from separate allocations, where node is confounded with fabric-hour
    # and run length (the 16.5% 1n gap). `1:node0 1:node1` in one job holds both
    # of those fixed and varies only the node.
    local _lo=$(( node + 1 )) _hi=$(( node + R ))
    sort -u "$PBS_NODEFILE" | sed -n "${_lo},${_hi}p" > "$nf"
    local got=$(wc -l < "$nf")
    if [ "$got" -ne "$R" ]; then
        # Do NOT silently fall back to the head of the allocation: a node1 rung
        # that quietly ran on node 0 would be a same-node repeat wearing a
        # cross-node name, i.e. a guaranteed null misread as a refutation.
        echo "SKIP rung $spec: nodes ${node}..$(( node + R - 1 )) needs $(( node + R ))"
        echo "         in the allocation, nodefile yielded $got of $R"
        return 0
    fi

    local ipe=$LADDER_IPE
    for ov in $LADDER_IPE_OVERRIDES; do
        [ "${ov%%=*}" = "$spec" ] && ipe="${ov##*=}"
    done

    local rtcfg
    rtcfg=$(runtime_cfg_for "$cfg") || { echo "  rung $spec: config resolve FAILED"; return 1; }

    # Build params.yaml on tmpfs, publish it to Lustre with rename(), and never
    # open a Lustre file O_TRUNC. Job 8742027 lost its whole allocation to the
    # two-step this replaces: `cp` created params.yaml and wrote 3480 bytes to an
    # OST, and the rewrite heredoc's `open(p, "w")` then TRUNCATED that object.
    # The truncate wedged the launcher in D state for the rest of the slot --
    # wchan osc_io_setattr_end (the Lustre OSC truncate path), syscall 257
    # (openat), flags 0x80241 (O_WRONLY|O_CREAT|O_TRUNC). `stat` showed 0 bytes
    # while it hung, so the size change had reached the MDS with the OST RPC
    # still outstanding.
    #
    # Blast radius was verified as exactly ONE inode: stat instant, a NEW file in
    # the same directory instant, dd 1 MB at 441 MB/s, while `head -c 10` and
    # O_TRUNC on that one file both blocked forever. D state is uninterruptible,
    # so no watchdog and no signal can recover it -- the only defence is to never
    # put a Lustre object in the state that wedges. Building on tmpfs means the
    # truncating write happens in RAM; the Lustre file is created once, at a name
    # that has never existed, and rename() then replaces the directory entry
    # without truncating what it replaces.
    local params_tmp="$JOBTMP/params_${name}.yaml"
    cp "$rtcfg" "$params_tmp" || return 1
    $PY - "$params_tmp" "$dir" "$ipe" <<'PY'
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
    publish_atomic "$params_tmp" "$params" \
        || { echo "  rung $spec: params publish FAILED, skipping rung"; return 0; }

    # store=staged: copy this rung's capped shard window from DAOS to each
    # node's /tmp first, and point the trainer at /tmp. The window is computed
    # by src/datasets/shard_window.py -- the SAME function the `cap` field feeds
    # to the loader on a DAOS rung, which is what makes staged-capped and
    # daos-capped read an identical shard set (tests/datasets/test_shard_cap.py).
    #
    # WITHOUT A CAP THIS IS UNAFFORDABLE at small node counts and the failure is
    # counterintuitive: each node takes ceil(S/N) shards per source, so staging
    # gets CHEAPER as N grows. Measured against vitG384_lbA, at N=2 node 0's
    # take is 2317 GB of the 4634 GB corpus -- ~90 min at the observed
    # 0.43 GB/s/node, longer than the whole debug slot. A short arm does not
    # need the corpus (24 ranks x bs2 x 100 iters = 4800 clips), so cap it.
    # Local slicing follows the WINDOW, not the storage path. Both a staged rung
    # and a capped DAOS rung give each node a DIFFERENT subset of each source,
    # so the URL list must be split among that node's 12 local ranks -- a global
    # urls[rank::world_size] slice would hand rank 13 an index into node 1's
    # window as though it were node 0's. Keeping this keyed on the window is
    # also what makes staged-capped and daos-capped read identically: same
    # shards, same per-rank split, only the path differs.
    local slicing=$WDS_LOCAL_SLICING
    { [ "$store" = "staged" ] || [ "$cap" != "0" ]; } && slicing=1

    local data_root=$DAOS_MNT
    if [ "$store" = "staged" ]; then
        data_root=/tmp/ladder_stage/${TAG}_${name}
        local t_stage=$(date +%s)
        echo "  staging to $data_root (cap=$cap shards/source/node) ..."
        # -ppn 1: one stager per node, each copying its own window in parallel.
        # Failure is NOT fatal -- a partial stage would silently measure a
        # different working set than intended, so bail on this rung only.
        PBS_NODEFILE="$nf" mpiexec -n $R -ppn 1 --hostfile "$nf" \
            --cpu-bind none --no-vni \
            python $ROOT/scripts/stage_node_shards.py \
                --params "$params" --local-root "$data_root" \
                --src-root "$DAOS_MNT" \
                --num-nodes $R --local-world-size $PPN --workers 16 \
                --partition-mode nodes --max-shards-per-node "$cap" \
            || { echo "  rung $name: STAGING FAILED, skipping rung"; return 0; }
        echo "  staging done in $(( $(date +%s) - t_stage ))s"
    fi

    echo "===== RUNG $name : ${R}n x ${PPN} = ${W} ranks, ipe=$ipe, num_workers=$nw, prefetch=$pf, scale_probe=$probe, cfg=$cfg, omp=$omp, store=$store, cap=$cap, local_slicing=$slicing ====="
    # Which physical nodes, by name. A cross-node pair is only interpretable if
    # the pairing is on the record -- inferring it from the offset assumes the
    # allocation's node order, which is PBS's to choose.
    echo "  nodes: $(tr '\n' ' ' < "$nf")"
    # Publish the rung's nodefile BEFORE arming the watchdog: the watchdog's
    # SIGUSR1 fanout and its reaper both read it, and a watchdog armed against
    # an empty RUNG_NODEFILE degrades to head-node-only -- which is the bug
    # being fixed here, reintroduced by ordering.
    RUNG_NODEFILE="$nf"
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
    # Prepend the node-local staged packages, ONLY if this node published a
    # complete stage. Two properties make this safe rather than a new failure
    # mode, and both are load-bearing:
    #   - the Lustre venv stays BEHIND it, so a package that was not staged
    #     resolves to Lustre -- correct, just slow;
    #   - a PYTHONPATH entry naming a directory that does not exist is silently
    #     ignored by Python, so a node whose stager failed while the head node's
    #     succeeded falls back on its own, with no per-node bookkeeping here.
    # The head node's marker is therefore a sufficient test even though staging
    # is per-node.
    local rung_pypath=$PYTHONPATH
    if [ -f "${VJEPA_VENV_LOCAL:-/nonexistent}/.complete" ]; then
        rung_pypath="$VJEPA_VENV_LOCAL:$PYTHONPATH"
    fi
    # --no-vni: DAOS RPCs fail NA_HOSTUNREACH without it.
    # -o/-e per rank: never funnel 3072 ranks' stdout through the head node,
    #   which also serves the rendezvous store and DAOS keepalives (job 8730678).
    # No --pmi=pmix: that is the DDP transport; HSDP needs launcher=none + ofi.
    PBS_NODEFILE="$nf" \
    MASTER_ADDR=$(head -n1 "$nf") \
    MASTER_PORT=$(( PORT_BASE + R )) \
    WORLD_SIZE=$W \
    VJEPA_NUM_WORKERS=$nw \
    VJEPA_PREFETCH_FACTOR=$pf \
    VJEPA_SCALE_PROBE=$probe \
    VJEPA_DECODE_PROFILE=$prof \
    VJEPA_DECODE_PROFILE_EVERY=${VJEPA_DECODE_PROFILE_EVERY:-40} \
    VJEPA_DECODE_PROFILE_RANKS=${VJEPA_DECODE_PROFILE_RANKS:-12} \
    VJEPA_SHARD_CAP=$cap \
    WDS_LOCAL_SLICING=$slicing \
    OMP_NUM_THREADS=$omp \
    PYTHONPATH=$rung_pypath \
    mpiexec -n $W -ppn $PPN --hostfile "$nf" --cpu-bind depth --depth $omp --no-vni \
        -o "$dir/rank.%r.out" -e "$dir/rank.%r.err" \
        python -m app.main_dist_aurora --train_mode \
            --fname "$params" --params_path "$params" \
            --local_data_root "$data_root"
    local rc=$?
    kill "$wd_pid" 2>/dev/null

    # ORPHAN SWEEP. mpiexec returning does not mean every rank is gone: a rank
    # blocked in a collective on a NON-head node outlives the launcher, and it
    # keeps its XPU tile, its host memory and its share of the CCL world. The
    # next rung then starts on a node that is not idle, which is state carried
    # across an A/B boundary -- the one thing a bracketed sweep cannot tolerate.
    #
    # Unconditional, not gated on rc: the reason to sweep is that rc does not
    # tell you whether all ranks exited, and a clean rc with a surviving orphan
    # is precisely the case that silently poisons the NEXT rung rather than this
    # one. Costs one mpiexec per rung when there is nothing to reap.
    #
    # Reported, so a leak is visible as a fact instead of inferred later from a
    # rung that mysteriously would not start.
    local _orph
    _orph=$(mpiexec -n $R -ppn 1 --hostfile "$nf" --cpu-bind none --no-vni \
        bash -c "pgrep -c -f 'app[.]main_dist_aurora' 2>/dev/null || true" 2>/dev/null \
        | awk '{s+=$1} END{print s+0}')
    if [ "${_orph:-0}" -gt 0 ]; then
        echo "  rung $name: $_orph orphan rank process(es) survived mpiexec -- reaping"
        reap_rung_everywhere
        sleep 10
    fi

    # Free the tmpfs immediately. /tmp is RAM here (504 GB of the node's DDR5),
    # so a staged window left behind is memory the NEXT rung does not have --
    # and the next rung may be the daos-capped control whose whole job is to
    # read the same shards under comparable conditions.
    if [ "$store" = "staged" ]; then
        PBS_NODEFILE="$nf" mpiexec -n $R -ppn 1 --hostfile "$nf" \
            --cpu-bind none --no-vni \
            bash -c "rm -rf '$data_root'" >/dev/null 2>&1 || true
    fi
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

# Job-level soft deadline. The per-rung watchdogs above catch a HUNG rung; they
# do not catch the ordinary case of the rungs simply being slower than budgeted,
# which ends with PBS hard-killing whichever rung is running when the wall
# expires ([[soft-deadline-watchdog-required-for-self-resubmit]]).
#
# That failure is not symmetric across rungs, and that is why it needs handling
# rather than accepting. Sweeps here are bracketed A/B/B/A so the CLOSING anchor
# measures the allocation's own drift -- and the closing anchor is precisely the
# arm a wall-clock kill deletes. Losing it does not cost one arm out of four; it
# costs the noise floor, and without the floor no delta in the run can be called
# ([[1n-anchor-does-not-reproduce]]).
#
# So: skip a rung we cannot finish, and say so loudly. A skipped-and-logged arm
# is a known gap; a truncated one silently reports fewer iterations from a
# different part of the run.
JOB_START=$(date +%s)
# Wall in seconds from the #PBS directive, so the two cannot drift apart.
JOB_WALL_S=${VJEPA_LADDER_JOB_WALL_S:-$(awk -F'walltime=' '/^#PBS -l walltime=/{split($2,t,":"); print t[1]*3600+t[2]*60+t[3]; exit}' "$0")}
JOB_WALL_S=${JOB_WALL_S:-3600}
# Reserve: the tail of the last rung (checkpoint save + teardown) plus the
# LADDER COMPLETE block. Measured rung teardown is under 2 min; 300 s is slack.
JOB_RESERVE_S=${VJEPA_LADDER_JOB_RESERVE_S:-300}

rung_i=0
n_rungs=$(echo $RUNGS | wc -w)
prev_rung_s=0
for spec in $RUNGS; do
    rung_i=$((rung_i+1))
    left=$(( JOB_WALL_S - JOB_RESERVE_S - ($(date +%s) - JOB_START) ))
    # Budget from the PREVIOUS rung's measured wall, not from ipe x an assumed
    # s/iter -- rung wall includes 350-500 s of rendezvous startup, and the
    # whole point of this study is that s/iter is not known in advance.
    if [ "$prev_rung_s" -gt 0 ] && [ "$left" -lt "$prev_rung_s" ]; then
        echo
        echo "  SKIP rung $spec ($rung_i of $n_rungs): ${left}s left, previous rung took ${prev_rung_s}s."
        echo "       Skipped deliberately rather than started and hard-killed by PBS."
        if [ "$rung_i" -eq "$n_rungs" ]; then
            echo "       *** THIS WAS THE CLOSING ANCHOR. The sweep has no measured noise"
            echo "       *** floor, so tail_arm_compare.py will refuse to rank the arms."
            echo "       *** Re-run with fewer rungs or a lower ipe; do not read the"
            echo "       *** remaining arms as a result."
        fi
        continue
    fi
    t_rung=$(date +%s)
    run_rung "$spec"
    prev_rung_s=$(( $(date +%s) - t_rung ))
done

echo
echo "LADDER COMPLETE $(date). Rungs in $OUTROOT:"
ls -1 "$OUTROOT" 2>/dev/null | sed 's/^/  /'
echo
echo "Analyze:  $PY $ROOT/scripts/scaling_efficiency.py --ladder $OUTROOT --clips-per-rank 2"
rm -rf "$JOBTMP"
