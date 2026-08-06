#!/usr/bin/env bash
# Bisect the nw>0 DataLoader-worker failures on Aurora XPU, cheaply.
#
# TWO distinct symptoms are now on the table, and this job is built to tell them
# apart rather than assume they are one thing:
#
#   (A) EXIT THROW -- job 8740311 rung n8_nw2, 8 nodes. All work COMPLETED
#       (rc=0, 96/96 CSVs, 40/40 iters, 15 GB checkpoint) and then 85 of 96 ranks
#       printed at exit:
#         terminate called after throwing an instance of 'std::system_error'
#           what():  No such file or directory
#
#   (B) STARTUP FAILURE -- job 8740716 rung n1_nw2_prof, 1 node. Zero iterations.
#       All 12 ranks died in DataLoader worker spawn:
#         RuntimeError: torch_shm_manager at ".../torch/bin/torch_shm_manager":
#         Invalid argument
#         Exception raised from start_manager at libshm/core.cpp:62
#       -> "Exceeded max retries (5) when loading data" after 5 refresh attempts.
#
# (B) is NOT a smaller version of (A): one never starts, the other finishes
# everything first. Do not merge them until something shows they share a cause.
#
# WHAT IS RULED OUT, so it is not re-proposed:
#   * config -- the THROUGHPUT KNOBS lines of the failing 1n rung and the working
#     8n rung are byte-identical (nw=2 pin_mem=True persistent_workers=True).
#   * nw=0 -- 0 of 300 ranks across three nw=0 rungs show either symptom.
#   * MP_SOCKET_DIR=/tmp -- VERIFIED A NO-OP. The string does not appear anywhere
#     in the installed torch: 0 hits in the python tree, 0 in libshm.so, 0 in the
#     torch_shm_manager binary. main_dist_aurora.py:39 sets it and the comments at
#     :455 credit it as an active mitigation; that is wrong and the code comment
#     should be corrected regardless of how this job turns out.
#   * set_sharing_strategy("file_system") -- active at main_dist_aurora.py:358 in
#     BOTH the failing and the working rung.
#   * the loader keepalive (98ca1ce) -- fixed the STALL; symptom (A) survived it.
#
# The manager binary itself is fine: run bare on a login node it prints
# /tmp/torch-shm-dir-.../manager.sock and exits 0. So "Invalid argument" is about
# the environment it is spawned into, not the executable.
#
# STAGES, each adding exactly one layer. The first that fails names the layer:
#   sigterm  no DataLoader; SIGTERM at the end. Tests whether the (A) exception is
#            just what signal-kill looks like here -- main_dist_aurora.py:443-449
#            records rank 0 throwing the SAME thing on an ordinary PBS SIGTERM,
#            with no workers involved. If this reproduces, the "forked workers"
#            framing for (A) is wrong. Runs FIRST for that reason.
#   bare     spawn workers, CPU tensors only. No XPU, no distributed.
#   xpu      + pin an XPU device and move batches to it.
#   dist     + init_process_group(xccl). The real trainer's shape.
#
# Seconds per stage, versus ~5 min of startup and a 22.8 GB checkpoint read
# before the real trainer can fail.
#
# Submit:
#   qsub -A AuroraGPT -q debug -l select=1 -l walltime=00:30:00 \
#        -l filesystems=home:flare scripts/repro_nw_exit_throw_pbs.sh
#
# NOTE: no `set -u` (memory set-u-module-load-trap: Lmod init trips it).
#
#PBS -N nwthrow
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
cd "$ROOT"

module load frameworks 2>/dev/null
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=29971
export PYTHONPATH="$ROOT:$PYTHONPATH"
export PYTHONFAULTHANDLER=1
# VJEPA_HARD_EXIT is irrelevant here (the repro never calls os._exit) but set it
# explicitly so nothing inherited from a recipe file can skip the destructors --
# destructors running is the entire subject of the test.
export VJEPA_HARD_EXIT=0

# Core files carry the C++ frame that the `terminate called` message drops.
ulimit -c unlimited 2>/dev/null
OUT=/flare/ModCon/ngetty/logs/nwthrow_$PBS_JOBID
mkdir -p "$OUT"
cd "$OUT"   # cores land here, not in the repo

echo "=== nw throw bisect, jobid=$PBS_JOBID ==="
echo "node : $(hostname)"
# TMPDIR is deliberately NOT overridden. PBS sets it per-job to
# /var/tmp/pbs.<jobid>..., and torch derives the shm socket path from it, so the
# directory that torch_shm_manager tries to create its socket in is job-scoped
# and torn down with the job. That is a candidate for symptom (B) EINVAL worth
# recording at every run -- but only a candidate: the ladder rung that FAILED and
# the 8n rung that WORKED were both ordinary PBS jobs with the same scheme, so
# TMPDIR alone does not separate them. Print it, do not assume it.
echo "tmp  : TMPDIR=$TMPDIR  df /tmp: $(df -h /tmp | tail -1)"
echo "       TMPDIR writable: $(test -w "$TMPDIR" && echo yes || echo NO)  \
len=$(printf %s "$TMPDIR" | wc -c) chars"
# A unix socket path is capped at 108 bytes of sun_path including the NUL, and
# OVERFLOWING IT IS REPORTED AS EINVAL -- the exact error symptom (B) shows.
#
# MEASURED (job 8740830), from inside the ranks -- this is symptom (B)'s cause:
#
#   PBS TMPDIR (job shell)  /var/tmp/pbs.<jobid>.<server>          68
#   + PALS per-launch uuid  .../<uuid>/tmp                        109   (+41)
#   + /pymp-XXXXXXXX                                              123
#   + /listener-XXXXXXXX                                          141   cap 107
#                                                                       OVER BY 34
#
# Headroom for TMPDIR is 107-32 = 75 chars. PBS alone (68) FITS; PALS's UUID is
# what breaks it. Verified two ways: a synthetic 68-char TMPDIR constructs a
# 100-byte listener and binds fine, and every rank here printed 141/107.
#
# ALCF documents this: user-guides/docs/aurora/known-issues.md #7, "Set TMPDIR
# to avoid AF_UNIX path too long", naming python multiprocessing and pytorch,
# and scoping it to mpiexec ON A SINGLE NODE -- which matches the asymmetry
# (1n rung failed, 8n rung worked). Fix per ALCF: export TMPDIR=/tmp before the
# launch, or mpiexec --env TMPDIR=/tmp.
#
# TWO WAYS THIS GOT MISREAD BEFORE, both worth not repeating:
#   * the torch_shm_manager socket is a DIFFERENT path and it FITS (102/107).
#     Reasoning about it answers a question nobody asked.
#   * anything computed off-node is void: tempfile.gettempdir() falls back to
#     /tmp when TMPDIR does not exist, so a login-node probe returns 36 bytes,
#     and the job shell misses the UUID and returns 102. Only the ranks' own
#     printed listener= length is evidence.
# DO NOT TRUST THIS NUMBER -- it is computed in the JOB SHELL, and the job shell
# does not see the TMPDIR the ranks get. Job 8740830 printed "~102 / 107" here,
# no warning, while every rank was at 141/107 and every worker was failing.
# PALS appends a per-mpiexec-launch UUID: the ranks run under
#   $PBS_TMPDIR/<uuid>/tmp    68 -> 109 chars (+41)
# so any estimate made before mpiexec understates by 41. The authoritative
# numbers are the `listener=` lengths the ranks and workers print themselves.
_sockguess=$(( $(printf %s "$TMPDIR" | wc -c) + 34 ))
echo "       job-shell est. ~${_sockguess}/107 -- UNDERSTATED by ~41: PALS adds"
echo "       /<uuid>/tmp per mpiexec launch. Read the ranks' own listener= lines."
echo "shm  : $(ls -ld /dev/shm 2>/dev/null); df /dev/shm: $(df -h /dev/shm | tail -1)"
echo "ulimit -n: $(ulimit -n)   -c: $(ulimit -c)"
# Symptom (B) is a socket/FD-creation failure, so capture the limits and the
# filesystem state that could produce EINVAL BEFORE running anything.
echo "stale torch-shm dirs in /tmp: $(ls -d /tmp/torch-shm-dir-* 2>/dev/null | wc -l)"
date

# Stage list. Beyond the four layer stages, two FIX ARMS run at the end against
# whichever layer failed -- `bare`, the smallest thing that reproduces it.
#
# ARM 1 winit: re-apply set_sharing_strategy INSIDE the worker. Job 8740789
#   showed the parent's setting does not cross the spawn boundary: the worker
#   traceback reaches reductions.py:616 DupFd, which is in the `else` branch and
#   is unreachable under file_system (that returns at :603). So the parent ran
#   file_system and the workers ran file_descriptor. main_dist_aurora.py:358
#   therefore protects the parent and not the population that fails.
# ARM 2 tmpdir: force a short TMPDIR. Independent of strategy -- it shortens the
#   AF_UNIX path instead of avoiding the socket. Run both: if only winit fixes
#   it the cause is the strategy, if only tmpdir does it is the path length, and
#   if both do then either is a valid fix and we pick on other grounds.
STAGES="sigterm bare xpu dist bare_winit bare_tmpdir"

# A hung stage must cost only itself. Stage `bare` in 8740789 hung on the
# non-fatal feeder-thread exception and consumed the remaining walltime, so
# three stages never ran and the job produced one usable line.
STAGE_TIMEOUT=${STAGE_TIMEOUT:-180}

for stage in $STAGES; do
  echo ""
  echo "########## STAGE $stage ##########"
  extra=""
  pystage="$stage"
  case "$stage" in
    bare_winit)  pystage=bare; extra="--worker-init 1" ;;
    bare_tmpdir) pystage=bare; extra="--short-tmpdir /tmp" ;;
  esac
  t0=$(date +%s)
  timeout -s KILL "$STAGE_TIMEOUT" \
  mpiexec -n 12 -ppn 12 --cpu-bind depth --depth 16 --no-vni \
      -o "$OUT/$stage.rank.%r.out" -e "$OUT/$stage.rank.%r.err" \
      python "$ROOT/scripts/repro_nw_exit_throw.py" \
          --stage "$pystage" --num-workers 2 $extra
  rc=$?
  echo "stage $stage rc=$rc in $(( $(date +%s) - t0 ))s\
$( [ "$rc" -eq 137 ] && echo "  <-- TIMED OUT after ${STAGE_TIMEOUT}s (hung)" )"
  # Stray ranks from a killed mpiexec would poison the next stage's port and
  # core count, so clear them before moving on.
  pkill -9 -f repro_nw_exit_throw.py 2>/dev/null

  thr=$(grep -l "std::system_error" "$OUT/$stage.rank."*.err 2>/dev/null | wc -l)
  shm=$(grep -l "torch_shm_manager" "$OUT/$stage.rank."*.err 2>/dev/null | wc -l)
  # AF_UNIX is the signature that actually appeared (48 hits, 12/12 ranks). It
  # is raised on the queue feeder THREAD and is non-fatal, so it never reaches
  # an exit code -- counting it explicitly is the only way it shows up.
  afu=$(grep -l "AF_UNIX path too long" "$OUT/$stage.rank."*.err 2>/dev/null | wc -l)
  cln=$(grep -l "exiting main" "$OUT/$stage.rank."*.out 2>/dev/null | wc -l)
  rel=$(grep -l "loader released cleanly" "$OUT/$stage.rank."*.out 2>/dev/null | wc -l)
  con=$(grep -l "consumed" "$OUT/$stage.rank."*.out 2>/dev/null | wc -l)
  echo "  ranks: exit-throw=$thr/12  shm-spawn-fail=$shm/12  af_unix=$afu/12"
  echo "         got-batches=$con/12  reached-exit=$cln/12  loader-released=$rel/12"
  # What the workers resolved, first rank only -- 12 copies of the same line is
  # noise, and any disagreement between ranks shows up in the counters above.
  grep -h "^\[worker pid=" "$OUT/$stage.rank.0.out" 2>/dev/null | head -1
  grep -h "TMPDIR=" "$OUT/$stage.rank.0.out" 2>/dev/null | head -1

  # DID THE STAGE ACTUALLY RUN? Every rank prints a banner as its first act. If
  # none did, the script died before doing any work and all four counters above
  # are zeros-because-nothing-happened, which reads identically to
  # zeros-because-clean. The first run of this job hit exactly that: a
  # RuntimeError at import killed all 12 ranks in every stage in <1s, and the
  # summary reported four stages of "no symptom" -- an absence of evidence
  # printed as evidence of absence. Never let that render as a result again.
  # Match on $pystage, not $stage: the fix arms run the `bare` stage under a
  # different arm name, so the banner they print says `stage=bare`. Matching
  # the arm name found 0 and cried STAGE DID NOT RUN over bare_tmpdir -- the
  # one arm that fully passed (rc=0, 12/12 batches). A guard against
  # nothing-happened rendering as a result must not itself render a result as
  # nothing-happened.
  ran=$(grep -l "stage=$pystage" "$OUT/$stage.rank."*.out 2>/dev/null | wc -l)
  if [ "$ran" -eq 0 ]; then
    echo "  !! STAGE DID NOT RUN -- 0/12 ranks reached the banner. The counters"
    echo "     above are meaningless. First error:"
    grep -hE "Error|Traceback" -A3 "$OUT/$stage.rank."*.err 2>/dev/null | head -8
  elif [ "$ran" -lt 12 ]; then
    echo "  !! only $ran/12 ranks started -- partial stage, treat counters as a lower bound"
  fi
done

echo ""
echo "=== cores ==="
ls -la "$OUT"/core* 2>/dev/null || echo "(none -- check /proc/sys/kernel/core_pattern)"
cat /proc/sys/kernel/core_pattern 2>/dev/null
echo ""
echo "=== READING THIS ==="
echo "-- layer stages (sigterm/bare/xpu/dist), first one that fails names the layer:"
echo "af_unix>0 + got-batches=0  -> symptom (B). The loader never delivers; the"
echo "                              OSError is on the feeder thread so the process"
echo "                              HANGS rather than exits. rc=137 confirms it."
echo "shm-spawn-fail on a stage  -> symptom (B) via a different path, worker spawn."
echo "exit-throw on sigterm      -> symptom (A) is generic signal-kill, NOT workers."
echo "exit-throw first at bare   -> multiprocessing teardown, independent of XPU/xccl."
echo "exit-throw first at xpu    -> XPU allocator/runtime destructor ordering."
echo "exit-throw first at dist   -> xccl teardown vs worker processes."
echo "loader-released < reached-exit -> the throw is in the loader destructor itself."
echo ""
echo "-- fix arms, both against \`bare\`. A fix arm is only meaningful if plain"
echo "   \`bare\` FAILED in this same job; if bare passed, the arms prove nothing."
echo "bare_winit  got-batches=12 -> the cause is the SHARING STRATEGY not crossing"
echo "                              spawn. Fix: worker_init_fn in the real loader."
echo "bare_tmpdir got-batches=12 -> the cause is TMPDIR LENGTH. Fix: short TMPDIR"
echo "                              exported in the launcher, before python starts."
echo "both pass                  -> either fix works; choose on blast radius."
echo "neither passes             -> both candidates are wrong; the worker's own"
echo "                              printed strategy/listener line says why."
echo "no symptom in any stage    -> the repro is too small; it needs the real"
echo "                              decode path or the 22.8 GB model, and the next"
echo "                              step is a real rung at VJEPA_HARD_EXIT=0."
date
