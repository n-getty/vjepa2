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
# OVERFLOWING IT IS REPORTED AS EINVAL -- the exact error symptom (B) shows. Worth
# measuring, because Aurora's PBS TMPDIR is long: the failing job's was
#   /var/tmp/pbs.8740716.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov  (68 ch)
# and torch appends /torch-shm-dir-XXXXXX/manager.sock (34), giving 102.
#
# 102 < 107, so IT FITS, and this is NOT by itself the explanation -- said
# plainly so the next reader does not spend the slot re-deriving it. It is
# printed anyway because the margin is 5 bytes: a longer hostname, jobid, or
# server name pushes an otherwise-identical run over, and then the failure would
# be real and would look like a mystery. The guard fires only on the real limit.
_sockguess=$(( $(printf %s "$TMPDIR" | wc -c) + 34 ))
echo "       est. shm socket path len ~${_sockguess} / 107 usable sun_path bytes\
$( [ "$_sockguess" -gt 107 ] && echo '  <-- OVER THE LIMIT: EINVAL is explained' )"
echo "shm  : $(ls -ld /dev/shm 2>/dev/null); df /dev/shm: $(df -h /dev/shm | tail -1)"
echo "ulimit -n: $(ulimit -n)   -c: $(ulimit -c)"
# Symptom (B) is a socket/FD-creation failure, so capture the limits and the
# filesystem state that could produce EINVAL BEFORE running anything.
echo "stale torch-shm dirs in /tmp: $(ls -d /tmp/torch-shm-dir-* 2>/dev/null | wc -l)"
date

for stage in sigterm bare xpu dist; do
  echo ""
  echo "########## STAGE $stage ##########"
  t0=$(date +%s)
  mpiexec -n 12 -ppn 12 --cpu-bind depth --depth 16 --no-vni \
      -o "$OUT/$stage.rank.%r.out" -e "$OUT/$stage.rank.%r.err" \
      python "$ROOT/scripts/repro_nw_exit_throw.py" \
          --stage "$stage" --num-workers 2
  rc=$?
  echo "stage $stage rc=$rc in $(( $(date +%s) - t0 ))s"

  thr=$(grep -l "std::system_error" "$OUT/$stage.rank."*.err 2>/dev/null | wc -l)
  shm=$(grep -l "torch_shm_manager" "$OUT/$stage.rank."*.err 2>/dev/null | wc -l)
  cln=$(grep -l "exiting main" "$OUT/$stage.rank."*.out 2>/dev/null | wc -l)
  rel=$(grep -l "loader released cleanly" "$OUT/$stage.rank."*.out 2>/dev/null | wc -l)
  echo "  ranks: exit-throw=$thr/12  shm-spawn-fail=$shm/12  reached-exit=$cln/12  loader-released=$rel/12"

  # DID THE STAGE ACTUALLY RUN? Every rank prints a banner as its first act. If
  # none did, the script died before doing any work and all four counters above
  # are zeros-because-nothing-happened, which reads identically to
  # zeros-because-clean. The first run of this job hit exactly that: a
  # RuntimeError at import killed all 12 ranks in every stage in <1s, and the
  # summary reported four stages of "no symptom" -- an absence of evidence
  # printed as evidence of absence. Never let that render as a result again.
  ran=$(grep -l "stage=$stage" "$OUT/$stage.rank."*.out 2>/dev/null | wc -l)
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
echo "shm-spawn-fail on a stage  -> symptom (B), worker spawn. That stage's layer owns it."
echo "exit-throw on sigterm      -> symptom (A) is generic signal-kill, NOT workers."
echo "exit-throw first at bare   -> multiprocessing teardown, independent of XPU/xccl."
echo "exit-throw first at xpu    -> XPU allocator/runtime destructor ordering."
echo "exit-throw first at dist   -> xccl teardown vs worker processes."
echo "loader-released < reached-exit -> the throw is in the loader destructor itself."
echo "no symptom in any stage    -> the repro is too small; it needs the real"
echo "                              decode path or the 22.8 GB model, and the next"
echo "                              step is a real rung at VJEPA_HARD_EXIT=0."
date
