"""Three silent-failure guards in the harness, each learned from a real job.

None of these is a correctness property of training. All are properties of the
*harness*, and all failed in a way that produced no error message -- which is
exactly why they need tests rather than a comment.

1. `app/main_dist_aurora.py` hard-exits with `os._exit()` to escape a C++
   destructor throw at interpreter shutdown that hung a completed run. Because
   that call sits in a `finally`, it also runs while an exception is propagating,
   and a hardcoded `os._exit(0)` there would report every crash as success.
   (Job 8739712.)

2. `scripts/scaling_ladder.sh` arms a per-rung stall watchdog. It used to capture
   the pid with `pid=$(start_rung_watchdog ...)`, which blocks until the
   backgrounded subshell exits and therefore silently disarmed it. Job 8739712
   ran both rungs unwatched and lost ~25 min of a 60 min slot to a hang nothing
   reaped.

3. The SIGUSR1 stack-dump handler was registered inside the training loop's
   setup, so every rank ran the whole startup -- import, XPU pin, rendezvous,
   model build, dataset open -- with SIGUSR1 at its DEFAULT disposition, which
   terminates. The shell watchdog's no-first-iter path exists to photograph a
   rank stuck in exactly that window; instead it killed all 24 ranks of job
   8741955 arm `n2_nw2_pf8_rep2` (rc=138 = 128+10) and wrote no stacks. The
   forensics were disarmed over precisely the window they were built for, and
   the only visible trace was a nonzero code with empty stderr.
"""

import os
import re
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = os.path.join(REPO, "scripts", "scaling_ladder.sh")


# --------------------------------------------------------------------------
# 1. Hard-exit code selection
# --------------------------------------------------------------------------
#
# The teardown block is a handful of lines inside run_training(), which cannot be
# imported without torch + a live PMI environment. Rather than mock all of that,
# these tests exercise the *shape* of the block against the real source: the
# snippet below is checked to still match main_dist_aurora.py, so it cannot drift.

_SNIPPET = textwrap.dedent(
    """
    import os, sys, traceback
    def run(should_raise):
        try:
            if should_raise:
                raise RuntimeError("training blew up")
        finally:
            if os.environ.get("VJEPA_HARD_EXIT", "1") == "1":
                _exc = sys.exc_info()[0]
                if _exc is not None:
                    traceback.print_exc()
                try:
                    sys.stdout.flush(); sys.stderr.flush()
                except Exception:
                    pass
                os._exit(1 if _exc is not None else 0)
    run(sys.argv[1] == "raise")
    """
)


def _run_snippet(mode, env=None):
    e = dict(os.environ)
    e.pop("VJEPA_HARD_EXIT", None)
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, "-c", _SNIPPET, mode], capture_output=True, text=True, env=e
    )


def test_hard_exit_reports_success_only_on_success():
    assert _run_snippet("clean").returncode == 0


def test_hard_exit_does_not_mask_a_crash():
    # The regression this guards: `os._exit(0)` in a finally makes mpiexec report
    # rc=0 for a run that trained nothing. Assert the code AND that the traceback
    # survived -- os._exit skips Python's own handler, so without the explicit
    # print_exc the operator would get a bare nonzero code and no reason.
    r = _run_snippet("raise")
    assert r.returncode == 1
    assert "RuntimeError: training blew up" in r.stderr


def test_hard_exit_opt_out_restores_normal_exit():
    # VJEPA_HARD_EXIT=0 must give a NORMAL interpreter exit, not a different code:
    # the whole point of the opt-out is to let the destructors run so the throw can
    # be debugged. A clean run still exits 0, but via the ordinary path.
    assert _run_snippet("clean", {"VJEPA_HARD_EXIT": "0"}).returncode == 0
    assert _run_snippet("raise", {"VJEPA_HARD_EXIT": "0"}).returncode == 1


def test_snippet_matches_the_real_teardown_block():
    """The snippet above is only meaningful if it mirrors the shipped code."""
    src = open(os.path.join(REPO, "app", "main_dist_aurora.py")).read()
    assert 'os.environ.get("VJEPA_HARD_EXIT", "1") == "1"' in src
    assert "_exc = sys.exc_info()[0]" in src
    assert "os._exit(1 if _exc is not None else 0)" in src
    # Guard the specific mistake: a literal os._exit(0) would silence crashes.
    assert "os._exit(0)" not in src


# --------------------------------------------------------------------------
# 2. Watchdog arming
# --------------------------------------------------------------------------


def test_watchdog_is_not_captured_by_command_substitution():
    """`pid=$(start_rung_watchdog ...)` disarms it. Pin the fix in the source.

    Command substitution reads the child's stdout until it closes, and a
    backgrounded subshell holds that pipe open for its whole life -- so the caller
    blocks until the watchdog EXITS, then stores the pid of a corpse. It looks
    exactly like a working watchdog.
    """
    # Match against CODE only. The fix's own comment quotes the broken spelling
    # so the next reader knows what not to write; that must not trip the test.
    code = "\n".join(
        ln for ln in open(LADDER).read().splitlines() if not ln.lstrip().startswith("#")
    )
    assert not re.search(r"=\s*\$\(\s*start_rung_watchdog", code), (
        "watchdog pid captured by command substitution -- this blocks until the "
        "subshell exits and silently disarms it (job 8739712)"
    )
    src = open(LADDER).read()
    assert "RUNG_WD_PID=$!" in src, "watchdog pid must come from $! of the background job"
    assert 'wd_pid=$RUNG_WD_PID' in src


def test_watchdog_arming_is_verified_at_runtime():
    """A watchdog that fails to arm must say so, not fail silently."""
    src = open(LADDER).read()
    assert 'kill -0 "$wd_pid"' in src
    assert "did NOT arm" in src


def test_watchdog_waits_before_probing_for_the_trainer():
    """The liveness test is `pgrep app.main_dist_aurora || exit`.

    Run before mpiexec has started python on every node, that test is guaranteed
    to fail and the watchdog exits immediately -- the second half of the 8739712
    bug. There must be a grace period ahead of the loop.
    """
    src = open(LADDER).read()
    assert "WD_START_GRACE" in src
    grace = src.index("sleep $WD_START_GRACE")
    loop = src.index("while true", grace - 2000 if grace > 2000 else 0)
    assert grace < src.index("pgrep -f \"app.main_dist_aurora\""), (
        "grace period must precede the first pgrep liveness check"
    )
    assert grace < loop or "sleep $WD_START_GRACE" in src.split("while true")[0]


# --------------------------------------------------------------------------
# 3. Reaping must reach every node of the rung, not just the head
# --------------------------------------------------------------------------


def test_every_kill_of_the_trainer_is_cluster_wide():
    """A bare `pkill -9` in the watchdog reaps the head node only.

    The watchdog runs on the head node. Its SIGUSR1 already fans out via
    mpiexec, so the plain `pkill -9` that followed reached 12 of a 2-node
    rung's 24 ranks and left the other 12 holding XPU tiles and their half of
    the CCL world -- state the NEXT rung inherits.

    Assert on the shape rather than on behaviour, because the behaviour needs a
    multi-node allocation to observe. Every -9 aimed at the trainer must either
    carry a hostfile or go through reap_rung_everywhere (which supplies one).

    The reaper's own body is exempted, but not waved through: it holds ONE
    deliberate bare pkill, the local fallback for when the fanout mpiexec
    cannot launch -- the failure that most often coincides with a hang. So the
    exemption is checked rather than assumed: exactly one bare kill, and a
    hostfile'd one ahead of it.
    """
    src = open(LADDER).read()
    lines = src.splitlines()
    r0 = src[:src.index("reap_rung_everywhere () {")].count("\n") + 1
    r1 = src[:src.index("\n}", src.index("reap_rung_everywhere () {"))].count("\n") + 1

    def logical_command(i):
        """The kill's own command, joined across backslash continuations.

        A fixed-size window of preceding lines is NOT good enough, and the
        difference is the whole test. Both watchdog sites send SIGUSR1 through
        an mpiexec fanout and then kill on the next line, so any window wide
        enough to catch a wrapped `--hostfile` also catches the UNRELATED
        hostfile of the SIGUSR1 above it -- and the head-node-only kill this
        test exists to reject sails through. Verified by mutation: with the
        window version, restoring the original bug left this test green and
        only a neighbouring test failed, by luck.
        """
        j = i - 1  # 0-indexed index of the pkill line itself
        while j > 0 and lines[j - 1].rstrip().endswith("\\"):
            j -= 1
        return "\n".join(lines[j:i])

    bare_in_reaper = 0
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("#") or "pkill -9" not in s or "main_dist_aurora" not in s:
            continue
        cmd = logical_command(i)
        if r0 <= i <= r1:
            if "--hostfile" not in cmd:
                bare_in_reaper += 1
            continue
        assert "--hostfile" in cmd or "reap_rung_everywhere" in cmd, (
            f"line {i} kills the trainer without a cluster-wide fanout: {s}"
        )

    body = "\n".join(lines[r0 - 1:r1])
    assert bare_in_reaper == 1, (
        f"reaper should hold exactly one local-fallback kill, found {bare_in_reaper}"
    )
    assert body.index("--hostfile") < body.rindex("pkill -9"), (
        "the local fallback must come AFTER the cluster-wide fanout, or it is "
        "the primary path and the fanout is dead code"
    )


def test_reaper_scopes_to_the_rung_not_the_allocation():
    """`-n` must come from the rung's nodefile, not from NNODES.

    A `1:node1` sub-world rung occupies one node of a larger allocation.
    Reaping with -n $NNODES against a 1-line hostfile oversubscribes, and in a
    world with concurrent rungs it would kill a rung that is doing fine.
    """
    src = open(LADDER).read()
    start = src.index("reap_rung_everywhere () {")
    body = src[start:src.index("\n}", start)]
    assert "$RUNG_NODEFILE" in body
    assert "NNODES" not in body, "reaper must not scope to the whole allocation"


def test_pkill_patterns_cannot_match_the_reaper_itself():
    """`pkill -f app.main_dist_aurora` inside `bash -c "..."` is self-matching.

    -f matches the whole command line, and the remote reaper's own argv
    contains the pattern verbatim -- so the plain spelling races the reaper
    against its targets. `app[.]main_dist_aurora` matches the same processes
    and not the pattern text.
    """
    src = open(LADDER).read()
    for i, ln in enumerate(src.splitlines(), 1):
        s = ln.strip()
        if s.startswith("#"):
            continue
        if ("pkill" in s or "pgrep -c" in s) and "main_dist_aurora" in s:
            assert "app[.]main_dist_aurora" in s, (
                f"line {i} uses a self-matching pkill/pgrep pattern: {s}"
            )


def test_rung_nodefile_is_published_before_the_watchdog_arms():
    """Ordering bug that would silently restore head-node-only reaping.

    The watchdog reads RUNG_NODEFILE when it fires, but it is armed early; if
    the assignment came after start_rung_watchdog the variable would still hold
    the PREVIOUS rung's nodefile -- correct-looking on rung 1, wrong after.
    """
    src = open(LADDER).read()
    run = src.index("run_rung () {")
    assign = src.index('RUNG_NODEFILE="$nf"', run)
    arm = src.index("start_rung_watchdog ", assign - 4000 if assign > 4000 else run)
    arm = src.index('start_rung_watchdog "$dir/log_r0.csv"', run)
    assert assign < arm, "RUNG_NODEFILE must be set before the watchdog is armed"


def test_orphan_sweep_runs_regardless_of_exit_code():
    """rc=0 with a surviving orphan is the case that poisons the NEXT rung.

    Gating the sweep on a nonzero rc would skip exactly that case, so assert
    the sweep is not inside an `if [ $rc ... ]`.
    """
    src = open(LADDER).read()
    i = src.index("ORPHAN SWEEP")
    block = src[i:i + 1400]
    assert "pgrep -c -f" in block
    head = src[src.index("local rc=$?", src.index("run_rung () {")):i]
    assert 'if [ "$rc"' not in head and "if [ $rc" not in head


# --------------------------------------------------------------------------
# 4. SIGUSR1 must be survivable from process start, not just in the train loop
# --------------------------------------------------------------------------

_USR1_SNIPPET = textwrap.dedent(
    """
    import os, signal, sys, time
    if sys.argv[1] == "armed":
        import faulthandler, signal as _s
        faulthandler.enable(all_threads=True)
        faulthandler.register(_s.SIGUSR1, all_threads=True, chain=False)
    os.kill(os.getpid(), signal.SIGUSR1)
    time.sleep(0.2)          # let the handler run before we claim survival
    print("SURVIVED")
    """
)


def _run_usr1(mode):
    return subprocess.run(
        [sys.executable, "-c", _USR1_SNIPPET, mode], capture_output=True, text=True, timeout=60
    )


def test_unhandled_sigusr1_kills_the_process():
    """The premise. Without this the fix below looks like defensive noise.

    Python installs no default SIGUSR1 handler, so the kernel's disposition
    applies and the process dies with signal 10 -- which mpiexec reports as
    rc=138, and which is what job 8741955 arm 3 actually was. Nothing is written
    to stderr, so from the artifacts alone it is indistinguishable from a crash.
    """
    r = _run_usr1("bare")
    assert r.returncode == -signal_num(), r
    assert "SURVIVED" not in r.stdout


def signal_num():
    import signal as _s

    return int(_s.SIGUSR1)


def test_registered_sigusr1_dumps_and_continues():
    """chain=False + register (not enable-only) must DUMP and RESUME.

    Both halves matter: a dump that then aborts still loses the rung, and a
    handler that resumes without dumping loses the forensics. The watchdog's
    contract is `pkill -USR1` to photograph, then `pkill -9` to reap.
    """
    r = _run_usr1("armed")
    assert r.returncode == 0, r
    assert "SURVIVED" in r.stdout
    # faulthandler writes the stack to stderr when no file is given.
    assert "Current thread" in r.stderr or "Thread" in r.stderr, r.stderr


def test_handler_is_armed_before_the_trainer_is_imported():
    """Registration must sit in run_training(), ahead of the scaffold hand-off.

    train.py's own register() is behind `importlib.import_module`, model build
    and dataset open -- minutes of wall time at 2n, and the entire window the
    no-first-iter watchdog covers. Position is the whole property here, so
    assert on ORDER in the source, not merely on presence.
    """
    src = open(os.path.join(REPO, "app", "main_dist_aurora.py")).read()
    reg = src.index("_fh0.register(_sig0.SIGUSR1")
    fn = src.index("def run_training(args):")
    assert fn < reg, "SIGUSR1 must be armed inside run_training()"
    # Everything the startup does after this point must come later in the file.
    for later in ("init_distributed(", "app_main(", "eval_main("):
        assert reg < src.index(later, fn), f"SIGUSR1 armed after {later}"


def test_early_registration_is_not_env_gated():
    """An unhandled SIGUSR1 is lethal whether or not diagnostics are enabled.

    Gating this on VJEPA_ITER_WATCHDOG_S (as the trainer's copy is) would leave
    the lethal default in place for every run that did not opt in -- including
    every production run the shell watchdog is watching.
    """
    src = open(os.path.join(REPO, "app", "main_dist_aurora.py")).read()
    reg = src.index("_fh0.register(_sig0.SIGUSR1")
    block = src[src.index("def run_training(args):"):reg]
    assert "VJEPA_ITER_WATCHDOG_S" not in block
    assert "VJEPA_SCALE_PROBE" not in block


def test_ladder_script_is_valid_bash():
    r = subprocess.run(["bash", "-n", LADDER], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_watchdog_arms_and_survives_when_the_trainer_is_absent_at_start():
    """End-to-end on the real function, with the real caller shape.

    Sources scaling_ladder.sh's watchdog by extracting it, then checks the pid is
    live 2 s later. Before the fix this asserted False in two independent ways
    (substitution blocked; grace period absent).
    """
    src = open(LADDER).read()
    start = src.index("start_rung_watchdog () {")
    end = src.index("\n}", start) + 2
    fn = src[start:end]
    # Neutralise the body's dependencies: we are testing arming, not reaping.
    harness = textwrap.dedent(
        f"""
        RUNG_WD_PID=""
        WD_START_GRACE=5
        STALL_DEADLINE=999
        FIRST_ITER_DEADLINE=999
        NNODES=1
        {fn}
        start_rung_watchdog /tmp/no_such.csv testtag /tmp/wd_arm_test_diag
        wd_pid=$RUNG_WD_PID
        sleep 2
        if kill -0 "$wd_pid" 2>/dev/null; then echo ARMED; kill "$wd_pid" 2>/dev/null
        else echo DEAD; fi
        """
    )
    r = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=60)
    assert "ARMED" in r.stdout, f"watchdog did not stay armed: {r.stdout!r} {r.stderr!r}"
