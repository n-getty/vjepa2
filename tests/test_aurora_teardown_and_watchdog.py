"""Two silent-failure guards, both learned from job 8739712.

Neither of these is a correctness property of training. Both are properties of
the *harness*, and both failed in a way that produced no error message -- which
is exactly why they need tests rather than a comment.

1. `app/main_dist_aurora.py` hard-exits with `os._exit()` to escape a C++
   destructor throw at interpreter shutdown that hung a completed run. Because
   that call sits in a `finally`, it also runs while an exception is propagating,
   and a hardcoded `os._exit(0)` there would report every crash as success.

2. `scripts/scaling_ladder.sh` arms a per-rung stall watchdog. It used to capture
   the pid with `pid=$(start_rung_watchdog ...)`, which blocks until the
   backgrounded subshell exits and therefore silently disarmed it. Job 8739712
   ran both rungs unwatched and lost ~25 min of a 60 min slot to a hang nothing
   reaped.
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
