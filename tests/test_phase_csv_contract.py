"""The PhaseTimer CSV column layout is a contract between the trainer and four
readers, and it is enforced only by convention.

`app/vjepa_2_1/train.py` writes positional columns; `scripts/scaling_efficiency.py`,
`scripts/analyze_phase_csv.py`, `scripts/analyze_straggler.py` and
`scripts/weak_scaling_report.py` read them by INDEX. Inserting a column rather
than appending one silently shifts every downstream number -- no exception, no
warning, just a wrong answer that looks plausible. These tests pin the order so
that mistake fails here instead of in a 256-node analysis.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO / "app" / "vjepa_2_1" / "train.py"

# The contract, in order. Index in this list == index in the CSV row.
EXPECTED_COLUMNS = [
    "epoch",
    "itr",
    "loss",
    "iter-time(ms)",
    "gpu-time(ms)",
    "dataload-time(ms)",
    "fwd-target-ms",
    "fwd-context-ms",
    "backward-ms",
    "opt-step-ms",
    "ema-ms",
    "loss-pred",
    "loss-context",
    "lambda",
    "l0-free-mib",
    "l0-ext-mib",
    "barrier-ms",
]


def _csv_logger_columns():
    """Column names in the order train.py passes them to CSVLogger."""
    src = TRAIN_PY.read_text()
    start = src.index("csv_logger = CSVLogger(")
    # Walk to the matching close paren so comments inside the call are included
    # but the rest of the file is not.
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                block = src[start:i + 1]
                break
    else:  # pragma: no cover - unbalanced parens would break the import anyway
        pytest.fail("could not find the end of the CSVLogger(...) call")
    return re.findall(r'\(\s*"%[^"]*"\s*,\s*"([^"]+)"\s*\)', block)


def test_csv_column_order_matches_contract():
    assert _csv_logger_columns() == EXPECTED_COLUMNS


def test_barrier_ms_is_last():
    """barrier-ms must stay APPENDED, never inserted.

    Readers index into 0..15 positionally, so a 17th column at the end is
    backward-compatible with every CSV written before 2026-08-06 while an
    inserted one would corrupt them all.
    """
    assert _csv_logger_columns()[-1] == "barrier-ms"
    assert _csv_logger_columns()[:16] == EXPECTED_COLUMNS[:16]


def test_scaling_efficiency_phase_indices_agree():
    """scripts/scaling_efficiency.py hardcodes column indices; check them."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "scaling_efficiency", REPO / "scripts" / "scaling_efficiency.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, col in mod.PHASES:
        # The analysis label is a prefix/abbreviation of the CSV column name.
        csv_name = EXPECTED_COLUMNS[col]
        stem = name.replace("fwd-tgt", "fwd-target").replace("fwd-ctx", "fwd-context")
        assert csv_name.startswith(stem), (
            f"scaling_efficiency PHASES maps '{name}' to column {col} "
            f"which the contract says is '{csv_name}'"
        )


def test_forward_marks_are_not_adjacent():
    """fwd_target_done and fwd_context_done must have real work between them.

    They were emitted on adjacent lines at three call sites, so fwd-context-ms
    was always ~0 and fwd-target-ms silently held the ENTIRE forward -- which
    made "the forward blew up at 64 nodes" impossible to attribute between the
    target encoder and the context encoder + predictor.
    """
    src = TRAIN_PY.read_text()
    assert not re.search(
        r'mark\("fwd_target_done"\)\s*\n\s*phase_timer\.mark\("fwd_context_done"\)',
        src,
    ), "fwd_target_done is immediately followed by fwd_context_done -- " \
       "fwd-context-ms will be ~0 and the forward is un-attributable"


def test_scale_probe_is_opt_in():
    """The barrier must be env-gated and off by default.

    It inserts a real collective into every iteration. Left on, it would change
    the production recipe that the throughput numbers were measured against.
    """
    src = TRAIN_PY.read_text()
    assert '_scale_probe = os.environ.get("VJEPA_SCALE_PROBE") == "1"' in src
    assert re.search(r"if _scale_probe and world_size > 1:", src)


# ---------------------------------------------------------------------------
# XPU event-counter wrap. Negative phase times were long treated as garbage and
# dropped; they are a 32-bit rollover (80 ns tick) and unwrap cleanly. Verified
# 2026-08-06 over 176,640 rank-rows: adding one period left ZERO rows negative.
# These tests pin the constant and the wall-clock/event-column split, because
# unwrapping a wall-clock column would invent a 343-second phase out of nothing.
# ---------------------------------------------------------------------------

import importlib.util as _ilu

_SE_PATH = REPO / "scripts" / "scaling_efficiency.py"


def _load_se():
    spec = _ilu.spec_from_file_location("scaling_efficiency", _SE_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wrap_period_matches_32bit_80ns_counter():
    se = _load_se()
    assert se.WRAP_MS == pytest.approx(2**32 * 80e-9 * 1000.0)
    assert se.WRAP_MS == pytest.approx(343597.38368)


def test_unwrap_recovers_a_real_wrapped_sample():
    """itr 7 of the 1n ladder rung, whose neighbours are 1357.93 and 1341.17."""
    se = _load_se()
    assert se.unwrap(-342245.12) == pytest.approx(1352.26, abs=0.01)


def test_unwrap_leaves_valid_and_unrecoverable_values_alone():
    se = _load_se()
    assert se.unwrap(1357.93) == 1357.93       # positive: untouched
    assert se.unwrap(0.0) == 0.0
    # Beyond a single period is not a wrap we can undo. It must come back
    # UNCHANGED, not merely still-negative: an unbounded `v + WRAP_MS` also
    # returns something negative here, so asserting the sign alone would let a
    # broken sample be silently shifted by 343 s before the caller rejects it.
    assert se.unwrap(-2 * se.WRAP_MS) == -2 * se.WRAP_MS
    assert se.unwrap(-400000.0) == -400000.0


def test_only_event_columns_are_unwrapped():
    """iter-time(3), dataload(5) and barrier-ms(16) are Python wall clock.

    They cannot wrap, and unwrapping one would turn a small negative into a
    343-second phase that would then dominate every max-over-ranks it entered.
    """
    se = _load_se()
    assert se.EVENT_COLS == {4, 6, 7, 8, 9, 10}
    for wall_col in (3, 5, 16):
        assert wall_col not in se.EVENT_COLS
