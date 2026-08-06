"""Per-source decode profiling: the instrument that can name the dataload tail.

Why this exists. The 1n->8n scaling loss is an order statistic, not degradation:
the per-rank dataload distribution is IDENTICAL at every rung (job 8740093 --
p50 1.2 s, p99 ~22 s, p(>10 s) 0.098 at 12, 24, 48 and 96 ranks) and only
max-over-ranks grows. So the tail is the only thing whose repair changes the
scaling asymptote, and the tail's owner has to be measured.

The candidate that motivated this module -- "the tail is the big-payload
sources" -- is ALREADY REFUTED by an offline decode benchmark, and that is
exactly why the instrument logs decode and gap separately rather than one
number:

    source            MB/clip   decode p50
    multibypass140       32.6         0.21 s
    grasp_noleak         96.6         0.74 s
    surgvu24_clean        9.8         3.15 s
    cholec80             22.5         3.20 s
    lemon                21.1         3.78 s

Payload size does not order decode time -- the largest source is among the
fastest. Whatever drives the tail, it is not bytes, so a profile that only
recorded total per-sample time would have reproduced the same ambiguity. The
split is the deliverable:

  * decode_ms -- demux + frame extract, CPU-side, attributable to the source;
  * gap_ms    -- everything upstream (tar read / DAOS / shuffle refill),
                 attributable to storage.

A payload/codec story puts the tail in decode_ms and sorts by source. A storage
story puts it in gap_ms and does not. These tests pin the properties the
distinction rests on -- that both halves are recorded, kept separate, attributed
per source, and that the tail survives into the percentiles instead of being
averaged away.
"""

import importlib
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class _SleepDecoder:
    """Stands in for VideoDecoder with a known, controllable decode cost."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.calls = 0

    def decode(self, sample, fpc, source_name=None):
        import time

        self.calls += 1
        time.sleep(self.seconds)
        return ("clip", 0, [0])


def _fresh(monkeypatch, enabled=True, every=1000):
    """Re-import webdataset with the profile env applied at module scope.

    _DECODE_PROFILE is resolved at import time (deliberately -- it must not cost
    a getenv per sample in the hot path), so the env has to be set before the
    module object is created.
    """
    monkeypatch.setenv("VJEPA_DECODE_PROFILE", "1" if enabled else "0")
    monkeypatch.setenv("VJEPA_DECODE_PROFILE_EVERY", str(every))
    monkeypatch.setenv("RANK", "0")  # canonical worker, else logging is gated off
    import src.datasets.webdataset as wd

    return importlib.reload(wd)


def test_off_by_default_costs_nothing(monkeypatch):
    """Default OFF, and the disabled path must not touch the profile state.

    This runs on every rank of every production job. At 3072 ranks a per-sample
    log line is what cost job 8730678 its MASTER_ADDR host to a DAOS ping
    timeout, so "off" has to mean genuinely inert, not merely quiet.
    """
    monkeypatch.delenv("VJEPA_DECODE_PROFILE", raising=False)
    import src.datasets.webdataset as wd

    wd = importlib.reload(wd)
    assert wd._DECODE_PROFILE is False, "decode profiling must default to OFF"

    dec = _SleepDecoder(0.0)
    p = wd._PerSampleDecode(dec, 16, source_name="sitl_2026")
    for _ in range(5):
        p({"__key__": "k"})
    assert dec.calls == 5, "the decoder must still be called when profiling is off"
    assert wd._decode_prof["n"] == 0, "profiling off must not accumulate samples"
    assert wd._decode_prof["by_src"] == {}, "profiling off must not record sources"


def test_decode_and_gap_are_recorded_separately(monkeypatch):
    """The split is the whole point -- one number could not tell the two apart.

    A total-time-only profile cannot distinguish "this source is expensive to
    decode" from "storage stalled before this source's sample arrived", and the
    offline benchmark shows the naive proxy (payload size) gets the ordering
    wrong. So: decode time must land in `dec`, upstream wait in `gap`, and they
    must not be conflated.
    """
    wd = _fresh(monkeypatch)
    dec = _SleepDecoder(0.05)
    p = wd._PerSampleDecode(dec, 16, source_name="sitl_2026")
    p({"__key__": "a"})
    p({"__key__": "b"})

    rec = wd._decode_prof["by_src"]["sitl_2026"]
    assert rec["n"] == 2
    assert all(d >= 0.04 for d in rec["dec"]), (
        f"decode times {rec['dec']} should reflect the 0.05 s decoder; if they "
        "are ~0 the timer is not wrapping the decode call"
    )
    # The first sample has no predecessor, so its gap is definitionally 0 -- not
    # "time since process start", which would look like a huge phantom stall.
    assert rec["gap"][0] == 0.0, "first sample must report gap 0, not process age"
    # The second sample's gap is the time between decodes, which here is only
    # test-harness overhead: it must NOT have absorbed the 0.05 s decode.
    assert rec["gap"][1] < 0.04, (
        f"gap {rec['gap'][1]:.3f}s absorbed the decode time -- gap and decode are "
        "being conflated, which defeats the storage-vs-payload split"
    )


def test_sources_are_attributed_separately(monkeypatch):
    """Per-source keying is what lets the tail be blamed on a source, or not.

    If the tail sorts by source it is a payload/codec property; if it is spread
    evenly it is storage. That test is only possible with per-source buckets.
    """
    wd = _fresh(monkeypatch)
    for src, cost, n in (("pe_video", 0.0, 3), ("sitl_2026", 0.03, 2)):
        p = wd._PerSampleDecode(_SleepDecoder(cost), 16, source_name=src)
        for _ in range(n):
            p({"__key__": "k"})

    by = wd._decode_prof["by_src"]
    assert set(by) == {"pe_video", "sitl_2026"}
    assert by["pe_video"]["n"] == 3 and by["sitl_2026"]["n"] == 2
    assert max(by["sitl_2026"]["dec"]) > max(by["pe_video"]["dec"]), (
        "the slower source must show the larger decode time; if not, samples "
        "are being attributed to the wrong bucket"
    )


def test_reservoir_is_bounded_but_keeps_the_tail(monkeypatch):
    """Memory must be bounded, and the bound must not erase what we are hunting.

    A running mean would bound memory too -- and average away the p99 that is
    the entire subject of the investigation. A capped recent-sample list keeps
    percentiles computable. This pins both halves of that trade.
    """
    wd = _fresh(monkeypatch)
    p = wd._PerSampleDecode(_SleepDecoder(0.0), 16, source_name="pe_video")
    for _ in range(600):
        p({"__key__": "k"})

    rec = wd._decode_prof["by_src"]["pe_video"]
    assert rec["n"] == 600, "the total count must be exact even when samples are dropped"
    assert len(rec["dec"]) == 512, (
        f"reservoir grew to {len(rec['dec'])}; unbounded growth over a "
        "multi-thousand-iteration run is a leak in every DataLoader worker"
    )
    # A late outlier must still be visible -- that is what the reservoir is for.
    slow = wd._PerSampleDecode(_SleepDecoder(0.05), 16, source_name="pe_video")
    slow({"__key__": "late-outlier"})
    assert max(wd._decode_prof["by_src"]["pe_video"]["dec"]) >= 0.04, (
        "a fresh outlier must survive into the reservoir, otherwise the tail is "
        "invisible exactly when it appears"
    )


def test_profiling_is_capped_to_the_first_n_ranks(monkeypatch):
    """Ranks below the cap profile; ranks above it stay silent.

    Rank 0 alone is not enough to see this tail. A >ceiling dataload event hits
    a median of 2 of the 12 ranks on a node, so rank 0 is in the stalling set
    roughly one iteration in six -- a rung could come back clean while the tail
    was happening two ranks over. Hence a cap, not a single rank.

    But the cap has to bind. Job 8730678 lost its MASTER_ADDR host to a DAOS
    ping timeout while ~58,000 setup lines funnelled through it, and that host
    was also serving the rendezvous store. Per-sample logging from 3072 ranks is
    strictly worse, so the ceiling on how many ranks emit is the safety property
    -- test both sides of it.
    """
    wd = _fresh(monkeypatch)
    monkeypatch.setenv("VJEPA_DECODE_PROFILE_RANKS", "12")

    monkeypatch.setenv("RANK", "7")  # inside the cap: must record
    p = wd._PerSampleDecode(_SleepDecoder(0.0), 16, source_name="sitl_2026")
    for _ in range(5):
        p({"__key__": "k"})
    assert wd._decode_prof["n"] == 5, (
        "rank 7 is within the 12-rank cap and must profile; rank-0-only gating "
        "would miss the stalling ranks this instrument exists to find"
    )

    before = wd._decode_prof["n"]
    monkeypatch.setenv("RANK", "12")  # first rank outside the cap
    for _ in range(5):
        p({"__key__": "k"})
    assert wd._decode_prof["n"] == before, (
        "rank 12 is outside the cap but recorded anyway -- the cap is what "
        "stops this becoming the log funnel that has taken down a head node"
    )

    monkeypatch.setenv("RANK", "3071")
    for _ in range(5):
        p({"__key__": "k"})
    assert wd._decode_prof["n"] == before, "a 3072-rank job must not profile at rank 3071"


def test_enabling_announces_itself_on_the_first_sample(monkeypatch, caplog):
    """An instrument that can be ON and silent is worse than no instrument.

    A ladder rung is `ipe x batch_size` samples per rank -- 80 x 2 = 160 -- and
    the default emission period is 200. Set that way, a `1:prof1` rung would burn
    a debug slot, produce an empty log, and be indistinguishable from "the tail
    did not happen this time". The announce line makes the ON state observable
    independently of whether any table is ever due, so an empty profile can be
    read as a real null rather than a misconfiguration.
    """
    import logging

    wd = _fresh(monkeypatch, every=1000)  # period far above any rung's sample count
    with caplog.at_level(logging.INFO, logger=wd.logger.name):
        p = wd._PerSampleDecode(_SleepDecoder(0.0), 16, source_name="sitl_2026")
        p({"__key__": "k"})

    hits = [r for r in caplog.records if "ENABLED" in r.getMessage()]
    assert len(hits) == 1, (
        "profiling must announce itself exactly once on the first sample; "
        f"got {len(hits)} announcements"
    )

    caplog.clear()
    for _ in range(10):
        p({"__key__": "k"})
    assert not [r for r in caplog.records if "ENABLED" in r.getMessage()], (
        "the announcement must not repeat -- at 12 ranks x thousands of samples "
        "a per-sample banner is the log funnel this module exists to avoid"
    )


def test_profiling_does_not_alter_the_decoded_value(monkeypatch):
    """An instrument that changes the data is not an instrument.

    Notably the None return (degenerate-clip reject -> .select(is_not_none)
    resamples) must pass through unchanged, or profiling would silently disable
    black-clip filtering.
    """
    wd = _fresh(monkeypatch)

    class _Passthrough:
        def __init__(self, val):
            self.val = val

        def decode(self, sample, fpc, source_name=None):
            return self.val

    for val in (("clip", 3, [1, 2]), None):
        p = wd._PerSampleDecode(_Passthrough(val), 16, source_name="s")
        assert p({"__key__": "k"}) == val, (
            "profiling changed the decoded value; a dropped None would defeat "
            "the degenerate-clip reject at webdataset.py:394"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
