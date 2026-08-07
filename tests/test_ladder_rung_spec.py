"""Rung-spec parsing in `scripts/scaling_ladder.sh`, for the `:pf<N>` field.

The rung spec is the ladder's whole experimental interface: a field that parses
wrong does not error, it runs a *different experiment* under a name promising
the one you asked for. Two such bugs have already shipped --

  * `local IFS=:` leaked past its block and silently dropped every per-rung ipe
    override whose spec contained a colon (job 8741170 budgeted ipe=40, ran 30);
  * `${OMP_NUM_THREADS:-16}` never fired because PBS exports 208, so the ladder
    and the capacity runs used different thread counts while the dir names said
    they matched.

Both failed *quietly*. These tests exercise the parser the way the script runs
it -- by sourcing the real file with a stubbed environment -- rather than
re-implementing the field grammar here, which would just drift alongside it.

`:pf<N>` sets `VJEPA_PREFETCH_FACTOR`, the DataLoader prefetch queue depth. The
property that matters most is the nw=0 rejection: `DataLoader` takes no
`prefetch_factor` without workers (`src/datasets/webdataset.py` makes the kwarg
conditional for exactly that reason), so `1:nw0:pf4` would run as a plain nw0
rung inside a directory named `..._pf4` and would then read as a null for the
lever. Refusing is the only outcome that cannot be misread later.
"""

import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = os.path.join(REPO, "scripts", "scaling_ladder.sh")
WDS = os.path.join(REPO, "src", "datasets", "webdataset.py")


def _parse_fields(spec, nw_default="2", pf_default="2", omp_default="16"):
    """Run the real field-parsing + dir-naming block against one spec.

    Extracts everything from the `local R=` line through the end of the name
    tagging, so the test sees the same code the job runs. Returns (rc, stdout).
    """
    src = open(LADDER).read()
    start = src.index('local R="${spec%%:*}"')
    # End at the allocation-size guard, the first thing after naming that needs
    # a real allocation. Anchored on the marker comment rather than the guard's
    # own text: the guard's condition changed once already (it grew the node
    # offset), and an anchor that moves with the code under test turns every
    # test in this file into a ValueError instead of a failure that says what
    # broke.
    end = src.index("# --- END RUNG SPEC PARSING ---")
    block = src[start:end]
    # `local` is only legal inside a function.
    block = "parse () {\n local spec=$1\n" + block + '\n echo "NAME=$name PF=$pf NW=$nw"\n}\n'
    script = (
        f'VJEPA_NUM_WORKERS={nw_default}\n'
        f'VJEPA_SCALE_PROBE=1\n'
        f'CFG_NAME=vitG384_lbA\n'
        f'LADDER_OMP_DEFAULT={omp_default}\n'
        f'LADDER_PF_DEFAULT={pf_default}\n'
        f'WDS_LOCAL_SLICING=0\n'
        + block
        + f'parse "{spec}"\n'
    )
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def _field(out, key):
    m = re.search(rf"\b{key}=(\S+)", out)
    return m.group(1) if m else None


def test_pf_absent_leaves_the_default_and_does_not_tag():
    rc, out = _parse_fields("2:nw2")
    assert rc == 0, out
    assert _field(out, "PF") == "2", out
    # A rung that did not ask for a pf must keep the historical dir name, or
    # scaling_efficiency.py --ladder discovery and every archived comparison
    # silently stop lining up with the new runs.
    assert _field(out, "NAME") == "n2_nw2", out


def test_pf_tags_the_dir_when_it_departs_from_the_default():
    rc, out = _parse_fields("2:nw2:pf6")
    assert rc == 0, out
    assert _field(out, "PF") == "6", out
    assert _field(out, "NAME") == "n2_nw2_pf6", out


def test_pf_equal_to_the_default_does_not_tag():
    """Asking for the default explicitly is not a different arm.

    Otherwise `2:nw2` and `2:nw2:pf2` -- the same experiment -- land in two
    directories and read as an A/B with a spurious zero delta.
    """
    rc, out = _parse_fields("2:nw2:pf2")
    assert rc == 0, out
    assert _field(out, "NAME") == "n2_nw2", out


def test_pf_at_nw0_is_rejected_not_dropped():
    rc, out = _parse_fields("2:nw0:pf4")
    assert rc != 0, "pf at nw=0 must fail the rung, not run it unprefetched"
    assert "requires nw>0" in out, out


def test_unknown_field_still_rejected():
    """The pf branch must not have widened the `pf*` glob into a catch-all."""
    rc, out = _parse_fields("2:nw2:pfx4:bogus9")
    assert rc != 0, out


# --------------------------------------------------------------------------
# :node<K> -- which node of the allocation the rung starts on
# --------------------------------------------------------------------------
#
# Without it every rung takes the FIRST R nodes, so two 1n rungs in one
# allocation are always the same node and the launcher cannot express "is this
# node-local?" at all. The cross-node evidence available before this field came
# from separate allocations, where node is confounded with fabric-hour and run
# length.


def test_node_absent_is_the_head_and_does_not_tag():
    rc, out = _parse_fields("1:nw2")
    assert rc == 0, out
    assert _field(out, "NAME") == "n1_nw2", out


def test_node0_explicit_does_not_tag():
    """node0 IS the historical behaviour, so it is not a different arm."""
    rc, out = _parse_fields("1:nw2:node0")
    assert rc == 0, out
    assert _field(out, "NAME") == "n1_nw2", out


def test_nonzero_node_tags_the_dir():
    """`1:node0 1:node1` must give two dirs, not a dir and a _rep2 suffix.

    The auto-rep suffix means "same arm, run again"; a cross-node pair is a
    different arm and has to be legible as one without consulting submission
    order after the fact.
    """
    rc, out = _parse_fields("1:nw2:node1")
    assert rc == 0, out
    assert _field(out, "NAME") == "n1_nw2_node1", out


def test_node_does_not_collide_with_nw():
    """`node*` and `nw*` both start 'n'; the case arms must stay disjoint."""
    rc, out = _parse_fields("1:node1:nw4")
    assert rc == 0, out
    assert _field(out, "NW") == "4", out
    assert _field(out, "NAME") == "n1_nw4_node1", out


def test_offset_window_is_skipped_not_clamped():
    """A window running off the end of the allocation must not fall back.

    Clamping would rerun node 0 under a name promising node 1 -- a same-node
    repeat wearing a cross-node name, which reads as a clean refutation of
    node-locality when it measured nothing of the kind.
    """
    src = open(LADDER).read()
    assert re.search(r'sed -n "\$\{_lo\},\$\{_hi\}p"', src), (
        "nodefile must be sliced from the offset, not from line 1")
    # The guard must count offset+R, not R alone.
    assert re.search(r'if \[ \$\(\( node \+ R \)\) -gt "\$NNODES" \]', src), (
        "allocation-size guard must include the node offset")


def test_rung_banner_records_the_physical_nodes():
    """The pairing has to be on the record, not inferred from the offset.

    Which physical node an index maps to is PBS's choice, so a cross-node claim
    that cannot name its two hosts is not checkable later.
    """
    src = open(LADDER).read()
    assert re.search(r"echo \"  nodes: \$\(tr '\\n' ' ' < \"\$nf\"\)\"", src), (
        "run_rung must echo the rung's node list")


def test_fields_are_order_independent():
    a = _parse_fields("2:nw2:pf6:omp8")[1]
    b = _parse_fields("2:omp8:pf6:nw2")[1]
    assert _field(a, "NAME") == _field(b, "NAME") != None, (a, b)


def test_ladder_default_matches_the_loader_default():
    """The two prefetch defaults must agree.

    `scaling_ladder.sh` exports VJEPA_PREFETCH_FACTOR and compares each rung
    against its own LADDER_PF_DEFAULT; `webdataset.py` falls back to its own
    literal when the env var is unset. If those drift, every rung gets tagged
    `_pf<N>` (or none does), and the dir names stop describing the runs.
    """
    lad = re.search(r"VJEPA_PREFETCH_FACTOR=\$\{VJEPA_PREFETCH_FACTOR:-(\d+)\}",
                    open(LADDER).read())
    wds = re.search(r'os\.environ\.get\("VJEPA_PREFETCH_FACTOR",\s*"(\d+)"\)',
                    open(WDS).read())
    assert lad and wds, (lad, wds)
    assert lad.group(1) == wds.group(1), (
        f"ladder default {lad.group(1)} != loader default {wds.group(1)}")


def test_prefetch_factor_is_exported_to_the_rung():
    """The parsed value has to reach mpiexec, not just the dir name.

    A `_pf6` directory whose run used pf=2 is worse than no arm at all: it is a
    null that looks like a measurement.
    """
    src = open(LADDER).read()
    assert re.search(r"^\s*VJEPA_PREFETCH_FACTOR=\$pf\s*\\", src, re.M), (
        "VJEPA_PREFETCH_FACTOR=$pf must be in the mpiexec env prefix")
