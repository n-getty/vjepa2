"""Guards on node-local venv staging -- every one of them fails SILENTLY.

Job 8742027's rung 1 never reached iteration 0: all 12 ranks of node 1 dumped the
same stack in `importlib._bootstrap_external:1191 get_data`, reading module bytes
off /lus/flare, and burned the 900 s watchdog deadline. The healthy import is not
free either -- 39-53 s per rung across four measured arms. `stage_venv_local.sh`
copies the imported packages to node-local /tmp and the ladder prepends them.

The reason this needs tests rather than review is that every way it can be wrong
produces rc=0:

1. `rsync -a --files-from=<dirs>` copies NOTHING. --files-from cancels the -r
   that -a implies, so it creates the directory entries and stops. Measured: 0
   files without an explicit -r, 2 with. rc=0 both times.
2. Absolute paths in --files-from reproduce the whole /lus/flare/... hierarchy
   under the destination, so the staged tree is not importable -- and again
   rc=0, with the right byte count.
3. Publishing an unverified tree. Two real staging failures (a dlopen'd
   libscipy_openblas .so, then torch/bin/torch_shm_manager) were invisible to any
   check comparing what was copied against what was asked for. Only importing
   through the tree catches them.
4. Prepending unconditionally. If the marker check is dropped, a node whose stage
   failed gets a PYTHONPATH entry to a partial tree; a package PRESENT but
   internally incomplete is found FIRST and then fails, whereas an absent one
   correctly falls through to Lustre. Partial-at-package-granularity is safe;
   partial-at-file-granularity is not, and that asymmetry is the whole design.
"""

import os
import subprocess
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGER = os.path.join(REPO, "scripts", "stage_venv_local.sh")
LADDER = os.path.join(REPO, "scripts", "scaling_ladder.sh")


def test_rsync_copies_package_contents_not_just_directory_entries():
    """The --files-from / -r trap, run against the real rsync.

    Not a static grep for "-r": the point is the observable behaviour, and the
    flag interaction is surprising enough that asserting on the flag rather than
    the outcome would be asserting on my own belief about rsync.
    """
    with tempfile.TemporaryDirectory() as td:
        site = os.path.join(td, "site")
        os.makedirs(os.path.join(site, "pkg", "sub"))
        open(os.path.join(site, "pkg", "__init__.py"), "w").write("x = 1\n")
        open(os.path.join(site, "pkg", "sub", "deep.py"), "w").write("y = 2\n")
        manifest = os.path.join(td, "m.txt")
        open(manifest, "w").write("pkg\n")
        dest = os.path.join(td, "dest")

        # Exactly the invocation the stager uses.
        subprocess.run(
            ["rsync", "-a", "-r", f"--files-from={manifest}", site + "/", dest + "/"],
            check=True,
        )
        assert os.path.isfile(os.path.join(dest, "pkg", "sub", "deep.py")), (
            "nested file missing: --files-from cancels the -r implied by -a"
        )

        # And the mutant -- documents WHY -r is written out.
        dest2 = os.path.join(td, "dest2")
        subprocess.run(
            ["rsync", "-a", f"--files-from={manifest}", site + "/", dest2 + "/"],
            check=True,
        )
        assert not os.path.exists(os.path.join(dest2, "pkg", "sub", "deep.py")), (
            "rsync now recurses without -r; the explicit -r may be removable, "
            "but check before trusting this test's premise"
        )


def test_stager_uses_relative_names_and_explicit_recursion():
    src = open(STAGER).read()
    assert "rsync -a -r --files-from=" in src, "explicit -r is required (see the trap above)"
    # The manifest is consumed relative to $SITE. An absolute-path variant would
    # stage an unimportable tree at the same byte count.
    assert 'echo "$pkg" >> "$TMP.pkgs"' in src, (
        "manifest names must stay relative to $SITE; --files-from implies -R"
    )
    assert '"$SITE/" "$TMP/"' in src


def _run_stager(td, site, manifest_body, dest):
    env = dict(os.environ)
    manifest = os.path.join(td, "m.txt")
    open(manifest, "w").write(manifest_body)
    env.update(
        VJEPA_VENV=os.path.dirname(os.path.dirname(os.path.dirname(site))),
        VJEPA_VENV_MANIFEST=manifest,
        VJEPA_VENV_LOCAL=dest,
        VJEPA_REPO_ROOT=REPO,
    )
    return subprocess.run(
        ["bash", STAGER], capture_output=True, text=True, env=env, timeout=900
    )


def test_a_broken_staged_package_is_not_published():
    """Fail-safe is the entire contract: degrade to Lustre, never break the run.

    The failure the verify exists to catch is PRESENT-but-incomplete, not absent
    -- see the companion test below for why absent is fine. Both real staging
    failures had this shape: torch was there, `torch/bin/torch_shm_manager` was
    not, and every check that compared copied-against-requested passed.

    Simulated here by staging a `yaml` that shadows the real one and raises on
    import: found first, fails, exactly like the missing-binary case.
    """
    with tempfile.TemporaryDirectory() as td:
        site = os.path.join(td, "venv", "lib", "python3.12", "site-packages")
        os.makedirs(os.path.join(site, "yaml"))
        open(os.path.join(site, "yaml", "__init__.py"), "w").write(
            "raise ImportError('simulated incomplete package')\n"
        )
        dest = os.path.join(td, "stage")
        r = _run_stager(td, site, "yaml\n", dest)
        # exit 0 even on failure: the caller must not treat this as fatal.
        assert r.returncode == 0, f"stager must always exit 0: {r.returncode} {r.stderr!r}"
        assert not os.path.exists(dest), f"a broken tree was published: {r.stdout!r}"
        assert "VERIFY FAILED" in r.stdout


def test_a_package_absent_from_the_stage_is_not_an_error():
    """The asymmetry that makes package granularity safe, pinned as a test.

    A package missing from the staged tree resolves to the Lustre venv still
    behind it on sys.path -- correct, just slow. So a manifest that covers only
    part of the closure must still publish. This is not a lax check: it is the
    property that lets the stager drop `triton` (2.7 GB of the 4.6) if tmpfs
    pressure ever demands it, without touching anything else.
    """
    with tempfile.TemporaryDirectory() as td:
        site = os.path.join(td, "venv", "lib", "python3.12", "site-packages")
        os.makedirs(site)
        open(os.path.join(site, "harmless.py"), "w").write("pass\n")
        dest = os.path.join(td, "stage")
        r = _run_stager(td, site, "harmless.py\n", dest)
        assert r.returncode == 0
        assert os.path.isfile(os.path.join(dest, ".complete")), (
            f"a partial-but-usable stage must publish: {r.stdout!r}"
        )


def test_publish_is_atomic_via_rename_and_marked_complete():
    """A reader sees a whole tree or none -- never one mid-copy.

    Same property the ladder's params.yaml publish has, for the same reason: the
    consumer here is 12 ranks on the node starting at an arbitrary moment.
    """
    src = open(STAGER).read()
    assert 'mv -T "$TMP" "$DEST"' in src, "publish must be a rename, not a copy into place"
    i_marker = src.index('> "$TMP/.complete"')
    i_publish = src.index('mv -T "$TMP" "$DEST"')
    assert i_marker < i_publish, (
        ".complete must be written INSIDE the temp tree before the rename; "
        "writing it after publish leaves a window where the tree is visible "
        "but unmarked, and the ladder's gate would skip a good stage"
    )


def test_ladder_prepends_only_when_the_stage_completed():
    """The conditional is what preserves fail-safe -- an unconditional prepend
    turns a failed stage into a broken run instead of a slow one."""
    src = open(LADDER).read()
    assert 'if [ -f "${VJEPA_VENV_LOCAL:-/nonexistent}/.complete" ]; then' in src, (
        "prepend must be gated on the completion marker"
    )
    assert 'rung_pypath="$VJEPA_VENV_LOCAL:$PYTHONPATH"' in src, (
        "the Lustre venv must stay BEHIND the staged tree so unstaged packages "
        "still resolve"
    )
    assert "PYTHONPATH=$rung_pypath \\" in src, "the rung's mpiexec must receive it"


def test_staging_is_bounded_and_disableable():
    """Two properties, both about not making things worse.

    The pass reads 4.6 GB over 22933 files off the SAME filesystem whose stalls
    it exists to avoid, so an unbounded stager can lose the allocation exactly as
    the import did. And /tmp is RAM ([[aurora-tmp-is-tmpfs]]) -- 4.5 GB/node
    competes with page cache, which the dataload-tail work has shown is not free,
    so staging must be A/B-able against itself.
    """
    src = open(LADDER).read()
    assert "timeout 900 mpiexec" in src, "the staging pass must have a deadline"
    assert 'if [ "${VJEPA_LADDER_STAGE_VENV:-1}" = "1" ]; then' in src
    assert "venv staging DISABLED" in src, "the off path must be visible in the log"
    # The manifest must come from the interpreter the RANKS use. $PY is the
    # frameworks python and resolves a different site-packages; a manifest built
    # against it names packages absent from $SITE.
    assert "python $ROOT/scripts/gen_import_closure.py" in src
    assert "$PY $ROOT/scripts/gen_import_closure.py" not in src, (
        "closure must be captured by the venv interpreter, not the frameworks one"
    )


def test_the_package_list_reaches_every_node_not_just_the_head_one():
    """Job 8742102, caught in flight, and the most expensive kind of bug: it made
    the two nodes of an A/B differ while the log looked clean.

    The launcher wrote the closure to $JOBTMP/pkgs.txt. $JOBTMP is /tmp, which is
    node-local tmpfs, so only the head node could read it. Node 1 hit the
    `[ -r "$MANIFEST" ]` guard, fell through to the built-in package list, and
    staged 19462 files against node 0's 22933. The only trace was one line
    reading "no manifest" and a file count nobody was comparing.

    Putting the manifest on Lustre would fix visibility by restoring the
    dependency this whole change removes, so the list travels in the environment.
    """
    src = open(LADDER).read()
    assert "export VJEPA_VENV_PKGS=" in src, (
        "the package list must travel in the environment; a file under $JOBTMP "
        "is node-local and silently reaches only the head node"
    )
    assert 'VJEPA_VENV_MANIFEST=$JOBTMP' not in src, (
        "$JOBTMP is tmpfs -- a manifest path there is unreadable on every other node"
    )
    stager = open(STAGER).read()
    assert "PKGS=${VJEPA_VENV_PKGS:-}" in stager
    assert 'printf \'%s\\n\' $PKGS > "$TMP.manifest"' in stager


def test_env_package_list_actually_stages(tmp_path):
    """End-to-end on the env channel: the fix above is only worth anything if the
    stager consumes it, and the failure mode it replaces was a silent fallback."""
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    (site / "alpha").mkdir(parents=True)
    (site / "alpha" / "__init__.py").write_text("pass\n")
    (site / "beta.py").write_text("pass\n")
    dest = tmp_path / "stage"

    env = dict(os.environ)
    env.update(
        VJEPA_VENV=str(tmp_path / "venv"),
        VJEPA_VENV_PKGS="alpha beta.py",
        VJEPA_VENV_LOCAL=str(dest),
        VJEPA_REPO_ROOT=REPO,
    )
    env.pop("VJEPA_VENV_MANIFEST", None)
    r = subprocess.run(
        ["bash", STAGER], capture_output=True, text=True, env=env, timeout=900
    )
    assert r.returncode == 0
    assert "no manifest" not in r.stdout, (
        f"the env list was ignored and the built-in fallback ran: {r.stdout!r}"
    )
    assert (dest / "alpha" / "__init__.py").is_file(), r.stdout
    assert (dest / "beta.py").is_file(), r.stdout


def test_missing_manifest_entries_are_dropped_and_counted():
    """A manifest/venv mismatch must be loud and survivable.

    rsync exits 23 on a missing source and would abort the whole stage; worse, a
    manifest built against the wrong interpreter's site-packages would fail
    entirely, and without the count the log would not say why.
    """
    src = open(STAGER).read()
    assert 'if [ -e "$SITE/$pkg" ]' in src, "entries must be filtered against $SITE"
    assert "not present under $SITE" in src, "the drop count must be reported"


def test_stager_verifies_with_the_venv_interpreter():
    src = open(STAGER).read()
    assert "VERIFY_PY=$VENV/bin/python" in src, (
        "verify must use the interpreter the ranks will run; under mpiexec a bare "
        "`python` may not be on PATH at all, which fails safe but silently "
        "abandons the optimization"
    )
    assert 'import torch, numpy, torchvision, PIL, yaml, timm' in src, (
        "verification must be a real import through the staged tree -- a file "
        "count passed both of the real failures"
    )


def test_reuse_is_keyed_on_the_marker_not_the_directory():
    """Staging is once per allocation, not once per rung, but a half-built tree
    from a killed stager must not be mistaken for a finished one."""
    src = open(STAGER).read()
    assert 'if [ -f "$DEST/.complete" ]; then' in src
    assert "reusing" in src


def test_the_harness_never_writes_into_the_venv():
    """The venv is shared and on Lustre. Writing into it from 256 nodes at once
    is both a correctness hazard and the write pattern that wedged an OST."""
    src = open(STAGER).read()
    for bad in ('> "$SITE', 'rm -rf "$SITE', 'mv -T "$TMP" "$SITE'):
        assert bad not in src, f"stager writes into the venv: {bad}"
