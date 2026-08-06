"""The num_workers>0 teardown defect, and why the first fix could not reach it.

Job 8739914 (2n, nw=2) finished all 40 iterations on all 24 ranks and still took
31 s for the last rank to leave; 2 of 24 threw

    terminate called after throwing an instance of 'std::system_error'
    what():  No such file or directory

The mechanism, from that job's artifacts:

  * both throwing ranks had written CSV row 39 -- all work complete;
  * zero ranks showed a mid-run worker failure;
  * `loader` and `unsupervised_loader` are LOCALS of train.py's main()
    (:861, :572), so when main() returns they are collected and
    _MultiProcessingDataLoaderIter.__del__ -> _shutdown_workers runs right
    there -- inside app_main, BEFORE run_training's `finally`.

That last point is why VJEPA_HARD_EXIT (an os._exit in that `finally`) did not
help: it is downstream of the destructor. The fix is not a different exit, it is
keeping the loader ALIVE across main()'s return so the destructor never runs and
the already-shipped hard exit becomes reachable.

These tests pin the ordering property, not the string of any one line, so they
survive refactors of the surrounding code.
"""

import ast
import os
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "app", "vjepa_2_1", "train.py")


def _src(path):
    with open(path) as f:
        return f.read()


def test_loader_outlives_main_so_the_destructor_cannot_run_at_return():
    """The loader must be reachable from something that outlives main()'s frame.

    This is the whole fix. If the only reference is a local, CPython drops the
    refcount to zero at return and _shutdown_workers runs before any exit hook
    we control.
    """
    src = _src(TRAIN)
    tree = ast.parse(src)
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    # Find a statement inside main() that stores the loader into something
    # non-local: a module global (via `global`) or a subscript/attribute/call on
    # a module-level container.
    keeps = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            f = node.func
            # _KEEP_ALIVE.append(...) / _KEEP_ALIVE.extend(...)
            if isinstance(f, ast.Attribute) and f.attr in ("append", "extend"):
                if isinstance(f.value, ast.Name) and f.value.id.isupper():
                    keeps.append(f.value.id)
    assert keeps, (
        "No module-level container in train.py main() retains the dataloader.\n"
        "Without it, `loader`/`unsupervised_loader` die at main()'s return and\n"
        "_MultiProcessingDataLoaderIter.__del__ runs INSIDE app_main -- before\n"
        "run_training's finally, so VJEPA_HARD_EXIT can never reach it. That is\n"
        "exactly the job 8739914 failure this module documents."
    )
    for name in set(keeps):
        assert any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            for n in tree.body
        ), f"{name} is appended to in main() but never defined at module level"


def test_persistent_workers_is_passed_to_init_data():
    """The 2.1 trainer was the only trainer not passing it.

    init_data -> make_webdataset already plumbs it (data_manager.py:120,
    webdataset.py:787); the signature default is False, so omitting the kwarg
    silently re-forks the worker pool around every epoch. app/vjepa/train.py:260
    and app/vjepa_droid/droid.py:66 both pass it.
    """
    src = _src(TRAIN)
    tree = ast.parse(src)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "init_data"
    ]
    assert calls, "no init_data(...) call found in train.py"
    for c in calls:
        kws = {k.arg for k in c.keywords if k.arg}
        assert "persistent_workers" in kws, (
            "init_data(...) called without persistent_workers; it defaults to "
            "False in data_manager.py:39, so the worker pool is re-forked around "
            "each epoch. Forking after init_device_mesh is the documented xccl "
            "hazard."
        )


def test_persistent_workers_defaults_to_true():
    """The resolved default must be True, not just plumbed.

    Passing the kwarg but defaulting it to False would satisfy the test above
    and still leave _LOADER_KEEPALIVE holding an object that owns no worker --
    see test_retaining_the_loader_only_retains_workers_when_persistent.
    """
    src = _src(TRAIN)
    tree = ast.parse(src)
    got = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "persistent_workers"
                for t in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "get"
        ):
            args = node.value.args
            if len(args) == 2 and isinstance(args[1], ast.Constant):
                got = args[1].value
    assert got is True, (
        f"persistent_workers resolves from cfgs_data.get(..., {got!r}); it must "
        "default to True. app/vjepa/train.py:116 and app/vjepa_droid/train.py:117 "
        "both do. It is ANDed with (num_workers > 0) downstream, so True is inert "
        "on the nw=0 path and cannot regress it."
    )


def test_retaining_the_loader_only_retains_workers_when_persistent():
    """The two fixes are ONE fix -- prove the coupling against real torch.

    The worker pool is owned by `DataLoader._iterator`, and __iter__ only stores
    it there when `persistent_workers and num_workers > 0`
    (torch/utils/data/dataloader.py:485). So at persistent_workers=False,
    retaining the loader retains nothing that owns a worker: every iter() hands
    back a fresh unreferenced iterator whose __del__ fires the moment the local
    holding it goes away. Fix A without fix B is a no-op, and this test is what
    stops someone reverting B as "unrelated tuning" and silently un-fixing A.
    """
    torch = __import__("importlib").import_module("torch")
    from torch.utils.data import IterableDataset

    import gc

    class _DS(IterableDataset):
        def __iter__(self):
            for i in range(64):
                yield torch.tensor([i])

    # Mirror src/datasets/webdataset.py:792 -- the loader train.py actually holds
    # is this wrapper, not a DataLoader, so the reference must survive __getattr__
    # delegation too.
    class _LenWrapper:
        def __init__(self, loader, length):
            self.loader, self.length = loader, length

        def __iter__(self):
            return iter(self.loader)

        def __len__(self):
            return self.length

        def __getattr__(self, n):
            return getattr(self.loader, n)

    def retained_iterator(persistent):
        from torch.utils.data import DataLoader

        keep = []
        dl = DataLoader(_DS(), batch_size=2, num_workers=2, persistent_workers=persistent)
        keep.append(_LenWrapper(dl, 32))  # fix A
        def use(w):  # stands in for main(): the iterator is a local that dies
            it = iter(w)
            next(it)
        use(keep[0])
        gc.collect()
        alive = dl._iterator is not None
        keep.clear()
        del dl
        gc.collect()
        return alive

    assert not retained_iterator(False), (
        "unexpected: persistent_workers=False retained an iterator. If torch "
        "changed this, re-derive the fix -- the comment in train.py is now wrong."
    )
    assert retained_iterator(True), (
        "retaining the loader with persistent_workers=True must keep "
        "DataLoader._iterator alive; if it does not, _LOADER_KEEPALIVE protects "
        "nothing and the teardown throw will come back"
    )


def test_keeping_a_reference_prevents_the_destructor():
    """Demonstrate the mechanism itself, independent of our source.

    A local dies at return and its __del__ runs; a globally-retained object's
    does not. This is the property the fix relies on, so it is worth pinning
    rather than assuming -- if a future Python changed it, the fix is void.
    """
    prog = textwrap.dedent(
        """
        import os
        KEEP = []
        class Loader:
            def __init__(self, keep): self.keep = keep
            def __del__(self): print("SHUTDOWN_WORKERS_RAN", flush=True)
        def main(keep):
            l = Loader(keep)
            if keep:
                KEEP.append(l)
            return "done"
        main(os.environ["KEEP"] == "1")
        print("MAIN_RETURNED", flush=True)
        os._exit(0)
        """
    )
    def run(keep):
        env = dict(os.environ, KEEP=keep)
        out = subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, env=env, timeout=60
        )
        return out.stdout

    dropped = run("0")
    retained = run("1")
    assert "SHUTDOWN_WORKERS_RAN" in dropped, (
        "baseline broken: an unreferenced object's __del__ should run at return"
    )
    assert dropped.index("SHUTDOWN_WORKERS_RAN") < dropped.index("MAIN_RETURNED"), (
        "the destructor must run BEFORE the caller resumes -- that ordering is "
        "the entire bug"
    )
    assert "SHUTDOWN_WORKERS_RAN" not in retained, (
        "retaining a reference must prevent the destructor; if this fails the "
        "fix in train.py does nothing"
    )


def test_hard_exit_still_guards_the_exit_code():
    """Regression guard on the earlier fix, which stays -- it is now reachable.

    A hardcoded os._exit(0) in a `finally` reports every crash as success.
    """
    src = _src(os.path.join(REPO, "app", "main_dist_aurora.py"))
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "os._exit(0)" not in code, (
        "os._exit(0) in run_training's finally would mask real crashes as rc=0"
    )
    assert "sys.exc_info()" in code, "exit code must be derived from the live exception"
