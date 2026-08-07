#!/usr/bin/env python3
"""Emit the TOP-LEVEL PACKAGES a trainer import actually reads.

Feeds scripts/stage_venv_local.sh, so node-local staging copies the packages the
run imports rather than all of site-packages.

WHY PACKAGES AND NOT FILES -- this is the whole design, and it was learned the
expensive way. A per-file manifest was built first and it failed twice:

  1. numpy imported, then died on `libscipy_openblas64_-017048f4.so: cannot open
     shared object file`. sys.modules names the extension module, never the
     libraries it dlopens through an RPATH. Adding /proc/self/maps fixed that
     particular miss.
  2. torch then died on `Unable to find torch_shm_manager at .../torch/bin/`.
     That is a plain BINARY -- not a module, not a mapped library, so no
     introspection of the running process can see it.

The second failure is not a bug to patch; it is the shape of the problem. A
package can reference arbitrary data files at runtime, so no observation of one
import can enumerate them. What makes package granularity safe is the fallback:

    the staged tree is PREPENDED to sys.path, with the Lustre venv still behind
    it. A package missing from the staged tree resolves to Lustre -- correct,
    just slow. A package PRESENT but internally incomplete is found first and
    then fails on the missing file.

So partial-at-package-granularity is safe and partial-at-file-granularity is
not. Copy whole packages.

The other trap, from sizing this by hand: DEDUPLICATE BY PATH before reporting
any total. 32 triton submodules share one __file__ (triton/_C/libtriton.so,
1.21 GB); summing per-module reports 38 GB, 30x the truth, which would make
staging look impossible.

Usage:
    python scripts/gen_import_closure.py --out /tmp/pkgs.txt
    python scripts/gen_import_closure.py --out - --stats
"""

import argparse
import importlib
import importlib.util
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--module",
        default="app.vjepa_2_1.train",
        help="module whose import closure to capture",
    )
    ap.add_argument(
        "--site",
        default=None,
        help="site-packages root; paths are emitted relative to it "
        "(default: inferred from the running interpreter)",
    )
    ap.add_argument("--out", required=True, help="output path, or - for stdout")
    ap.add_argument("--stats", action="store_true", help="print a size breakdown to stderr")
    args = ap.parse_args()

    site = args.site
    if site is None:
        # The venv's own site-packages, not the system one: with
        # include-system-site-packages=true both are on the path and only the
        # venv copy lives on Lustre.
        cands = [p for p in sys.path if p.endswith("site-packages") and os.path.isdir(p)]
        if not cands:
            print("cannot infer site-packages; pass --site", file=sys.stderr)
            return 2
        site = cands[0]
    site = os.path.realpath(site).rstrip("/")

    # Import for effect. Failures are fatal: a partial closure staged as though
    # it were complete is the failure mode this whole path exists to avoid.
    importlib.import_module(args.module)

    seen = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.realpath(f)
        if not f.startswith(site + "/"):
            continue
        seen.add(f)
        # A .py whose bytecode is cached is NOT the file that gets read.
        if f.endswith(".py"):
            try:
                c = importlib.util.cache_from_source(f)
            except (NotImplementedError, ValueError):
                c = None
            if c and os.path.exists(c):
                seen.add(os.path.realpath(c))

    # Everything the process has actually mapped from site-packages. This is what
    # catches the transitively-dlopen'd libraries that no module names -- see the
    # numpy/libscipy_openblas note in the docstring. Reading maps beats walking
    # `<pkg>.libs` by convention: it reports what was loaded, including anything
    # reached by RPATH from a directory nobody would have thought to include.
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                parts = line.rstrip("\n").split(None, 5)
                if len(parts) < 6:
                    continue
                p = parts[5]
                if not p.startswith("/"):
                    continue
                p = os.path.realpath(p)
                if p.startswith(site + "/") and os.path.isfile(p):
                    seen.add(p)
    except OSError:
        # Not Linux, or /proc unavailable. The manifest is then incomplete in a
        # way we cannot detect here, so say so rather than emit it silently.
        print(
            "WARNING: could not read /proc/self/maps; manifest may omit "
            "dlopen'd shared libraries",
            file=sys.stderr,
        )

    # Package metadata: importlib.metadata reads *.dist-info at runtime for some
    # packages, and a missing one raises rather than degrading. Cheap to include.
    for d in os.listdir(site):
        if d.endswith((".dist-info", ".egg-info")):
            p = os.path.join(site, d)
            for r, _, fs in os.walk(p):
                for fn in fs:
                    seen.add(os.path.realpath(os.path.join(r, fn)))

    # Collapse to top-level entries. A single-file module (`six.py`) is its own
    # entry; anything under a directory becomes that directory.
    tops = sorted({p[len(site) + 1:].split("/")[0] for p in seen})

    if args.stats:
        rows = []
        for t in tops:
            p = os.path.join(site, t)
            n = b = 0
            if os.path.isdir(p):
                for r, _, fs in os.walk(p):
                    for fn in fs:
                        try:
                            b += os.path.getsize(os.path.join(r, fn))
                            n += 1
                        except OSError:
                            pass
            else:
                try:
                    b = os.path.getsize(p)
                    n = 1
                except OSError:
                    pass
            rows.append((b, n, t))
        rows.sort(reverse=True)
        print(f"{'package':30} {'files':>7} {'MB':>9}", file=sys.stderr)
        for b, n, t in rows[:15]:
            print(f"{t:30} {n:7d} {b / 1e6:9.1f}", file=sys.stderr)
        print(
            f"{'TOTAL (%d packages)' % len(rows):30} "
            f"{sum(r[1] for r in rows):7d} {sum(r[0] for r in rows) / 1e6:9.1f}",
            file=sys.stderr,
        )
        print(f"site={site}", file=sys.stderr)

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    try:
        for t in tops:
            out.write(t + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
