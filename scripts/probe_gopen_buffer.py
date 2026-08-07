#!/usr/bin/env python
"""Does the tar read buffer size matter on DAOS? -- a candidate owner of the tail.

WHY THIS EXISTS. The dataload tail is now known to be UPSTREAM of decode: the
live nw=2 profile (job 8741045) measured worst decode 4.22 s against worst gap
53.84 s, and 8.7% of samples exceed the 10.74 s bs=2 pure-decode ceiling. So the
cost is in "get the bytes", not "turn the bytes into frames" -- but "upstream"
is a direction, not a mechanism, and naming a mechanism without evidence is the
error [[no-lazy-cause-labels]] exists to prevent. This probe tests ONE specific,
falsifiable mechanism.

THE MECHANISM. Our read path is
    url_opener (tariterators.py:98) -> gopen.gopen (gopen.py:524)
and because shard URLs are plain paths with no scheme, gopen falls into its
no-scheme branch (gopen.py:579-583):

    bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
    return open(url, mode, buffering=bufsize)

`buffering=-1` makes CPython pick the file's `st_blksize`. Meanwhile
`tar_file_iterator` opens the stream as `tarfile.open(fileobj=..., mode="r|*")`,
and stream mode requests **10240 bytes per read**, measured, regardless of what
buffering the underlying object was given. So the syscall size reaching the
filesystem is set entirely by `st_blksize`:

    st_blksize   fs reads for a 12 MB tar   (measured locally)
          4096                        2348
          8192                        1468
         65536                         185
       1048576                          13

If dfuse reports 4 KB, a 1.5 GB shard becomes ~190 k read RPCs through the DAOS
client instead of ~1.5 k at 1 MB. That is a tail story and not merely a body
story: the per-read cost does not have to be pathological, only its COUNT has to
be large enough that the max over ~200 k draws is far from the median. It also
fits the two facts that killed the other candidates -- it is invisible offline
(local page cache, no DAOS client) and it is per-node-client rather than fabric.

WHAT WOULD REFUTE IT. If DAOS already reports a large st_blksize, Python is
already issuing big reads and there is nothing here -- the mechanism is dead and
the tail's owner is still open. That is a real possible outcome and this probe
is written to report it plainly rather than to confirm the hypothesis.

READ THE OUTPUT AS:
  * st_blksize on the DAOS mount is the whole ballgame. Large => refuted.
  * The GOPEN_BUFFER sweep is the intervention. A flat curve means buffer size
    does not gate DAOS read cost (dfuse is doing its own readahead) -- also a
    refutation, and a more decisive one than st_blksize alone.
  * Lustre is a CONTROL, not a target. It says whether any effect is specific to
    the DAOS client or is generic to this Python read path.

This is a READ-ONLY probe: it opens existing shards and times them. It writes
nothing, needs 1 node, and must run ON a compute node -- st_blksize and read
cost on a login node are not the numbers that matter.
"""

import argparse
import io
import os
import sys
import time

DAOS_MNT = os.environ.get("DAOS_MNT", "/tmp/AuroraGPT/vjepa_surg_wds")
LUSTRE_ROOT = "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded"


def _find_shards(root, source, n):
    """First n .tar files under root/source.

    os.listdir, never glob -- glob hangs on dfuse.

    Returns n DISTINCT shards because every timed trial must read a shard this
    process has not touched. See _time_tar's note on cold reads.
    """
    d = os.path.join(root, source)
    try:
        names = sorted(x for x in os.listdir(d) if x.endswith(".tar"))
    except OSError as e:
        return None, f"{d}: {e}"
    if len(names) < n:
        return None, f"{d}: need {n} shards, found {len(names)}"
    return [os.path.join(d, x) for x in names[:n]], None


class _CountingRaw(io.RawIOBase):
    """Sits BELOW the BufferedReader and counts what actually reaches the fs.

    io.BufferedReader calls readinto() with a buffer sized by its buffer_size,
    so len(b) here IS the request size the kernel/dfuse sees. Counting at the
    BufferedReader level instead would only ever report tarfile's fixed 10240.
    """

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.n = 0
        self.bytes = 0

    def readinto(self, b):
        data = os.read(self.fd, len(b))
        self.n += 1
        self.bytes += len(data)
        b[: len(data)] = data
        return len(data)

    def readable(self):
        return True

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


def _time_tar(path, bufsize, count_reads=False, limit=None):
    """Iterate the tar exactly as webdataset does; return (seconds, members, reads).

    COLD READS ONLY -- the caller must pass a shard this process has not read.
    An earlier draft of this probe timed two passes over one shard and reported
    the second, reasoning that the first "warms the cache". That would have
    measured the page cache and reported a flat curve, i.e. a FALSE REFUTATION
    of the very hypothesis under test: buffer size cannot matter when the bytes
    never leave RAM. Training reads each shard once per epoch, so the read that
    matters is always cold, and the probe has to be cold too.
    """
    import tarfile

    raw = _CountingRaw(path) if count_reads else None
    t0 = time.time()
    if raw is not None:
        fobj = io.BufferedReader(raw, buffer_size=bufsize)
    else:
        # The real path: gopen's open(url, "rb", buffering=bufsize).
        fobj = open(path, "rb", buffering=bufsize)
    n = 0
    try:
        stream = tarfile.open(fileobj=fobj, mode="r|*")
        for ti in stream:
            if ti.isreg():
                # Read the member body. Iterating TarInfos alone would skip
                # ahead and measure seeks, not the sequential read the loader
                # actually performs when it extracts each sample.
                f = stream.extractfile(ti)
                if f is not None:
                    while f.read(1 << 20):
                        pass
                n += 1
                if limit and n >= limit:
                    break
    finally:
        try:
            fobj.close()
        except OSError:
            pass
    dt = time.time() - t0
    return dt, n, (raw.n if raw is not None else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cholec80_g16")
    ap.add_argument("--members", type=int, default=24,
                    help="members to read per trial; caps runtime on big shards")
    # -1 MUST be in this list and MUST come first. It is the value training
    # actually runs at (GOPEN_BUFFER unset -> gopen passes buffering=-1 ->
    # CPython uses the mount's st_blksize), and the first run of this probe
    # omitted it -- sweeping 4096..4 MB and reporting a 42x DAOS speedup that
    # was really the gap between a setting nobody uses and the one already in
    # force. Without the -1 row the sweep cannot distinguish "big lever" from
    # "already at the top of the curve", which are opposite conclusions.
    ap.add_argument("--bufsizes", default="-1,4096,65536,1048576,4194304")
    ap.add_argument("--repeats", type=int, default=3,
                    help="trials per bufsize, each on its own cold shard")
    ap.add_argument("--require-daos", action="store_true",
                    help="hard-fail if the DAOS mount yields no shards")
    args = ap.parse_args()

    bufs = [int(x) for x in args.bufsizes.split(",")]
    # Each (bufsize, repeat) trial burns one never-before-read shard, plus one
    # for the read-count part. Cold reads are the whole point; see _time_tar.
    need = len(bufs) * args.repeats + 1

    print("=" * 74)
    print("GOPEN_BUFFER / st_blksize probe -- is tar read granularity the tail?")
    print("=" * 74)
    print(f"host={os.uname().nodename}  source={args.source}  "
          f"members/trial={args.members}  repeats={args.repeats}")
    print(f"COLD reads: every trial uses a distinct shard ({need} needed/path)")
    print()

    targets = []
    for label, root in (("DAOS", DAOS_MNT), ("Lustre", LUSTRE_ROOT)):
        paths, err = _find_shards(root, args.source, need)
        if paths is None:
            print(f"[{label}] SKIP: {err}")
            continue
        targets.append((label, paths))

    if not targets:
        print("FATAL: no shards found on either path -- nothing to measure.")
        return 1
    if args.require_daos and not any(lbl == "DAOS" for lbl, _ in targets):
        # The DAOS column is the entire reason this runs on a compute node.
        # Without this guard a missing mount degrades to a Lustre-only run that
        # prints a one-line SKIP and then a full, plausible-looking table --
        # the same shape as job 8730443, which exited 0 with correct headers and
        # zero numbers and read as a success.
        print("FATAL: --require-daos set and no DAOS shards found. The Lustre")
        print("       column alone cannot answer this question; not reporting.")
        return 2

    # --- Part 1: what does buffering=-1 actually pick? ---------------------
    print("--- st_blksize (what Python's default buffering=-1 resolves to) ---")
    print(f"{'path':>8} {'st_blksize':>12} {'size MB':>10}   verdict")
    # No verdict column here. The first run of this probe printed
    # "LARGE -> mechanism REFUTED here" for DAOS (st_blksize 2 MB) on the same
    # run whose intervention arm measured 36-44x, i.e. the heuristic contradicted
    # the measurement sitting twenty lines below it. st_blksize says where the
    # DEFAULT sits on the curve; it says nothing about the curve's shape. Only
    # Part 3 decides, and the -1 row there is what locates the default on it.
    for label, paths in targets:
        st = os.stat(paths[0])
        print(f"{label:>8} {st.st_blksize:>12} {st.st_size/1e6:>10.1f}   "
              f"(= what the -1 row in Part 3 runs at)")
    print()

    # --- Part 2: does the read count actually track buffer size here? ------
    # Read COUNT is deterministic given (bufsize, bytes read), so unlike the
    # timing part this one may reuse a single shard -- caching changes its speed
    # but not how many read() calls tarfile's 10240-byte requests turn into.
    print("--- fs-level read() count vs buffer size (first target) ---")
    label, paths = targets[0]
    probe_path = paths[-1]
    print(f"{'bufsize':>10} {'fs reads':>10}")
    for b in bufs:
        _, _, reads = _time_tar(probe_path, b, count_reads=True, limit=args.members)
        print(f"{b:>10} {reads:>10}")
    print()

    # --- Part 3: the intervention, on the real gopen path ------------------
    print("--- wall time through the REAL gopen path: open(buffering=N) ---")
    print("(this is what GOPEN_BUFFER=N would do in training; COLD, distinct shards)")
    print(f"{'path':>8} {'bufsize':>10} {'median s':>10} {'trials':>7} "
          f"{'members':>8} {'vs 4096':>9}")
    for label, paths in targets:
        nxt = 0
        base = None
        for b in bufs:
            ts, members = [], 0
            for _ in range(args.repeats):
                dt, members, _ = _time_tar(paths[nxt], b, limit=args.members)
                nxt += 1
                ts.append(dt)
            ts.sort()
            med = ts[len(ts) // 2]
            if base is None:
                base = med
            print(f"{label:>8} {b:>10} {med:>10.2f} {len(ts):>7} {members:>8} "
                  f"{base / max(med, 1e-9):>8.2f}x")
    print()
    print("CAVEAT: distinct shards means distinct CONTENT, so shard-to-shard")
    print("variance rides on every ratio. --repeats is the defence; if the")
    print("spread across repeats rivals the effect, the effect is not there.")
    print()
    print("=" * 74)
    print("HOW TO JUDGE")
    print("  Speedup >=1.5x at 1 MB vs 4 KB on DAOS  -> GOPEN_BUFFER is a real")
    print("     lever; next step is a paired live ladder rung, not a rollout.")
    print("  Flat curve on DAOS                      -> REFUTED. dfuse is doing")
    print("     its own readahead; the tail's owner remains unidentified and")
    print("     this mechanism should be written off explicitly.")
    print("  Effect on Lustre but not DAOS (or both) -> it is the Python read")
    print("     path, not the DAOS client. Note it, but it does not explain a")
    print("     tail that only appears when reading through DAOS.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
