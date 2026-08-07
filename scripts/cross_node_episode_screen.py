#!/usr/bin/env python3
"""Is the node-synchronous compute episode NODE-LOCAL? A screen over archived
multi-node runs, so the question does not cost a queue slot.

THE QUESTION, AND WHY IT WAS STUCK
----------------------------------
A 1-node rung (job 8741769, x4610c4s3b0n0) spent 20.3% of its post-warmup wall
in node-synchronous `fwd-context` episodes: dataload 0.00 s on every episode
iteration, the 12 tiles TIGHTER together than normal, the whole node slow at
once. The replication on a different node (8741810, x4104c2s1b0n0) showed
**zero** episodes -- twice, in the same allocation.

So the leading hypothesis is that it is node-local. Confirming it needed two 1n
rungs on DIFFERENT nodes in ONE allocation, and `scaling_ladder.sh` could not
express that until the `:node<K>` field landed: every rung sliced the first R
nodes of the allocation. Meanwhile the sampling was brutal -- 1 of 4 observed
rungs episodic, so a 2-rung test had ~19% chance of the one-clean-one-episodic
split the read needs, and a null would have been uninformative.

But the breadth this needs is already on disk. Every 16n and 64n arm ever run is
16 or 64 nodes observed CONCURRENTLY, in ONE allocation, same fabric hour --
which is exactly the design the task asked for, at 8-32x the width, for free.
This script reads those.

WHAT COUPLING DOES TO THIS, AND THE STATISTIC THAT SURVIVES IT
--------------------------------------------------------------
At >1 node the nodes are not independent observers. `target_encoder` and the
predictor are HSDP-wrapped, so there is an all-gather INSIDE forward: when one
node stalls, every other node's forward column inflates while it waits. Job
8741855 measured exactly this at 2n -- the slow node showed it in `fwd-target`
and the other node's `backward` inflated in sympathy.

Two consequences, and they are not symmetric:

  * **Coincidence is NOT interpretable here.** "Do episodes on different nodes
    happen at the same iteration" -- read (b) of the original task -- is
    manufactured by coupling at any node count above 1. A lockstep collective
    guarantees co-occurrence whatever the cause. This script therefore does not
    report a shared-vs-independent verdict, and a multi-node run cannot answer
    read (b) at all. Only concurrent, UNCOUPLED 1n sub-worlds can
    (`1:node0 1:node1`), which is what the `:node<K>` field is for.

  * **Concentration IS interpretable**, via argmax. At each episodic iteration,
    take the node with the HIGHEST median fwd-context. Under coupling the
    waiters inflate too, but they inflate *less* than the source -- the source
    pays the stall and the wait, the waiters pay only the wait. So argmax is a
    source estimator that coupling degrades but does not invert. Then:

        node-local + rotating  ->  argmax spread ~uniformly over nodes
        one bad node           ->  argmax concentrated on it
        a global/external cause->  argmax uniform AND rates high everywhere

    The uniform cases are separated by the RATE, not the argmax: a rotating
    node-local effect is rare per node, a global one is common everywhere.

THE NULL THIS SCREEN CAN RETURN, AND WHAT IT MEANS
---------------------------------------------------
If no arm shows node-synchronous episodes at all, that is "not reproduced at
this depth", NOT "refuted" ([[no-lazy-cause-labels]]). The archived multi-node
arms are 30-60 iterations, and the 1n original's rate only cleared 10% from
about iteration 150. A short arm structurally cannot see the effect, so this
screen is only trustworthy when it finds something. Arms too short to be
informative are reported as SHORT rather than as clean.

The within-node tightness gate is what stops this counting stragglers. The 16n
arm 8741594/n16_nw2 fires a naive fwdc threshold on 2 of 46 iterations, but its
across-rank spread is 1.84 vs 1.29 normal -- a laggard, not a node slowing as a
unit. `fwdc_episode_scan.py` already rejects that arm for the same reason; this
script applies the gate PER NODE so one straggler-afflicted node cannot
disqualify the other fifteen.
"""
import argparse
import collections
import glob
import os
import re
import statistics as st

# 0=epoch 1=itr 2=loss 3=iter 4=gpu 5=dload 6=fwdtgt 7=fwdctx 8=bwd 9=opt
# 10=ema ... 16=barrier 17=host-avail-mib 18=rss-mib
COL = {"iter": 3, "dload": 5, "fwdt": 6, "fwdc": 7, "bwd": 8}
EVENT_COLS = {4, 6, 7, 8, 9, 10}
WRAP_MS = 343597.0  # 2**32 * 80 ns; a wrapped row is a SLOW row -- unwrap it

PPN = 12
WARMUP_FRAC = 0.24
SPIKE_MULT = 1.5
# Within-node tightness. The 1n episodes ran max/min = 1.026 across tiles
# (TIGHTER than the 1.057 of normal iterations); a straggler gives a large
# spread. 1.30 is deliberately loose -- this is a screen, and the argmax read
# below is what carries the claim, so the gate should not be the thing that
# decides the answer.
TIGHT_MAX = 1.30
# Below this many post-warmup iterations an arm cannot see an effect whose rate
# only clears 10% after ~150 iterations at 1n, so it is reported SHORT.
MIN_ITERS = 25


def rank_hosts(d):
    """rank -> physical hostname, from the hang-watchdog's `host=` field.

    Worth the extra file reads: which node a rank index maps to is PBS's choice,
    and a per-node claim that cannot name its hosts is not checkable later. Also
    validates that rank//PPN is the right grouping rather than assuming it.
    """
    hosts = {}
    for f in sorted(glob.glob(os.path.join(d, "rank.*.out"))):
        m = re.search(r"rank\.(\d+)\.out$", f)
        if not m:
            continue
        try:
            for ln in open(f, errors="replace"):
                h = re.search(r"rank=\d+ host=(x\S+?)\)", ln)
                if h:
                    hosts[int(m.group(1))] = h.group(1)
                    break
        except OSError:
            continue
    return hosts


def read_run(d):
    """-> {(epoch, itr): {rank: {phase: seconds}}}, last segment only."""
    per = collections.defaultdict(dict)
    seg_of = {}
    for f in sorted(glob.glob(os.path.join(d, "log_r*.csv"))):
        m = re.search(r"log_r(\d+)\.csv$", f)
        if not m:
            continue
        rk = int(m.group(1))
        seg = -1
        try:
            lines = open(f, errors="replace").read().splitlines()
        except OSError:
            continue
        for ln in lines:
            if ln.startswith("epoch,"):
                seg += 1
                continue
            p = ln.split(",")
            if len(p) <= max(COL.values()):
                continue
            try:
                key = (int(p[0]), int(p[1]))
                row = {}
                for k, c in COL.items():
                    v = float(p[c])
                    if v < 0 and c in EVENT_COLS:
                        v += WRAP_MS
                    row[k] = v / 1000.0
            except ValueError:
                continue
            per[key][rk] = row
            seg_of[key] = max(seg, seg_of.get(key, seg))
    if not per:
        return {}
    last = max(seg_of.values())
    return {k: v for k, v in per.items() if seg_of[k] == last}


def analyze(d, verbose=True):
    per = read_run(d)
    if not per:
        return None
    nr = max(len(v) for v in per.values())
    if nr < 2 * PPN:
        return dict(dir=d, status="1N", note="single node -- no cross-node read")
    keys = sorted(k for k, v in per.items() if len(v) == nr)
    if not keys:
        return None
    post = keys[int(WARMUP_FRAC * len(keys)):]
    nodes = sorted({rk // PPN for rk in per[keys[0]]})

    hosts = rank_hosts(d)
    hostname = {}
    grouping_ok = True
    if hosts:
        for n in nodes:
            hs = {hosts[r] for r in range(n * PPN, (n + 1) * PPN) if r in hosts}
            if len(hs) == 1:
                hostname[n] = hs.pop()
            elif hs:
                grouping_ok = False
                hostname[n] = "|".join(sorted(hs))

    if len(post) < MIN_ITERS:
        return dict(dir=d, status="SHORT", n_nodes=len(nodes), n_iters=len(post))

    fc_med = st.median(r["fwdc"] for k in post for r in per[k].values())
    thr = SPIKE_MULT * fc_med

    # Per (iteration, node): the node's own median fwd-context, and whether its
    # tiles moved together. Median not mean: one straggler tile must not carry
    # the node over the threshold -- that is the failure mode the tightness gate
    # exists for, and using the mean would let it in through the front door.
    ep_by_node = collections.Counter()
    argmax_by_node = collections.Counter()
    ep_iters = []
    for k in post:
        best, best_n = -1.0, None
        hits = []
        for n in nodes:
            v = [per[k][r]["fwdc"] for r in range(n * PPN, (n + 1) * PPN)
                 if r in per[k]]
            if len(v) < PPN:
                continue
            m = st.median(v)
            tight = (max(v) / min(v)) if min(v) > 1e-9 else float("inf")
            if m > best:
                best, best_n = m, n
            if m > thr and tight <= TIGHT_MAX:
                hits.append(n)
        for n in hits:
            ep_by_node[n] += 1
        if hits:
            ep_iters.append((k, hits))
            if best_n is not None:
                argmax_by_node[best_n] += 1

    n_ep = len(ep_iters)
    res = dict(dir=d, status="OK", n_nodes=len(nodes), n_iters=len(post),
               n_ep=n_ep, thr=thr, ep_by_node=ep_by_node,
               argmax_by_node=argmax_by_node, hostname=hostname,
               grouping_ok=grouping_ok)

    if not verbose:
        return res

    print(f"\n{d}")
    print(f"  {nr} ranks / {len(nodes)} nodes, {len(post)} post-warmup iters, "
          f"fwdc threshold {thr:.3f} s")
    if hosts and not grouping_ok:
        print("  WARNING: rank//12 does not partition cleanly by host. Node "
              "indices below are\n           NOT physical nodes; treat the "
              "per-node table as unreliable.")
    if n_ep == 0:
        print("  NO node-synchronous episodes (tightness-gated) on any node.")
        print("  This is 'not reproduced at this depth', NOT a refutation: the")
        print(f"  1n original's rate only cleared 10% after ~150 iters and this")
        print(f"  arm has {len(post)}.")
        return res

    print(f"  {n_ep} of {len(post)} iterations have >=1 episodic node "
          f"({100*n_ep/len(post):.1f}%)")
    print(f"\n  {'node':<6} {'host':<16} {'episodes':>9} {'rate':>7} "
          f"{'argmax':>7}")
    for n in nodes:
        c, a = ep_by_node[n], argmax_by_node[n]
        if not (c or a):
            continue
        print(f"  {n:<6} {hostname.get(n,'?'):<16} {c:>9} "
              f"{c/len(post):>7.3f} {a:>7}")

    # Concentration. Under a rotating node-local effect the source rotates, so
    # argmax spreads; under one bad node it piles up. Reported as the share held
    # by the top node against the 1/len(nodes) a uniform draw would give.
    if argmax_by_node:
        top_n, top_c = argmax_by_node.most_common(1)[0]
        share = top_c / sum(argmax_by_node.values())
        unif = 1.0 / len(nodes)
        print(f"\n  argmax concentration: top node {top_n} "
              f"({hostname.get(top_n,'?')}) holds {share:.1%} of "
              f"{sum(argmax_by_node.values())} episodic iterations; "
              f"uniform would be {unif:.1%}")
        if share > 4 * unif and top_c >= 5:
            print("  -> CONCENTRATED. Consistent with a node-local effect "
                  "attached to one node.")
        elif len(argmax_by_node) >= max(3, len(nodes) // 4):
            print("  -> SPREAD across nodes. Not one bad node. Distinguish "
                  "'rotating node-local'\n     from 'global' by the RATE "
                  "column above, not by this line.")
        else:
            print("  -> too few episodic iterations to call concentration; "
                  "report the counts.")
    print("\n  Coincidence is deliberately NOT reported. Above 1 node the HSDP")
    print("  all-gather inside forward makes co-occurrence automatic whatever "
          "the cause,\n  so a shared-vs-independent verdict from this data "
          "would be an artifact of\n  coupling. That read needs concurrent, "
          "UNCOUPLED 1n sub-worlds "
          "(`1:node0 1:node1`).")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--scan", help="walk a checkpoint root instead")
    a = ap.parse_args()
    dirs = list(a.dirs)
    if a.scan:
        for root, _, files in os.walk(a.scan):
            if any(f.startswith("log_r") and f.endswith(".csv") for f in files):
                dirs.append(root)
    out = []
    for d in sorted(set(dirs)):
        r = analyze(d)
        if r:
            out.append(r)
    ok = [r for r in out if r.get("status") == "OK"]
    if len(ok) > 1:
        print("\n" + "=" * 72)
        print("SUMMARY (multi-node arms only)")
        for r in ok:
            tot = sum(r["argmax_by_node"].values())
            top = (r["argmax_by_node"].most_common(1)[0] if tot else (None, 0))
            print(f"  {os.path.basename(os.path.dirname(r['dir']))}/"
                  f"{os.path.basename(r['dir']):<28} "
                  f"{r['n_nodes']:>3}n {r['n_iters']:>4}it  "
                  f"ep_iters={r['n_ep']:<4} top_node_share="
                  f"{(top[1]/tot if tot else 0):.0%}")
    short = [r for r in out if r.get("status") == "SHORT"]
    if short:
        print(f"\n  {len(short)} arm(s) skipped as too short (<{MIN_ITERS} "
              f"post-warmup iters) -- a null from those would be uninformative.")


if __name__ == "__main__":
    main()
