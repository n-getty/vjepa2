"""Self-resubmitting PBS chain to run the scaling sweep unattended across 1h debug-scaling windows.

debug-scaling caps walltime at 1h and max_run=1 job/user, and prod needs >=256 nodes (unusable for
our <=16-node cells). So a long sweep must be a CHAIN of 1h jobs, each resuming from latest.pth.tar.

This differs from the hold-node pattern (which needs a live driver to re-trigger): each job here
RESUBMITS ITS OWN SUCCESSOR just before it ends, so the chain self-perpetuates with no live session.
The trainer resumes because every real config has load_checkpoint:True + CHECKPOINT_FREQ=1 (per-epoch
latest.pth.tar); first launch finds no checkpoint and starts from scratch (train.py:406-408).

Each job:
  1. splits its held nodefile into per-cell contiguous blocks (from each cell's _launch.json),
  2. launches one MPI world per cell (private hostfile, own port, cell env), backgrounds, waits ~50min,
  3. checks a STOP sentinel and a per-cell DONE test (final epoch reached) — drops finished cells,
  4. if any cell is unfinished AND no STOP AND chain depth < max, qsub the next link.

Control files in <ctrl>:
  chain.cfglist   the newline-separated cell YAMLs this chain runs (written by `start`)
  chain_depth     integer, incremented each link
  STOP            touch to halt the chain after the current link
  link_<n>.log    PBS stdout of link n
  cell_<slug>.log per-cell trainer log (overwritten each link; the csv in the run folder accumulates)

Subcommands:
  start   — write cfglist, submit link 1.
  stop    — touch STOP (chain halts after current link finishes).
  status  — show chain depth, per-cell latest epoch vs target, running job.
"""

import argparse
import glob
import json
import os
import subprocess

AURORA_ENV = """ulimit -c unlimited
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export FI_PROVIDER=cxi
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128\""""


def _cells_from_cfglist(ctrl):
    with open(os.path.join(ctrl, "chain.cfglist")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _target_epochs(cfg):
    import yaml
    return yaml.safe_load(open(cfg))["optimization"]["epochs"]


def _run_folder(cfg):
    import yaml
    return yaml.safe_load(open(cfg))["folder"]


def _current_epoch(cfg):
    """Epoch of the LAST valid row in log_r0.csv (0 if none) = real current
    training position. NOT max(epoch): a cell that restarted from scratch (e.g.
    after a corrupt-checkpoint recovery) has stale high-epoch rows from its prior
    run followed by fresh low-epoch rows; max() would report the stale peak and
    (a) mis-display progress, (b) FREEZE this cell's contribution to the crash-
    loop guard's progress delta until real training passes the stale peak — a
    false-stall trap when it's the sole unfinished cell. Last-row is correct for
    monotonic runs, normal resumes, AND scratch restarts.

    NOTE: the log epoch is 1-INDEXED and written DURING the epoch, so a value of N
    means "epoch N is IN PROGRESS", NOT "N epochs completed". Use this for
    progress-delta / display only. For the DONE decision use _completed_epochs()."""
    fldr = _run_folder(cfg)
    csv = os.path.join(fldr, "log_r0.csv")
    if not os.path.exists(csv):
        return 0
    last = 0
    with open(csv) as f:
        next(f, None)
        for line in f:
            try:
                last = int(line.split(",")[0])
            except (ValueError, IndexError):
                pass  # skip garbled/partial line, keep prior good value
    return last


def _completed_epochs(cfg):
    """Number of FULLY-COMPLETED, resumable epochs — the authoritative DONE signal.
    Uses the checkpoint's `epoch` field: the trainer calls save_checkpoint(epoch+1)
    only AFTER an epoch's full ipe iters complete, so ckpt.epoch == count of finished
    epochs. This is NOT the same as _current_epoch (log last-row), which counts the
    IN-PROGRESS epoch: a single-epoch cell (target=1) writes log-epoch '1' at iter 0
    and _current_epoch would call it done at ~30/500 iters -> a garbage 6%-trained
    high-N point. Falls back to 0 if no/corrupt checkpoint (cell must (re)train)."""
    import torch
    ckpt = os.path.join(_run_folder(cfg), "latest.pth.tar")
    if not os.path.exists(ckpt):
        return 0
    try:
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        return int(d.get("epoch", 0) or 0)
    except Exception:
        return 0  # corrupt/unreadable -> treat as not-done (will retrain/recover)


# Liveness lock via log mtime: the trainer appends to log_r0.csv every iteration, so a recently-touched
# log means ANOTHER launcher (e.g. a parallel capacity job) is actively training this cell. This lets
# two independent jobs (debug-scaling chain + a capacity hedge) share one run set WITHOUT colliding on
# latest.pth.tar — each only claims cells whose log is stale. No trainer change, no separate lockfile.
LIVENESS_STALE_S = int(os.environ.get("SCALING_LIVENESS_STALE_S", "900"))  # 15 min


def _is_live(cfg, stale_s=LIVENESS_STALE_S):
    """True only if this cell's log_r0.csv is ACTIVELY GROWING right now (another
    launcher is training it). We must NOT rely on 'mtime within stale_s': the chain
    kills its own cells at walltime, and a fast-scheduled successor can start within
    minutes, so the just-killed logs are still 'recent' -> every cell SKIPPED ->
    empty link -> zero progress -> crash-loop guard false-trips (observed: a 2-sec
    link that skipped all 6 cells 9 min after the prior link's SIGTERM). Instead
    sample (size, mtime) twice ~probe_s apart: a genuinely training cell appends
    iteration rows in that window; a dead/killed cell is static. Quick static check
    first (mtime older than stale_s => definitely not live) to avoid the sleep on
    the common case."""
    import time
    csv = os.path.join(_run_folder(cfg), "log_r0.csv")
    if not os.path.exists(csv):
        return False
    try:
        st1 = os.stat(csv)
        if (time.time() - st1.st_mtime) >= stale_s:
            return False  # old enough to be certainly dead; no need to probe
        probe_s = float(os.environ.get("SCALING_LIVENESS_PROBE_S", "3"))
        time.sleep(probe_s)
        st2 = os.stat(csv)
        return (st2.st_size, st2.st_mtime) != (st1.st_size, st1.st_mtime)
    except OSError:
        return False


def build_link_script(ctrl, nodes, account, partition, code_folder, cpus_per_task,
                      max_depth, softlimit_s, python_exe):
    """PBS script for ONE chain link. Reads chain.cfglist, runs unfinished cells, resubmits successor."""
    return f"""#!/bin/bash -l
#PBS -N sweepchain_{nodes}n
#PBS -l select={nodes}
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q {partition}
#PBS -A {account}
#PBS -j oe
#PBS -o {ctrl}/link.log

set -o pipefail   # no set -u (lmod trap); no set -e (a cell fail must not abort the chain resubmit)

CTRL="{ctrl}"
CODE="{code_folder}"
cd "$CODE"
{AURORA_ENV}

DEPTH=$(cat "$CTRL/chain_depth" 2>/dev/null || echo 0)
DEPTH=$((DEPTH+1)); echo "$DEPTH" > "$CTRL/chain_depth"
LOG="$CTRL/link_$DEPTH.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== CHAIN LINK $DEPTH start $(date) nodes={nodes} ==="
cp "$PBS_NODEFILE" "$CTRL/nodefile.full"
cat "$CTRL/nodefile.full"

# ---- decide which cells still need work; assign node blocks; launch ----
# Python emits a bash launch block for the unfinished cells (private hostfiles, per-cell env).
LAUNCH=$({python_exe} -m scaling.overnight_chain _emit --ctrl "$CTRL" --cpus {cpus_per_task})
if [ -z "$LAUNCH" ]; then
  echo "ALL CELLS COMPLETE — chain done, not resubmitting."
  touch "$CTRL/CHAIN_COMPLETE"
  exit 0
fi
echo "$LAUNCH" > "$CTRL/_launch_block_$DEPTH.sh"

# snapshot total epochs BEFORE this link, to detect whether the link made progress (crash-loop guard).
EP_BEFORE=$({python_exe} -m scaling.overnight_chain _progress --ctrl "$CTRL")
echo "epochs-before=$EP_BEFORE"

# ---- run the cells for up to ~{softlimit_s}s, then let walltime handle the rest ----
# NOTE: resubmit is END-ONLY (not early). An EARLY resubmit (at link start) races debug-scaling's
# "max 1 job in Q state per user" limit — if any of our jobs is still queued, the early qsub fails
# ("would exceed per-user limit of jobs in Q state") and the chain silently drops (observed depth 6).
# At END, THIS job is in R/E state (not Q), so its own slot doesn't count against the Q limit and the
# successor qsub succeeds. We lose pure walltime-survival (if PBS hard-kills before we reach the END
# qsub), but the {softlimit_s}s timeout returns control well before the 1h walltime, so END runs.
timeout {softlimit_s} bash "$CTRL/_launch_block_$DEPTH.sh"
echo "=== CHAIN LINK $DEPTH cells returned $(date) ==="

# did we make progress this link?
EP_AFTER=$({python_exe} -m scaling.overnight_chain _progress --ctrl "$CTRL")
echo "epochs-after=$EP_AFTER (before=$EP_BEFORE)"
if [ "$EP_AFTER" -gt "$EP_BEFORE" ]; then
  touch "$CTRL/HEARTBEAT_OK"
fi

# END resubmit, gated on progress (crash-loop guard). The per-user "jobs in Q" limit is transient BUT
# can be held by SIBLING jobs (other experiments' qsubs) for a LONG time — a 6x60s (6 min) retry is far
# too short: on 2026-07-11 a giant-smoke + asformer + FLASH sibling set held the Q limit, all 6 retries
# failed, and the chain dropped silently for the rest of the night. A dropped chain is catastrophic
# (unattended, no progress until a human notices), so retry HARD: up to RESUB_TRIES attempts spaced
# RESUB_GAP_S apart. Default 40 tries x 60s = ~40 min of retrying, which comfortably outlasts any
# sibling job's time-in-Q (they start running and free the slot) while staying inside PBS walltime
# (this runs after the {softlimit_s}s cell timeout, with the full 1h-softlimit headroom to spare).
if [ ! -f "$CTRL/STOP" ] && [ ! -f "$CTRL/CHAIN_COMPLETE" ] && [ "$DEPTH" -lt {max_depth} ]; then
  if [ "$EP_AFTER" -gt "$EP_BEFORE" ] || [ "$DEPTH" -eq 1 ]; then
    ok=0
    RESUB_TRIES=${{SCALING_RESUB_TRIES:-40}}
    RESUB_GAP_S=${{SCALING_RESUB_GAP_S:-60}}
    for attempt in $(seq 1 $RESUB_TRIES); do
      out=$(qsub "$CTRL/link.pbs" 2>&1)
      echo "$out" > "$CTRL/next_jobid_$DEPTH.txt"
      if echo "$out" | grep -q "aurora-pbs"; then
        echo "successor queued at END (progress=$((EP_AFTER-EP_BEFORE)) depth=$DEPTH attempt=$attempt): $out"; ok=1; break
      fi
      echo "qsub attempt $attempt/$RESUB_TRIES failed ($out) — retrying in ${{RESUB_GAP_S}}s"; sleep $RESUB_GAP_S
    done
    if [ "$ok" -eq 0 ]; then
      echo "SUCCESSOR QSUB FAILED after $RESUB_TRIES tries — chain dropped; liveness watchdog must catch it."
      touch "$CTRL/CHAIN_QSUB_FAILED"
    fi
  else
    echo "NO PROGRESS at depth $DEPTH >=2 — CRASH-LOOP GUARD, not resubmitting."
    touch "$CTRL/CHAIN_STALLED"
  fi
fi
echo "=== CHAIN LINK $DEPTH end $(date) ==="
"""


def emit_launch_block(ctrl, cpus_per_task, nodefile="nodefile.full", max_nodes=None):
    """Print a bash block launching UNFINISHED, UNCLAIMED cells on contiguous node blocks of `nodefile`.
    Empty if nothing to claim. `nodefile` lets a parallel capacity job use its own nodefile.cap +
    private nf split files (prefixed by the nodefile stem) so it doesn't clobber the debug chain's.
    `max_nodes` caps how many cells are packed (so the block fits the job's allocation)."""
    stem = nodefile.replace("nodefile.", "").replace("nodefile", "full") or "full"  # full|cap for private nf_ files
    cells = _cells_from_cfglist(ctrl)
    todo = []
    for cfg in cells:
        # DONE = COMPLETED epochs (ckpt.epoch), NOT in-progress log epoch. A single-epoch cell writes
        # log-epoch '1' at iter 0; _current_epoch would drop it as done at ~30/500 iters -> garbage
        # high-N point. _completed_epochs uses the checkpoint (saved only after a full epoch).
        cur, tgt = _completed_epochs(cfg), _target_epochs(cfg)
        if cur >= tgt:
            continue                       # already at target -> skip
        if _is_live(cfg):
            print(f"# SKIP {os.path.basename(cfg)}: log fresh (<{LIVENESS_STALE_S}s) — another job is "
                  f"training it; avoid checkpoint collision")
            continue                       # another launcher (e.g. capacity hedge) owns it right now
        todo.append(cfg)
    if not todo:
        return ""
    # BAKE the absolute ctrl path into the emitted block. It runs as a standalone `bash <file>` where
    # $CTRL from the outer PBS is NOT inherited (bug: link 1 wrote to /nf_* -> permission denied, 0
    # cells launched). Use an absolute CTRL literal so the block is self-contained.
    ctrl_abs = os.path.abspath(ctrl)
    lines = [f'CTRL="{ctrl_abs}"',
             'echo "launching %d unfinished cells (CTRL=$CTRL)"' % len(todo),
             'declare -a PIDS=()']
    cursor = 0
    for i, cfg in enumerate(todo):
        base = os.path.splitext(cfg)[0]
        spec = json.load(open(base + "_launch.json"))
        nodes, tiles = int(spec["nodes"]), int(spec["tiles"])
        # stop packing once we'd exceed this job's node allocation (capacity may be smaller than sum)
        if max_nodes is not None and cursor + nodes > max_nodes:
            lines.append(f'echo "# capacity full at {cursor} nodes; deferring remaining cells to next round"')
            break
        ppn = min(tiles, 12)
        lo, hi = cursor + 1, cursor + nodes
        cursor += nodes
        port = 29500 + i
        slug = os.path.basename(base)
        # Empty value => UNSET the var (not `export K=""`). oneCCL's env parser treats an EMPTY
        # CCL_KVS_MODE as a fatal enum error ("unexpected value: ''") — empty != unset — which crashed
        # every HSDP cell at the first FSDP all-gather (job 8667049). A per-cell env of "" is the
        # intent "neutralize the global AURORA_ENV default" (e.g. the ddp CCL_KVS_MODE=mpi that the
        # ofi/launcher=none HSDP path must not see), so emit a real `unset`.
        env = "".join(
            (f'  unset {k}\n' if v == "" else f'  export {k}="{v}"\n')
            for k, v in (spec.get("env") or {}).items()
        )
        # mpiexec flavor is dictated by the CCL transport, which differs by strategy:
        #   ddp  -> global AURORA_ENV sets CCL_PROCESS_LAUNCHER=pmix + CCL_ATL_TRANSPORT=mpi,
        #           so mpiexec MUST carry --pmi=pmix (oneCCL bootstraps its KVS over PMI/MPI).
        #   hsdp -> the per-cell env overrides to CCL_PROCESS_LAUNCHER=none + CCL_ATL_TRANSPORT=ofi
        #           + CCL_KVS_IFACE=hsn0 (oneCCL brings up its OWN KVS over the CXI fabric). That
        #           path does NOT use PMI; passing --pmi=pmix alongside launcher=none conflicts
        #           (validated: the 2n OFI smoke used plain `mpiexec` with no --pmi). So HSDP cells
        #           launch WITHOUT --pmi=pmix. Do not "simplify" this back to one invocation.
        pmi = "" if str(spec.get("dist_strategy", "ddp")).lower() == "hsdp" else "--pmi=pmix "
        lines += [
            f'NF{i}="$CTRL/nf_{stem}_{i}"; sed -n "{lo},{hi}p" "$CTRL/{nodefile}" > "$NF{i}"',
            f'H{i}=$(head -1 "$NF{i}")',
            f'echo "[{slug}] nodes {lo}..{hi} head $H{i} port {port} {spec["dist_strategy"]}"',
            "(",
            f'  export PBS_NODEFILE="$NF{i}"',
            f'  export MASTER_ADDR="$H{i}"; export MASTER_PORT={port}; export WORLD_SIZE={tiles}',
            (env.rstrip("\n") if env else "  :"),
            f'  mpiexec {pmi}-n {tiles} -ppn {ppn} --hostfile "$NF{i}" \\',
            f'      --cpu-bind depth --depth {cpus_per_task} \\',
            f'      python -m app.main_dist_aurora --train_mode \\',
            f'          --fname {os.path.abspath(cfg)} --params_path {os.path.abspath(cfg)} \\',
            f'      > "$CTRL/cell_{slug}.log" 2>&1',
            ") & PIDS+=($!)",
        ]
    lines += ['for p in "${PIDS[@]}"; do wait $p || true; done',
              'echo "link cells finished"']
    return "\n".join(lines)


def build_capacity_script(ctrl, nodes, account, code_folder, cpus_per_task, walltime_h, python_exe):
    """PBS for a PARALLEL capacity HEDGE: one long job (capacity allows up to 168h) that INTERNALLY
    loops re-emit->run until all cells are done or walltime. No PBS-chaining needed (capacity has long
    walltime, unlike debug-scaling's 1h). Shares configs/run-folders with the debug-scaling chain; the
    _emit liveness lock (log mtime) makes the two cooperate — each only claims cells the other isn't
    actively training, so no latest.pth.tar collision. Whichever job runs first drains the work; the
    other finds cells already done/live and idles or exits. Safe to run both; cancel the loser."""
    hh = int(walltime_h)
    return f"""#!/bin/bash -l
#PBS -N sweepcap_{nodes}n
#PBS -l select={nodes}
#PBS -l walltime={hh:02d}:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -A {account}
#PBS -j oe
#PBS -o {ctrl}/capacity.log

set -o pipefail

CTRL="{ctrl}"
CODE="{code_folder}"
cd "$CODE"
{AURORA_ENV}
cp "$PBS_NODEFILE" "$CTRL/nodefile.cap"
echo "=== CAPACITY HEDGE start $(date) nodes={nodes} walltime={hh}h ==="

# internal loop: while cells remain and no STOP, emit unclaimed/unfinished cells and run them.
round=0
while [ ! -f "$CTRL/STOP" ] && [ ! -f "$CTRL/CHAIN_COMPLETE" ]; do
  round=$((round+1))
  LAUNCH=$({python_exe} -m scaling.overnight_chain _emit --ctrl "$CTRL" --cpus {cpus_per_task} --nodefile nodefile.cap --max-nodes {nodes})
  if [ -z "$LAUNCH" ]; then
    # nothing to claim: either all done, or the debug chain is actively training everything left.
    if [ "$({python_exe} -m scaling.overnight_chain _remaining --ctrl "$CTRL")" -eq 0 ]; then
      echo "ALL CELLS COMPLETE — capacity hedge done."; touch "$CTRL/CHAIN_COMPLETE"; break
    fi
    echo "[cap round $round] no unclaimed cells (debug chain owns them) — idle 300s"; sleep 300; continue
  fi
  echo "$LAUNCH" > "$CTRL/_cap_block_$round.sh"
  echo "[cap round $round] running unclaimed cells $(date)"
  bash "$CTRL/_cap_block_$round.sh"
  sleep 10
done
echo "=== CAPACITY HEDGE end $(date) ==="
"""


def start(ctrl, configs, nodes, account, partition, code_folder, cpus_per_task, max_depth,
          softlimit_s, python_exe, dry_run):
    os.makedirs(ctrl, exist_ok=True)
    for f in ("STOP", "CHAIN_COMPLETE", "chain_depth"):
        p = os.path.join(ctrl, f)
        if os.path.exists(p):
            os.remove(p)
    with open(os.path.join(ctrl, "chain.cfglist"), "w") as f:
        f.write("\n".join(os.path.abspath(c) for c in configs) + "\n")
    script = build_link_script(ctrl, nodes, account, partition, os.path.abspath(code_folder),
                               cpus_per_task, max_depth, softlimit_s, python_exe)
    pbs = os.path.join(ctrl, "link.pbs")
    with open(pbs, "w") as f:
        f.write(script)
    print(f"wrote {pbs}; cfglist has {len(configs)} cells; nodes/link={nodes}")
    if dry_run:
        print("--dry-run: not submitting"); return
    out = subprocess.run(["qsub", pbs], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("qsub FAILED: " + out.stderr)
    print("chain link 1 submitted:", out.stdout.strip())


def status(ctrl):
    if not os.path.isdir(ctrl):
        raise SystemExit(f"no ctrl dir {ctrl}")
    depth = open(os.path.join(ctrl, "chain_depth")).read().strip() if os.path.exists(
        os.path.join(ctrl, "chain_depth")) else "0"
    print(f"chain depth: {depth}  STOP={'yes' if os.path.exists(os.path.join(ctrl,'STOP')) else 'no'}"
          f"  COMPLETE={'yes' if os.path.exists(os.path.join(ctrl,'CHAIN_COMPLETE')) else 'no'}")
    for cfg in _cells_from_cfglist(ctrl):
        done_ep, tgt = _completed_epochs(cfg), _target_epochs(cfg)
        cur = _current_epoch(cfg)  # in-progress log epoch, for display
        mark = "DONE" if done_ep >= tgt else ""
        print(f"  {os.path.basename(cfg):30s} epoch {cur}/{tgt} (done {done_ep}) {mark}")


def main():
    ap = argparse.ArgumentParser(description="Self-resubmitting scaling-sweep chain")
    # NOTE: no required=True — the login node's python3.6 argparse rejects it. Guard below instead.
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("start")
    s.add_argument("--ctrl", required=True)
    s.add_argument("--configs", required=True, help="glob of cell YAMLs")
    s.add_argument("--wave", default=None, help="substring filter on slug (keep only matching)")
    s.add_argument("--exclude", default=None, help="comma-list of substrings to DROP (e.g. giant,gigantic)")
    s.add_argument("--nodes", type=int, required=True, help="nodes per link (>= sum of cell nodes)")
    s.add_argument("--account", default="AuroraGPT")
    s.add_argument("--partition", default="debug-scaling")
    s.add_argument("--code-folder", default=os.getcwd())
    s.add_argument("--cpus-per-task", type=int, default=16)
    s.add_argument("--max-depth", type=int, default=24, help="max chain links (safety cap)")
    s.add_argument("--softlimit-s", type=int, default=3300, help="seconds to run cells before yielding")
    s.add_argument("--python", default="python", help="python exe used inside PBS")
    s.add_argument("--dry-run", action="store_true")

    st = sub.add_parser("status"); st.add_argument("--ctrl", required=True)
    sp = sub.add_parser("stop"); sp.add_argument("--ctrl", required=True)
    em = sub.add_parser("_emit"); em.add_argument("--ctrl", required=True); em.add_argument("--cpus", type=int, default=16)
    em.add_argument("--nodefile", default="nodefile.full"); em.add_argument("--max-nodes", type=int, default=None)
    pr = sub.add_parser("_progress"); pr.add_argument("--ctrl", required=True)
    rm = sub.add_parser("_remaining"); rm.add_argument("--ctrl", required=True)
    cap = sub.add_parser("capacity", help="submit a PARALLEL capacity hedge (shares run set; liveness-locked)")
    cap.add_argument("--ctrl", required=True); cap.add_argument("--nodes", type=int, required=True)
    cap.add_argument("--account", default="AuroraGPT"); cap.add_argument("--code-folder", default=os.getcwd())
    cap.add_argument("--cpus-per-task", type=int, default=16); cap.add_argument("--walltime-h", type=int, default=48)
    cap.add_argument("--python", default="python"); cap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    if not args.cmd:
        ap.error("a subcommand is required (start/status/stop/capacity/_emit/_progress/_remaining)")
    if args.cmd == "start":
        configs = sorted(glob.glob(args.configs))
        if args.wave:
            configs = [c for c in configs if args.wave in os.path.basename(c)]
        if args.exclude:
            drops = [x for x in args.exclude.split(",") if x]
            configs = [c for c in configs if not any(d in os.path.basename(c) for d in drops)]
        if not configs:
            raise SystemExit(f"no configs match {args.configs} wave={args.wave} exclude={args.exclude}")
        start(args.ctrl, configs, args.nodes, args.account, args.partition, args.code_folder,
              args.cpus_per_task, args.max_depth, args.softlimit_s, args.python, args.dry_run)
    elif args.cmd == "capacity":
        # reuses the SAME ctrl (cfglist + run folders) as an existing chain; liveness lock prevents
        # collision. Does NOT touch chain.cfglist — the chain must already have written it.
        if not os.path.exists(os.path.join(args.ctrl, "chain.cfglist")):
            raise SystemExit(f"{args.ctrl}/chain.cfglist missing — run `start` (the debug chain) first")
        script = build_capacity_script(args.ctrl, args.nodes, args.account,
                                       os.path.abspath(args.code_folder), args.cpus_per_task,
                                       args.walltime_h, args.python)
        pbs = os.path.join(args.ctrl, "capacity.pbs")
        with open(pbs, "w") as f:
            f.write(script)
        print(f"wrote {pbs}; capacity hedge on {args.nodes} nodes, {args.walltime_h}h")
        if args.dry_run:
            print("--dry-run: not submitting"); return
        out = subprocess.run(["qsub", pbs], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit("qsub FAILED: " + out.stderr)
        print("capacity hedge submitted:", out.stdout.strip())
    elif args.cmd == "status":
        status(args.ctrl)
    elif args.cmd == "stop":
        open(os.path.join(args.ctrl, "STOP"), "w").close()
        print(f"touched STOP in {args.ctrl} — chain + capacity hedge halt after current work")
    elif args.cmd == "_emit":
        print(emit_launch_block(args.ctrl, args.cpus, nodefile=args.nodefile, max_nodes=args.max_nodes))
    elif args.cmd == "_progress":
        print(sum(_current_epoch(cfg) for cfg in _cells_from_cfglist(args.ctrl)))
    elif args.cmd == "_remaining":
        # count cells not yet at target epoch (for the capacity loop's completion check).
        # Use COMPLETED epochs (ckpt), not in-progress log epoch — same reason as the emit DONE gate.
        print(sum(1 for cfg in _cells_from_cfglist(args.ctrl)
                  if _completed_epochs(cfg) < _target_epochs(cfg)))


if __name__ == "__main__":
    main()
