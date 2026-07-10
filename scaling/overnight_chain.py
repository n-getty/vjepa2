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
    """Highest epoch recorded in the run's log_r0.csv (0 if none)."""
    fldr = _run_folder(cfg)
    csv = os.path.join(fldr, "log_r0.csv")
    if not os.path.exists(csv):
        return 0
    last = 0
    with open(csv) as f:
        next(f, None)
        for line in f:
            try:
                last = max(last, int(line.split(",")[0]))
            except (ValueError, IndexError):
                pass
    return last


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

# END resubmit, gated on progress (crash-loop guard). Retry the qsub a few times in case a sibling
# job is momentarily still in Q (the per-user Q-limit is transient as other jobs start running).
if [ ! -f "$CTRL/STOP" ] && [ ! -f "$CTRL/CHAIN_COMPLETE" ] && [ "$DEPTH" -lt {max_depth} ]; then
  if [ "$EP_AFTER" -gt "$EP_BEFORE" ] || [ "$DEPTH" -eq 1 ]; then
    ok=0
    for attempt in 1 2 3 4 5 6; do
      out=$(qsub "$CTRL/link.pbs" 2>&1)
      echo "$out" > "$CTRL/next_jobid_$DEPTH.txt"
      if echo "$out" | grep -q "aurora-pbs"; then
        echo "successor queued at END (progress=$((EP_AFTER-EP_BEFORE)) depth=$DEPTH): $out"; ok=1; break
      fi
      echo "qsub attempt $attempt failed ($out) — retrying in 60s"; sleep 60
    done
    if [ "$ok" -eq 0 ]; then
      echo "SUCCESSOR QSUB FAILED after retries — chain will drop; monitor should catch it."
      touch "$CTRL/CHAIN_QSUB_FAILED"
    fi
  else
    echo "NO PROGRESS at depth $DEPTH >=2 — CRASH-LOOP GUARD, not resubmitting."
    touch "$CTRL/CHAIN_STALLED"
  fi
fi
echo "=== CHAIN LINK $DEPTH end $(date) ==="
"""


def emit_launch_block(ctrl, cpus_per_task):
    """Print a bash block launching all UNFINISHED cells on contiguous node blocks. Empty if all done."""
    cells = _cells_from_cfglist(ctrl)
    todo = []
    for cfg in cells:
        cur, tgt = _current_epoch(cfg), _target_epochs(cfg)
        if cur < tgt:
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
        ppn = min(tiles, 12)
        lo, hi = cursor + 1, cursor + nodes
        cursor += nodes
        port = 29500 + i
        slug = os.path.basename(base)
        env = "".join(f'  export {k}="{v}"\n' for k, v in (spec.get("env") or {}).items())
        lines += [
            f'NF{i}="$CTRL/nf_{i}"; sed -n "{lo},{hi}p" "$CTRL/nodefile.full" > "$NF{i}"',
            f'H{i}=$(head -1 "$NF{i}")',
            f'echo "[{slug}] nodes {lo}..{hi} head $H{i} port {port} {spec["dist_strategy"]}"',
            "(",
            f'  export PBS_NODEFILE="$NF{i}"',
            f'  export MASTER_ADDR="$H{i}"; export MASTER_PORT={port}; export WORLD_SIZE={tiles}',
            (env.rstrip("\n") if env else "  :"),
            f'  mpiexec --pmi=pmix -n {tiles} -ppn {ppn} --hostfile "$NF{i}" \\',
            f'      --cpu-bind depth --depth {cpus_per_task} \\',
            f'      python -m app.main_dist_aurora --train_mode \\',
            f'          --fname {os.path.abspath(cfg)} --params_path {os.path.abspath(cfg)} \\',
            f'      > "$CTRL/cell_{slug}.log" 2>&1',
            ") & PIDS+=($!)",
        ]
    lines += ['for p in "${PIDS[@]}"; do wait $p || true; done',
              'echo "link cells finished"']
    return "\n".join(lines)


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
        cur, tgt = _current_epoch(cfg), _target_epochs(cfg)
        mark = "DONE" if cur >= tgt else ""
        print(f"  {os.path.basename(cfg):30s} epoch {cur}/{tgt} {mark}")


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
    pr = sub.add_parser("_progress"); pr.add_argument("--ctrl", required=True)

    args = ap.parse_args()
    if not args.cmd:
        ap.error("a subcommand is required (start/status/stop/_emit/_progress)")
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
    elif args.cmd == "status":
        status(args.ctrl)
    elif args.cmd == "stop":
        open(os.path.join(args.ctrl, "STOP"), "w").close()
        print(f"touched STOP in {args.ctrl} — chain halts after current link")
    elif args.cmd == "_emit":
        print(emit_launch_block(args.ctrl, args.cpus))
    elif args.cmd == "_progress":
        # sum of current epochs across all cells (crash-loop / progress detector)
        print(sum(_current_epoch(cfg) for cfg in _cells_from_cfglist(args.ctrl)))


if __name__ == "__main__":
    main()
