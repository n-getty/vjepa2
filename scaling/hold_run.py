"""Drive a held-node job (scaling/hold_node_job.py): run a fan-out of cells on the held nodes, in
place, without requeuing. Write run.sh into the ctrl dir, touch trigger, wait for done_<n>.

This reuses the SAME per-cell launch logic as scaling/fanout_pbs.py (private single-host nodefiles,
one MPI world per cell) but targets the already-held nodefile (ctrl/nodefile.full) instead of a fresh
PBS allocation. So when a cell hits a bug: fix code, re-run `trigger` — the allocation is never lost.

Usage:
  # after `hold_node_job.py submit` and the job is RUNNING:
  python -m scaling.hold_run trigger --ctrl <dir> --configs 'configs/scaling/pilot6/*.yaml' \
      --tiles-per-node 12
  python -m scaling.hold_run wait --ctrl <dir>          # block until the run finishes, print tails
"""

import argparse
import glob
import os
import time


def _build_run_sh(configs, tiles_per_node, cpus_per_task, code_folder, ctrl_dir, port_base):
    lines = [
        "#!/bin/bash",
        "# run.sh — executed by the hold loop on the head node. Launches one MPI world per cell,",
        "# each pinned to its own node via a private single-host nodefile (avoids world_size=N*tiles).",
        f"cd {code_folder}",
        f'CTRL="{ctrl_dir}"',
        'FULL="$CTRL/nodefile.full"',
        'echo "cells starting on nodes:"; cat "$FULL"',
    ]
    for i, cfg in enumerate(configs):
        slug = os.path.splitext(os.path.basename(cfg))[0]
        port = port_base + i
        lines += [
            f'# ---- cell {i}: {slug} ----',
            f'NF{i}="$CTRL/nf_{i}"; sed -n "{i+1}p" "$FULL" > "$NF{i}"',
            f'H{i}=$(cat "$NF{i}")',
            f'echo "[cell {i}] {slug} -> $H{i} port {port}"',
            '(',
            f'  export PBS_NODEFILE="$NF{i}"',
            f'  export MASTER_ADDR="$H{i}"; export MASTER_PORT={port}; export WORLD_SIZE={tiles_per_node}',
            f'  mpiexec --pmi=pmix -n {tiles_per_node} -ppn {tiles_per_node} \\',
            f'      --hostfile "$NF{i}" --cpu-bind depth --depth {cpus_per_task} \\',
            f'      python -m app.main_dist_aurora --train_mode \\',
            f'          --fname {os.path.abspath(cfg)} --params_path {os.path.abspath(cfg)} \\',
            f'      > "$CTRL/cell_{i}_{slug}.log" 2>&1',
            f') & PID{i}=$!',
        ]
    pids = " ".join(f"$PID{i}" for i in range(len(configs)))
    lines += [
        f'echo "launched {len(configs)} cells; waiting"',
        "rc=0",
        f'for p in {pids}; do wait $p || rc=1; done',
        'echo "all cells done aggregate_rc=$rc"',
        'for f in "$CTRL"/cell_*.log; do echo "===== $f (tail) ====="; tail -4 "$f"; done',
        "exit $rc",
    ]
    return "\n".join(lines) + "\n"


def _next_run_index(ctrl):
    done = [f for f in os.listdir(ctrl) if f.startswith("done_")]
    return max([int(f[5:]) for f in done], default=0) + 1


def trigger(ctrl, configs, tiles_per_node, cpus_per_task, code_folder, port_base):
    if not os.path.exists(os.path.join(ctrl, "nodefile.full")):
        raise SystemExit(f"{ctrl}/nodefile.full absent — is the hold job RUNNING? (check status)")
    run_sh = _build_run_sh(configs, tiles_per_node, cpus_per_task, code_folder, ctrl, port_base)
    with open(os.path.join(ctrl, "run.sh"), "w") as f:
        f.write(run_sh)
    idx = _next_run_index(ctrl)
    open(os.path.join(ctrl, "trigger"), "w").close()
    print(f"triggered run {idx} with {len(configs)} cells; ctrl={ctrl}")
    print(f"  cells: {', '.join(os.path.basename(c) for c in configs)}")
    return idx


def wait(ctrl, timeout_s=3600):
    """Block until the current run produces a done_<n>, then print its tail."""
    start_done = set(f for f in os.listdir(ctrl) if f.startswith("done_"))
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        now = set(f for f in os.listdir(ctrl) if f.startswith("done_"))
        new = now - start_done
        if new:
            d = sorted(new, key=lambda x: int(x[5:]))[-1]
            rc = open(os.path.join(ctrl, d)).read().strip()
            n = d[5:]
            log = os.path.join(ctrl, f"run_{n}.log")
            print(f"run {n} finished rc={rc}")
            if os.path.exists(log):
                print("\n".join(open(log).read().splitlines()[-25:]))
            return rc
        time.sleep(10)
    print(f"wait timed out after {timeout_s}s (run still going? check status)")
    return None


def main():
    ap = argparse.ArgumentParser(description="Drive a held-node job to run cells in place")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trigger")
    t.add_argument("--ctrl", required=True)
    t.add_argument("--configs", required=True, help="glob of per-cell YAMLs")
    t.add_argument("--tiles-per-node", type=int, default=12)
    t.add_argument("--cpus-per-task", type=int, default=16)
    t.add_argument("--code-folder", default=os.getcwd())
    t.add_argument("--port-base", type=int, default=29500)

    w = sub.add_parser("wait")
    w.add_argument("--ctrl", required=True)
    w.add_argument("--timeout", type=int, default=3600)

    args = ap.parse_args()
    if args.cmd == "trigger":
        configs = sorted(glob.glob(args.configs))
        if not configs:
            raise SystemExit(f"no configs match {args.configs}")
        trigger(args.ctrl, configs, args.tiles_per_node, args.cpus_per_task,
                os.path.abspath(args.code_folder), args.port_base)
    else:
        wait(args.ctrl, args.timeout)


if __name__ == "__main__":
    main()
