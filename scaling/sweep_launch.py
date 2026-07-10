"""Topology-aware sweep launcher: run scaling cells on a held-node allocation, in place.

Each cell has a `<slug>_launch.json` (from gen_configs --use-topology) giving its node/tile/accum/
strategy/env. This builds a run.sh that:
  * assigns each cell a CONTIGUOUS BLOCK of nodes from the held nodefile (multi-node cells supported,
    unlike hold_run.py which is 1-node-per-cell),
  * writes a private per-cell hostfile (that block) so each MPI world sees world_size = tiles, not
    all-nodes (the aurora-multi-mpi-per-pbs-worldsize trap),
  * exports the cell's env dict (VJEPA_DIST_STRATEGY=hsdp, VJEPA_NUM_WORKERS=0, etc),
  * backgrounds all cells, waits, reports per-cell rc.

Because the trainer auto-resumes from latest.pth.tar in the run folder, re-triggering the SAME wave
after a 1h window resumes every cell where it left off — that's the overnight chain mechanism.

Usage (drops run.sh into the hold ctrl dir and touches trigger):
  python -m scaling.sweep_launch trigger --ctrl <dir> --configs 'configs/scaling/real/*.yaml' \
      --wave 1e18            # optional: only cells whose slug contains this substring
"""

import argparse
import glob
import json
import os


def _load_specs(configs):
    """For each config, load its _launch.json sidecar (fallback: single-node 12-tile ddp)."""
    specs = []
    for cfg in configs:
        base = os.path.splitext(cfg)[0]
        lj = base + "_launch.json"
        if os.path.exists(lj):
            spec = json.load(open(lj))
        else:
            spec = {"tiles": 12, "nodes": 1, "per_rank_bs": 8, "accum": 1,
                    "dist_strategy": "ddp", "env": {}}
        spec["_cfg"] = os.path.abspath(cfg)
        spec["_slug"] = os.path.basename(base)
        specs.append(spec)
    return specs


def _build_run_sh(specs, cpus_per_task, code_folder, ctrl_dir, port_base):
    lines = [
        "#!/bin/bash",
        "# sweep run.sh — one MPI world per cell, each on a contiguous node BLOCK of the held nodefile.",
        f"cd {code_folder}",
        f'CTRL="{ctrl_dir}"',
        'FULL="$CTRL/nodefile.full"',
        'echo "sweep starting; held nodes:"; cat "$FULL"',
        "declare -a PIDS=()",
    ]
    node_cursor = 0
    for i, s in enumerate(specs):
        nodes = int(s["nodes"])
        tiles = int(s["tiles"])
        ppn = min(tiles, 12)
        lo = node_cursor + 1               # sed is 1-indexed
        hi = node_cursor + nodes
        node_cursor += nodes
        port = port_base + i
        slug = s["_slug"]
        env_exports = "".join(f'  export {k}="{v}"\n' for k, v in (s.get("env") or {}).items())
        lines += [
            f'# ---- cell {i}: {slug}  (nodes {lo}..{hi}, {tiles} tiles, {s["dist_strategy"]}) ----',
            f'NF{i}="$CTRL/nf_{i}"; sed -n "{lo},{hi}p" "$FULL" > "$NF{i}"',
            f'H{i}=$(head -1 "$NF{i}")',
            f'echo "[cell {i}] {slug} -> nodes {lo}..{hi} head $H{i} port {port} ({s["dist_strategy"]})"',
            "(",
            f'  export PBS_NODEFILE="$NF{i}"',
            f'  export MASTER_ADDR="$H{i}"; export MASTER_PORT={port}; export WORLD_SIZE={tiles}',
            env_exports.rstrip("\n") if env_exports else "  :",
            f'  mpiexec --pmi=pmix -n {tiles} -ppn {ppn} \\',
            f'      --hostfile "$NF{i}" --cpu-bind depth --depth {cpus_per_task} \\',
            f'      python -m app.main_dist_aurora --train_mode \\',
            f'          --fname {s["_cfg"]} --params_path {s["_cfg"]} \\',
            f'      > "$CTRL/cell_{i}_{slug}.log" 2>&1',
            f') & PIDS+=($!)',
        ]
    lines += [
        f'echo "launched {len(specs)} cells on {node_cursor} nodes; waiting"',
        "rc=0",
        'for p in "${PIDS[@]}"; do wait $p || rc=1; done',
        'echo "sweep wave done aggregate_rc=$rc"',
        'for f in "$CTRL"/cell_*.log; do echo "===== $f (tail) ====="; tail -6 "$f"; done',
        "exit $rc",
    ]
    return "\n".join(lines) + "\n"


def _next_run_index(ctrl):
    done = [f for f in os.listdir(ctrl) if f.startswith("done_")]
    return max([int(f[5:]) for f in done], default=0) + 1


def trigger(ctrl, configs, cpus_per_task, code_folder, port_base):
    nf = os.path.join(ctrl, "nodefile.full")
    if not os.path.exists(nf):
        raise SystemExit(f"{nf} absent — is the hold job RUNNING?")
    specs = _load_specs(configs)
    need = sum(int(s["nodes"]) for s in specs)
    have = sum(1 for _ in open(nf))
    if need > have:
        raise SystemExit(f"cells need {need} nodes but only {have} held. Split into smaller waves.")
    run_sh = _build_run_sh(specs, cpus_per_task, code_folder, ctrl, port_base)
    with open(os.path.join(ctrl, "run.sh"), "w") as f:
        f.write(run_sh)
    idx = _next_run_index(ctrl)
    open(os.path.join(ctrl, "trigger"), "w").close()
    print(f"triggered wave {idx}: {len(specs)} cells on {need}/{have} nodes; ctrl={ctrl}")
    for s in specs:
        print(f"  {s['_slug']}: {s['nodes']}n {s['tiles']}t {s['dist_strategy']} accum={s.get('accum',1)}")
    return idx


def main():
    ap = argparse.ArgumentParser(description="Topology-aware scaling sweep launcher (hold-node)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trigger")
    t.add_argument("--ctrl", required=True)
    t.add_argument("--configs", required=True, help="glob of per-cell YAMLs (with _launch.json)")
    t.add_argument("--wave", default=None, help="substring filter on slug (e.g. C1e+18)")
    t.add_argument("--cpus-per-task", type=int, default=16)
    t.add_argument("--code-folder", default=os.getcwd())
    t.add_argument("--port-base", type=int, default=29500)
    args = ap.parse_args()

    configs = sorted(glob.glob(args.configs))
    if args.wave:
        configs = [c for c in configs if args.wave in os.path.basename(c)]
    if not configs:
        raise SystemExit(f"no configs match {args.configs} (wave={args.wave})")
    trigger(args.ctrl, configs, args.cpus_per_task, os.path.abspath(args.code_folder), args.port_base)


if __name__ == "__main__":
    main()
