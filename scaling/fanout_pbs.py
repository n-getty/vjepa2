"""Generate ONE multi-node PBS job that runs many scaling cells CONCURRENTLY, one node per cell.

Why: Aurora's debug and debug-scaling queues both cap `max_run = 1 job/user`, so we cannot submit N
separate jobs. Instead we request `select=N` nodes in a single job and launch N MPI worlds in parallel
— each cell pinned to its OWN node via a PRIVATE single-host nodefile.

The private-nodefile part is load-bearing (memory: aurora-multi-mpi-per-pbs-worldsize): if each mpiexec
saw the full PBS_NODEFILE, oneCCL would compute world_size = local_ranks * ALL_nodes and the per-world
rendezvous would hang (rc=143). Splitting PBS_NODEFILE into one-host files and pointing each world's
mpiexec at its own file keeps every world at world_size = tiles_per_node (12).

Reuses the exact env block from app/main_dist_aurora.py (module load, ZE_FLAT, CCL_*, proxy) so the
distributed setup is identical to the validated single-cell path — we only change the launch topology.

Usage:
    python -m scaling.fanout_pbs --configs 'configs/scaling/pilot/*.yaml' \
        --account AuroraGPT --partition debug-scaling --time 60 \
        --tiles-per-node 12 --out scaling/pilot_fanout.pbs
    qsub scaling/pilot_fanout.pbs
"""

import argparse
import glob
import os

import yaml

# The canonical Aurora env block (kept in sync with app/main_dist_aurora.py:build_pbs_script).
AURORA_ENV = [
    "ulimit -c unlimited",
    "module load frameworks",
    "export ZE_FLAT_DEVICE_HIERARCHY=FLAT",
    "export MPICH_GPU_SUPPORT_ENABLED=1",
    "export CCL_PROCESS_LAUNCHER=pmix",
    "export CCL_ATL_TRANSPORT=mpi",
    "export CCL_KVS_MODE=mpi",
    "export CCL_KVS_USE_MPI_RANKS=1",
    "export CCL_CONFIGURATION=cpu_gpu_dpcpp",
    "export CCL_KVS_CONNECTION_TIMEOUT=600",
    "export CCL_OP_SYNC=1",
    "export CCL_WORKER_COUNT=1",
    "export CCL_ALLREDUCE=ring",
    "export CCL_CHUNK_SIZE=16777216",
    "export FI_PROVIDER=cxi",
    "export PYTHONFAULTHANDLER=1",
    "export TMPDIR=/tmp",
    'export http_proxy="http://proxy.alcf.anl.gov:3128"',
    'export https_proxy="http://proxy.alcf.anl.gov:3128"',
    'export ftp_proxy="http://proxy.alcf.anl.gov:3128"',
]


def build(configs, account, partition, walltime_min, tiles_per_node, cpus_per_task,
          master_port_base, code_folder):
    n = len(configs)
    hours, mins = divmod(int(walltime_min), 60)
    walltime = f"{hours:02d}:{mins:02d}:00"

    # Each cell writes params to its own folder; the trainer reads --fname. We point each world at its
    # config directly (already fully-specified per-cell YAMLs from gen_configs).
    env_block = "\n".join(AURORA_ENV)

    # Per-cell launch stanzas, each backgrounded, each with a private 1-host nodefile.
    launches = []
    for i, cfg in enumerate(configs):
        slug = os.path.splitext(os.path.basename(cfg))[0]
        port = master_port_base + i
        launches.append(f"""
# ---- cell {i}: {slug} ----
NODEFILE_{i}="$JOBTMP/nodefile_{i}"
sed -n '{i+1}p' "$PBS_NODEFILE" > "$NODEFILE_{i}"
HOST_{i}=$(cat "$NODEFILE_{i}")
echo "[cell {i}] {slug} -> node $HOST_{i} port {port}"
(
  export PBS_NODEFILE="$NODEFILE_{i}"
  export MASTER_ADDR="$HOST_{i}"
  export MASTER_PORT={port}
  export WORLD_SIZE={tiles_per_node}
  mpiexec --pmi=pmix -n {tiles_per_node} -ppn {tiles_per_node} \\
      --hostfile "$NODEFILE_{i}" --cpu-bind depth --depth {cpus_per_task} \\
      python -m app.main_dist_aurora --train_mode \\
          --fname {os.path.abspath(cfg)} --params_path {os.path.abspath(cfg)} \\
      > "$JOBTMP/cell_{i}_{slug}.log" 2>&1
) &
PID_{i}=$!
""")

    launch_block = "\n".join(launches)
    wait_pids = " ".join(f"$PID_{i}" for i in range(n))

    script = f"""#!/bin/bash -l
#PBS -N scaling_fanout_{n}
#PBS -l select={n}
#PBS -l walltime={walltime}
#PBS -l filesystems=home:flare
#PBS -q {partition}
#PBS -A {account}

set -o pipefail
# NOTE: deliberately NOT `set -u` — lmod's init references unbound vars (ZSH_EVAL_CONTEXT) and
# `module load` aborts under `set -u` (the python-env-shadowing-hpc trap). The validated single-cell
# launcher uses `set -eo pipefail` for the same reason. We also avoid `set -e` so one cell's failure
# doesn't kill the whole job before `wait` collects the others.

echo "FANOUT JOB START: $(date)  cells={n}  nodes={n}"
cat "$PBS_NODEFILE"
cd {code_folder}

{env_block}

# Per-job scratch for private nodefiles + per-cell logs (survives on flare via folder copy at end).
JOBTMP="{code_folder}/../_fanout_${{PBS_JOBID%%.*}}"
mkdir -p "$JOBTMP"
echo "JOBTMP=$JOBTMP"

{launch_block}

echo "launched {n} cells; waiting..."
rc=0
for pid in {wait_pids}; do
  wait $pid || rc=1
done
echo "FANOUT JOB END: $(date)  aggregate_rc=$rc"
# Surface per-cell tails for quick triage in the PBS stdout.
for f in "$JOBTMP"/cell_*.log; do echo "===== $f (tail) ====="; tail -5 "$f"; done
exit $rc
"""
    return script


def main():
    ap = argparse.ArgumentParser(description="Build a multi-node fan-out PBS for the scaling sweep")
    ap.add_argument("--configs", required=True, help="glob of per-cell YAMLs")
    ap.add_argument("--account", default="AuroraGPT")
    ap.add_argument("--partition", default="debug-scaling")
    ap.add_argument("--time", type=int, default=60, help="walltime minutes (must fit the SLOWEST cell)")
    ap.add_argument("--tiles-per-node", type=int, default=12)
    ap.add_argument("--cpus-per-task", type=int, default=16)
    ap.add_argument("--master-port-base", type=int, default=29500)
    ap.add_argument("--code-folder", default=os.getcwd(),
                    help="repo root the compute nodes cd into (default: cwd)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    configs = sorted(glob.glob(args.configs))
    if not configs:
        raise SystemExit(f"no configs match {args.configs}")
    # sanity: every cell must be a valid YAML with a folder + a model spec. Training configs carry
    # `model:`; eval configs (eval_name + model_kwargs, e.g. the SSv2 Metric-A probes) carry
    # `model_kwargs:` instead. Accept either so this fan-out drives both training and eval cells.
    for c in configs:
        y = yaml.safe_load(open(c))
        assert ("model" in y or "model_kwargs" in y) and "folder" in y, f"{c} missing model/folder"

    script = build(configs, args.account, args.partition, args.time,
                   args.tiles_per_node, args.cpus_per_task, args.master_port_base,
                   os.path.abspath(args.code_folder))
    with open(args.out, "w") as f:
        f.write(script)
    print(f"wrote {args.out}: {len(configs)} cells on {len(configs)} nodes ({args.partition})")
    for i, c in enumerate(configs):
        print(f"  node {i}: {os.path.basename(c)}")
    print(f"\nsubmit with:  qsub {args.out}")


if __name__ == "__main__":
    main()
