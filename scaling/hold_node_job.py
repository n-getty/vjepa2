"""Hold-node + file-triggered control loop, so we iterate on bugs IN PLACE without requeuing.

Aurora queue slots are hard-won (reservations + 1-job/user caps). Losing an allocation to a bug and
going back to the queue is the expensive failure mode. This submits ONE batch job that grabs N nodes
and then just *waits*, running whatever command we drop into a control dir on flare. From the login
node we write `run.sh`, `touch trigger`, and the held job executes it on its nodes; if it crashes we
fix and re-trigger — the allocation never releases until walltime or an explicit STOP.

Layout (all on flare so login+compute both see it):
  <ctrl>/nodefile.full   full PBS_NODEFILE (the held nodes)
  <ctrl>/run.sh          the command to run on the head node (we write this)
  <ctrl>/trigger         touch to request a run; the loop deletes it and runs run.sh
  <ctrl>/run_<n>.log     stdout/stderr of run n
  <ctrl>/running         present while a run is in flight
  <ctrl>/done_<n>        touched when run n finishes (contains its exit code)
  <ctrl>/STOP            touch to end the hold loop (releases the allocation)
  <ctrl>/status          heartbeat: HELD <ts> or RUNNING <ts>

Subcommands:
  submit  — generate + qsub the hold job. Prints the ctrl dir.
  status  — print heartbeat + last run tail.
  stop    — touch STOP (release the allocation).
The actual "run a fanout on the held nodes" logic is driven by scaling/hold_run.py (writes run.sh +
triggers), so the hold job itself is generic.
"""

import argparse
import os
import subprocess

# Canonical Aurora env (kept in sync with app/main_dist_aurora.py). NOTE: no `set -u` (lmod trap).
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


def build_script(nodes, walltime_min, account, partition, code_folder, ctrl_dir):
    hours, mins = divmod(int(walltime_min), 60)
    walltime = f"{hours:02d}:{mins:02d}:00"
    return f"""#!/bin/bash -l
#PBS -N hold_{nodes}n
#PBS -l select={nodes}
#PBS -l walltime={walltime}
#PBS -l filesystems=home:flare
#PBS -q {partition}
#PBS -A {account}

set -o pipefail   # NOT set -u (lmod ZSH_EVAL_CONTEXT trap); NOT set -e (a failed run must not kill hold)

echo "HOLD JOB START: $(date)  nodes={nodes}"
cd {code_folder}
{AURORA_ENV}

CTRL="{ctrl_dir}"
mkdir -p "$CTRL"
cp "$PBS_NODEFILE" "$CTRL/nodefile.full"
n=0
echo "HELD $(date +%s)" > "$CTRL/status"
echo "hold loop ready; ctrl=$CTRL  nodes:"; cat "$CTRL/nodefile.full"

# Keep the allocation alive; run run.sh on each trigger. Ends on STOP or walltime.
while [ ! -f "$CTRL/STOP" ]; do
  if [ -f "$CTRL/trigger" ]; then
    rm -f "$CTRL/trigger"
    n=$((n+1))
    touch "$CTRL/running"
    echo "RUNNING $(date +%s) run=$n" > "$CTRL/status"
    echo "=== run $n start $(date) ===" > "$CTRL/run_$n.log"
    bash "$CTRL/run.sh" >> "$CTRL/run_$n.log" 2>&1
    rc=$?
    echo "$rc" > "$CTRL/done_$n"
    rm -f "$CTRL/running"
    echo "HELD $(date +%s) last_run=$n last_rc=$rc" > "$CTRL/status"
    echo "=== run $n end rc=$rc $(date) ===" >> "$CTRL/run_$n.log"
  fi
  sleep 5
done
echo "HOLD JOB END (STOP seen): $(date)"
"""


def main():
    ap = argparse.ArgumentParser(description="Submit a hold-node control-loop job")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--nodes", type=int, required=True)
    s.add_argument("--time", type=int, default=60, help="walltime minutes (debug-scaling max 60)")
    s.add_argument("--account", default="AuroraGPT")
    s.add_argument("--partition", default="debug-scaling")
    s.add_argument("--code-folder", default=os.getcwd())
    s.add_argument("--ctrl-dir", default="/flare/ModCon/ngetty/experiments/_hold_ctrl")
    s.add_argument("--dry-run", action="store_true")

    st = sub.add_parser("status")
    st.add_argument("--ctrl-dir", default="/flare/ModCon/ngetty/experiments/_hold_ctrl")

    sp = sub.add_parser("stop")
    sp.add_argument("--ctrl-dir", default="/flare/ModCon/ngetty/experiments/_hold_ctrl")

    args = ap.parse_args()

    if args.cmd == "submit":
        os.makedirs(args.ctrl_dir, exist_ok=True)
        # clear stale control files from a prior hold
        for f in ("STOP", "trigger", "running", "status", "run.sh"):
            p = os.path.join(args.ctrl_dir, f)
            if os.path.exists(p):
                os.remove(p)
        script = build_script(args.nodes, args.time, args.account, args.partition,
                              os.path.abspath(args.code_folder), os.path.abspath(args.ctrl_dir))
        pbs_path = os.path.join(args.ctrl_dir, "_hold.pbs")
        with open(pbs_path, "w") as f:
            f.write(script)
        print(f"wrote {pbs_path}  (ctrl={args.ctrl_dir})")
        if args.dry_run:
            print("--dry-run: not submitting")
            return
        out = subprocess.run(["qsub", pbs_path], capture_output=True, text=True)
        if out.returncode != 0:
            print("qsub FAILED:", out.stderr.strip())
            raise SystemExit(1)
        print("qsub:", out.stdout.strip())
        print(f"ctrl dir: {args.ctrl_dir}")

    elif args.cmd == "status":
        c = args.ctrl_dir
        for f in ("status", "nodefile.full"):
            p = os.path.join(c, f)
            print(f"--- {f} ---")
            if os.path.exists(p):
                print(open(p).read().strip())
            else:
                print("(absent)")
        # last run log tail
        logs = sorted([x for x in os.listdir(c) if x.startswith("run_") and x.endswith(".log")],
                      key=lambda x: int(x[4:-4])) if os.path.isdir(c) else []
        if logs:
            last = os.path.join(c, logs[-1])
            print(f"--- {logs[-1]} (tail) ---")
            print("\n".join(open(last).read().splitlines()[-15:]))

    elif args.cmd == "stop":
        open(os.path.join(args.ctrl_dir, "STOP"), "w").close()
        print(f"touched STOP in {args.ctrl_dir} — hold loop will exit and release nodes")


if __name__ == "__main__":
    main()
