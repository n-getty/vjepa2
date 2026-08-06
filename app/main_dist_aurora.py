# Aurora (PBS + PALS/mpiexec, Intel XPU + oneCCL) multi-node launcher for V-JEPA 2.1.
#
# Mirrors the structure of app/main_dist_polaris.py:
#   login-node (default): read YAML, materialize <folder>/params-pretrain.yaml,
#                         generate <folder>/_submit.pbs, qsub it.
#   compute-node (--train_mode): pin XPU tile from PALS local rank,
#                                let init_distributed() pick up PMI/PALS env,
#                                hand off to app.scaffold.main.
#
# The Aurora-specific CCL env block (CCL_PROCESS_LAUNCHER=pmix, CCL_ATL_TRANSPORT=mpi,
# CCL_WORKER_COUNT=1, FI_PROVIDER=cxi, ...) follows the "Production multi-node"
# Aurora oneCCL recommendations from the ALCF user-guides.

import argparse
import os
import sys


def _configure_threading_env():
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(key, "1")


_configure_threading_env()


# Defensive: pin per-rank XPU visibility *before* any torch import. Without
# ZE_AFFINITY_MASK pinning, level-zero opens 144 contexts (12 ranks * 12 tiles)
# on a node which trips the L0 device-context limit (BaseMM_PRISM gotcha).
if "--train_mode" in sys.argv:
    for var in ("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "MPI_LOCALRANKID",
                "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"):
        if var in os.environ:
            os.environ["ZE_AFFINITY_MASK"] = os.environ[var]
            break
    # Aurora workaround: torch multiprocessing's file_descriptor sharing
    # exhausts file handles under XPU; the file_system strategy stays bounded.
    os.environ.setdefault("MP_SOCKET_DIR", "/tmp")

import copy
import datetime
import pprint
import shutil
import subprocess
import traceback
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from evals.scaffold import main as eval_main
from src.utils.logging import get_logger, git_information

logger = get_logger(force=True)


parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, default="configs.yaml",
                    help="YAML config file to launch.")
parser.add_argument("--folder", type=str, default=None,
                    help="If set, overrides the 'folder' field in the YAML.")
parser.add_argument("--account", type=str, default=os.environ.get("VJEPA_ACCOUNT"),
                    help="PBS account. Defaults to $VJEPA_ACCOUNT. Required.")
parser.add_argument("--partition", type=str, default="debug",
                    help="PBS queue (Aurora: debug ≤2n×1h, debug-scaling 2–256n×1h, "
                         "prod routes to small/medium/large).")
parser.add_argument("--time", type=int, default=60,
                    help="Wall time in minutes.")
parser.add_argument("--filesystems", type=str, default="home:flare",
                    help="PBS -l filesystems= value.")
parser.add_argument("--nodes", type=int, default=None,
                    help="Override 'nodes' field in YAML.")
parser.add_argument("--tasks_per_node", type=int, default=None,
                    help="Override 'tasks_per_node' field (tiles per node; 12 on Aurora).")
parser.add_argument("--cpus_per_task", type=int, default=8,
                    help="mpiexec --depth value. 8 leaves headroom for OMP+dataloader.")
parser.add_argument("--master_port", type=int, default=29500)
parser.add_argument("--stage_data", action="store_true",
                    help="Rsync data.datasets entries to /tmp/vjepa_data/<job> on each node "
                         "before training. Only correct if entries are directories "
                         "(WebDataset shards), not CSV files.")
parser.add_argument("--no_copy_code", action="store_true",
                    help="Skip copying the code tree into <folder>/code (use repo as-is).")
parser.add_argument("--extra_env", action="append", default=[],
                    help="Extra `KEY=VAL` env vars to export in the PBS script (repeatable).")
parser.add_argument("--venv", type=str,
                    default=os.environ.get("PYTHON_BIN_VENV"),
                    help="Optional venv to activate after `module load frameworks`. "
                         "On Aurora the recommended path is no venv — install missing "
                         "packages to --user so the IPEX/oneCCL stack stays consistent.")
parser.add_argument("--depends_on", type=str, default=None,
                    help="PBS job id this job should run after (afterok dependency).")
parser.add_argument("--dry_run", action="store_true",
                    help="Build PBS script and dump it, but don't qsub.")

# compute-node-only flags
parser.add_argument("--train_mode", action="store_true",
                    help="Internal: signals execution on a compute node.")
parser.add_argument("--local_data_root", type=str, default=None,
                    help="Internal: /tmp/vjepa_data/<job> root to remap data.datasets onto.")
parser.add_argument("--params_path", type=str, default=None,
                    help="Internal: path to params-pretrain.yaml dumped by the login node. "
                         "Overrides --fname on the compute node so all ranks read the same file.")


# ----------------------------------------------------------------------------
# Login-node: build PBS script and submit
# ----------------------------------------------------------------------------

def copy_code_folder(code_folder):
    ignore_patterns = ["__pycache__", ".vscode", ".git", "core", ".runtime_configs",
                       "checkpoints", "head_phases"]
    if os.path.exists(code_folder):
        logger.info(f"Removing existing code folder: {code_folder}")
        shutil.rmtree(code_folder)
    shutil.copytree(".", code_folder, ignore=shutil.ignore_patterns(*ignore_patterns))


def maybe_timestamp_folder(params):
    folder = params["folder"]
    load_checkpoint = params.get("meta", {}).get("load_checkpoint", False)
    if not load_checkpoint and Path(folder).exists() and any(Path(folder).iterdir()):
        ts = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        new_folder = folder.rstrip("/") + f"_{ts}"
        logger.info(f"Folder {folder} exists; using {new_folder}")
        params["folder"] = new_folder
    return params


def build_pbs_script(folder, job_name, account, partition, walltime_minutes,
                     nodes, tasks_per_node, cpus_per_task, filesystems, master_port,
                     params_path, code_folder, stage_data, dataset_paths, extra_env,
                     venv):
    nnodes = int(nodes)
    nranks = int(tasks_per_node)
    ndepth = int(cpus_per_task)
    ntotranks = nnodes * nranks

    hours, mins = divmod(int(walltime_minutes), 60)
    walltime = f"{hours:02d}:{mins:02d}:00"

    aurora_env = [
        "ulimit -c unlimited",
        # Default frameworks (2025.3.1) is the only env where mpi4py + mpich
        # libs are co-loaded — required for the pre-`init_process_group` MPI
        # barrier (torchtune CLAUDE.md production multi-node row).
        # webdataset+decord+braceexpand are user-installed for python3.12.
        "module load frameworks",
        f"source {venv}/bin/activate" if venv else "",
        # Tile addressing: split each GPU into 2 tiles → 12 tiles/node.
        "export ZE_FLAT_DEVICE_HIERARCHY=FLAT",
        "export MPICH_GPU_SUPPORT_ENABLED=1",
        # oneCCL multi-node block (torchtune CLAUDE.md "Production multi-node" row).
        # Tested 2026-06-05: CCL_ALLREDUCE=topo + rabenseifner from ALCF
        # docs was ~4% SLOWER than ring at 16n / 192 ranks for our DDP workload
        # (rabenseifner's intra-node fanout overhead exceeds inter-node savings
        # at only 16 nodes — ALCF's "large scale" recs target 64+ nodes).
        "export CCL_PROCESS_LAUNCHER=pmix",
        "export CCL_ATL_TRANSPORT=mpi",
        "export CCL_KVS_MODE=mpi",
        "export CCL_KVS_USE_MPI_RANKS=1",
        "export CCL_CONFIGURATION=cpu_gpu_dpcpp",
        "export CCL_KVS_CONNECTION_TIMEOUT=600",
        "export CCL_OP_SYNC=1",
        # CCL_WORKER_COUNT=4 caused a 48x AllGather regression in torchtune.
        "export CCL_WORKER_COUNT=1",
        "export CCL_ALLREDUCE=ring",
        "export CCL_CHUNK_SIZE=16777216",
        # Slingshot 11 fabric.
        "export FI_PROVIDER=cxi",
        "export PYTHONFAULTHANDLER=1",
        "export TMPDIR=/tmp",
        f"export OMP_NUM_THREADS={ndepth}",
        'export http_proxy="http://proxy.alcf.anl.gov:3128"',
        'export https_proxy="http://proxy.alcf.anl.gov:3128"',
        'export ftp_proxy="http://proxy.alcf.anl.gov:3128"',
    ]
    aurora_env = [line for line in aurora_env if line]
    for kv in extra_env:
        aurora_env.append(f"export {kv}")

    master = [
        'if [[ -f "${PBS_NODEFILE:-}" ]]; then',
        '  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")',
        'else',
        '  MASTER_ADDR=$(hostname)',
        'fi',
        "export MASTER_ADDR",
        f"export MASTER_PORT={master_port}",
        f"export WORLD_SIZE={ntotranks}",
    ]

    staging_block = ""
    local_data_flag = ""
    if stage_data and dataset_paths:
        stage_script_path = os.path.join(folder, "_stage_data.sh")
        stage_script = ["#!/bin/bash", "set -eo pipefail",
                        'echo "[$HOSTNAME] staging started: $(date)"',
                        f'export LOCAL_DATA_DIR="/tmp/vjepa_data/{job_name}_${{PBS_JOBID}}"',
                        "mkdir -p $LOCAL_DATA_DIR"]
        for src in dataset_paths:
            name = os.path.basename(src.rstrip("/"))
            stage_script.append(
                f'rsync -a --info=progress2 --ignore-existing {src}/ $LOCAL_DATA_DIR/{name}/ || true'
            )
        stage_script.append('echo "[$HOSTNAME] staging complete: $(date)"')
        with open(stage_script_path, "w") as f:
            f.write("\n".join(stage_script))
        os.chmod(stage_script_path, 0o755)

        staging_block = "\n".join([
            'echo "--- starting parallel data staging ---"',
            # One process per node, no cpu-bind so rsync has free run of all cores.
            f'mpiexec -n {nnodes} -ppn 1 --cpu-bind none {stage_script_path}',
            f'export LOCAL_DATA_DIR="/tmp/vjepa_data/{job_name}_${{PBS_JOBID}}"',
            'echo "--- staging complete ---"',
        ])
        local_data_flag = "--local_data_root $LOCAL_DATA_DIR"

    script = f"""#!/bin/bash -l
#PBS -N {job_name}
#PBS -l select={nnodes}
#PBS -l walltime={walltime}
#PBS -l filesystems={filesystems}
#PBS -q {partition}
#PBS -A {account}

set -eo pipefail

echo "JOB START: $(date)"
echo "Nodes: {nnodes} Total ranks: {ntotranks} ppn: {nranks} cpus/task: {ndepth}"

cd {code_folder}

{chr(10).join(aurora_env)}

{chr(10).join(master)}

{staging_block}

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

mpiexec --pmi=pmix -n {ntotranks} -ppn {nranks} \\
    --cpu-bind depth --depth {ndepth} \\
    python -m app.main_dist_aurora --train_mode \\
        --fname {params_path} --params_path {params_path} {local_data_flag}

echo "JOB END: $(date)"
"""
    return script


def submit(args):
    with open(args.fname, "r") as f:
        params = yaml.safe_load(f)

    if args.folder:
        params["folder"] = args.folder
    if args.nodes:
        params["nodes"] = args.nodes
    if args.tasks_per_node:
        params["tasks_per_node"] = args.tasks_per_node

    params = maybe_timestamp_folder(params)

    folder = params["folder"]
    Path(folder).mkdir(parents=True, exist_ok=True)
    job_name = Path(folder.rstrip("/")).name

    code_folder = os.path.join(folder, "code")
    if args.no_copy_code:
        code_folder = os.getcwd()
    else:
        copy_code_folder(code_folder)

    params_path = os.path.join(folder, "params-pretrain.yaml")
    with open(params_path, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    with open(os.path.join(folder, "git-info.txt"), "w") as f:
        f.write(git_information())

    dataset_paths = params.get("data", {}).get("datasets", []) or []

    nodes = int(params.get("nodes", 1))
    tasks_per_node = int(params.get("tasks_per_node", 12))
    cpus_per_task = int(params.get("cpus_per_task", args.cpus_per_task))

    pbs = build_pbs_script(
        folder=folder, job_name=job_name, account=args.account,
        partition=args.partition, walltime_minutes=args.time,
        nodes=nodes, tasks_per_node=tasks_per_node, cpus_per_task=cpus_per_task,
        filesystems=args.filesystems, master_port=args.master_port,
        params_path=params_path, code_folder=code_folder,
        stage_data=args.stage_data, dataset_paths=dataset_paths,
        extra_env=args.extra_env, venv=args.venv,
    )

    pbs_path = os.path.join(folder, "_submit.pbs")
    with open(pbs_path, "w") as f:
        f.write(pbs)
    logger.info(f"PBS script: {pbs_path}")

    if args.dry_run:
        logger.info("dry run — not submitting")
        print(pbs_path)
        return

    qsub_cmd = ["qsub"]
    if args.depends_on:
        qsub_cmd += ["-W", f"depend=afterok:{args.depends_on}"]
    qsub_cmd.append(pbs_path)

    try:
        result = subprocess.run(qsub_cmd, capture_output=True, text=True,
                                check=True, cwd=folder)
        job_id = result.stdout.strip()
        logger.info(f"qsub: {job_id}")
        print(job_id)
    except FileNotFoundError:
        logger.error("qsub not found. Run from an Aurora login node.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"qsub failed: {e.stderr}")
        raise


# ----------------------------------------------------------------------------
# Compute-node: pin XPU tile, init dist, hand off to scaffold.main
# ----------------------------------------------------------------------------

def run_training(args):
    from src.utils.distributed import _get_pmi_env, init_distributed
    import torch

    # IPEX provides the XPU primitives + the oneccl_bindings_for_pytorch
    # import side effect required when init_distributed() falls back to ccl.
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except Exception as e:
        logger.warning(f"intel_extension_for_pytorch not importable: {e}")

    # ZE_AFFINITY_MASK was pinned at module load (before torch import) when
    # --train_mode was on sys.argv — see top-of-file note. Don't repin here.
    local_rank = int(_get_pmi_env("LOCAL_RANK") or "0")

    if not os.environ.get("MASTER_ADDR"):
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            with open(nodefile) as f:
                os.environ["MASTER_ADDR"] = f.readline().strip()
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    # torch.multiprocessing default sharing strategy exhausts FDs under XPU
    # dataloader workers; file_system is the documented Aurora fallback.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    cfg_path = args.params_path or args.fname
    with open(cfg_path, "r") as f:
        params = yaml.safe_load(f)
    params.setdefault("data", {})

    if args.local_data_root and "datasets" in params["data"]:
        new_paths = []
        for p in params["data"]["datasets"]:
            name = os.path.basename(p.rstrip("/"))
            new_paths.append(os.path.join(args.local_data_root, name))
        params["data"]["datasets"] = new_paths

    # Eval probes (CSV-of-clips VideoDataset) carry their data paths under
    # experiment.data.dataset_{train,val}. When clips were staged to node-local
    # /tmp, stage_probe_clips.py wrote prefix-swapped CSVs under
    # <local_data_root>/csv/; repoint the config at those so VideoDataset reads
    # from /tmp. Only rewrite when the local CSV actually exists (staging ran),
    # so an unstaged run is untouched.
    if args.local_data_root and "experiment" in params:
        edata = params.get("experiment", {}).get("data", {})
        for key in ("dataset_train", "dataset_val"):
            csv = edata.get(key)
            if csv:
                local_csv = os.path.join(
                    args.local_data_root, "csv", os.path.basename(csv)
                )
                if os.path.exists(local_csv):
                    edata[key] = local_csv

    # Pin the XPU tile for THIS rank BEFORE init_process_group. Required for
    # both xccl and ccl backends; init_process_group does not accept device_id
    # on XPU multi-node (it hangs DataLoader workers — torchtune table).
    # ZE_AFFINITY_MASK was set at module load (line 32-ish) so torch.xpu only
    # sees the one tile for this rank — the valid device index is 0, NOT
    # local_rank. Passing local_rank here errors "index out of range [0,1)".
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(0)

    world_size, rank = init_distributed()

    # Ensure the run folder exists before app_main opens log_r{rank}.csv.
    # Normally submit() pre-creates it on the login node, but when --train_mode
    # is launched directly (e.g. via run_train.sh on a held node) nothing has.
    # Rank 0 creates it; other ranks poll the filesystem instead of using
    # dist.barrier() — barrier on XPU here segfaults because init_process_group
    # was called without device_id (deliberately, per torchtune table; passing
    # device_id hangs DataLoader workers on multi-node XPU).
    folder = params.get("folder")
    if folder:
        if rank == 0:
            Path(folder).mkdir(parents=True, exist_ok=True)
        else:
            import time
            for _ in range(600):
                if Path(folder).exists():
                    break
                time.sleep(0.1)

    if rank == 0:
        logger.info(f"World size: {world_size}; loaded params:")
        pprint.PrettyPrinter(indent=2).pprint(params)

    try:
        # Eval YAMLs (e.g. configs/heads/sarrarp50/*) carry `eval_name` and route
        # through `evals.scaffold.main`. Pretraining YAMLs carry `app` and route
        # through `app.scaffold.main`. Backward compatible: only the presence of
        # the `eval_name` key changes routing.
        if "eval_name" in params:
            eval_main(params["eval_name"], args_eval=params, resume_preempt=False)
        else:
            app_main(params["app"], args=params, resume_preempt=False)
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

        # Aurora teardown guard. Job 8739712's num_workers=2 arm did all its work
        # and still never returned: all 24 ranks wrote all 40 CSV rows at 15:46:53
        # and rank 0 finished the 15.0 GB checkpoint at 15:47:12. Ranks 12 and 15
        # had already exited at 15:47:02-03 with an uncaught
        #   terminate called after throwing an instance of 'std::system_error'
        #   what():  No such file or directory
        # thrown from a C++ destructor at interpreter shutdown. The rung then sat
        # until the walltime cap.
        #
        # Read that exception carefully: rank 0 threw the SAME one at 16:25:45,
        # when PBS sent SIGTERM. So it is what an ordinary signal-kill looks like
        # on this stack, not a signature unique to forked DataLoader workers, and
        # it was never traced to a frame -- whether it is causal or itself a
        # symptom of the shutdown path is not established. What IS specific to the
        # nw2 arm is that two ranks left early, on their own, after their work was
        # done; zero nw0 ranks in the same allocation did.
        #
        # `MP_SOCKET_DIR=/tmp` (module top) and set_sharing_strategy("file_system")
        # (below) were BOTH already active -- do not re-propose them as the fix.
        #
        # A throw out of a destructor during shutdown cannot be caught in Python,
        # so this does not try. It leaves the interpreter before the destructors
        # run: everything that must be durable (checkpoint, per-rank CSV, logs) is
        # already fsynced by this point, and destroy_process_group() above has
        # released the collectives. os._exit skips atexit handlers and static
        # destructors by design -- that is the entire point here.
        #
        # Opt-out with VJEPA_HARD_EXIT=0 if you need a normal exit to debug the
        # throw (e.g. under gdb, or to let faulthandler dump).
        #
        # The exit CODE must not be hardcoded to 0. This runs in a `finally`, so
        # it also executes while a real exception is propagating -- exiting 0
        # there would turn every crash into a silent success and mpiexec would
        # report rc=0 on a run that trained nothing. sys.exc_info() is still
        # populated during finally-on-exception, so use it to pick the code, and
        # print the traceback first since os._exit skips Python's own handler.
        if os.environ.get("VJEPA_HARD_EXIT", "1") == "1":
            _exc = sys.exc_info()[0]
            if _exc is not None:
                traceback.print_exc()
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(1 if _exc is not None else 0)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.train_mode:
        run_training(args)
    else:
        if not args.account:
            parser.error("missing required argument: --account (or env $VJEPA_ACCOUNT)")
        submit(args)
