# Polaris (PBS + mpiexec) multi-node launcher for V-JEPA 2.1.
#
# Two modes (selected by --train_mode):
#   login-node: read YAML, materialize <folder>/params-pretrain.yaml,
#               generate <folder>/_submit.pbs, qsub it.
#   compute-node (--train_mode): set CUDA_VISIBLE_DEVICES from PMI local rank,
#                                let init_distributed() pick up PMI_RANK/PMI_SIZE,
#                                hand off to app.scaffold.main.
#
# Data staging to /local/scratch is opt-in via --stage_data and assumes
# data.datasets entries are directories (WebDataset-style). The 2.1 ablation
# configs use CSV files of video paths, so leave staging off for those.

import argparse
import os
import sys


def _configure_threading_env():
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(key, "1")


_configure_threading_env()


# Defensive: pin CUDA_VISIBLE_DEVICES *before* importing torch. On current
# torch (2.5.1) the late pin works correctly (torch re-reads the env on
# device_count()), but other torch versions cache the device list at import
# time. Pinning early avoids that footgun across versions.
if "--train_mode" in sys.argv:
    for var in ("PMI_LOCAL_RANK", "MPI_LOCALRANKID",
                "OMPI_COMM_WORLD_LOCAL_RANK", "PALS_LOCAL_RANKID", "LOCAL_RANK"):
        if var in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[var]
            break

import copy
import datetime
import pprint
import shutil
import subprocess
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from src.utils.logging import get_logger, git_information

logger = get_logger(force=True)


parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, default="configs.yaml",
                    help="YAML config file to launch.")
parser.add_argument("--folder", type=str, default=None,
                    help="If set, overrides the 'folder' field in the YAML.")
parser.add_argument("--account", type=str, default=os.environ.get("VJEPA_ACCOUNT"),
                    help="PBS account. Defaults to $VJEPA_ACCOUNT. Required.")
parser.add_argument("--partition", type=str, default="prod",
                    help="PBS queue.")
parser.add_argument("--qos", type=str, default=None)
parser.add_argument("--time", type=int, default=720,
                    help="Wall time in minutes.")
parser.add_argument("--filesystems", type=str, default="home:eagle",
                    help="PBS -l filesystems= value.")
parser.add_argument("--nodes", type=int, default=None,
                    help="Override 'nodes' field in YAML.")
parser.add_argument("--tasks_per_node", type=int, default=None,
                    help="Override 'tasks_per_node' field (GPUs per node, e.g. 4 on Polaris).")
parser.add_argument("--cpus_per_task", type=int, default=8)
parser.add_argument("--master_port", type=int, default=29500)
parser.add_argument("--stage_data", action="store_true",
                    help="Rsync data.datasets entries to /local/scratch on each node before training. "
                         "Only correct if entries are directories (WebDataset shards), not CSV files.")
parser.add_argument("--no_copy_code", action="store_true",
                    help="Skip copying the code tree into <folder>/code (use repo as-is).")
parser.add_argument("--extra_env", action="append", default=[],
                    help="Extra `KEY=VAL` env vars to export in the PBS script (repeatable).")
parser.add_argument("--venv", type=str,
                    default=os.environ.get("PYTHON_BIN_VENV"),
                    help="venv to activate inside the PBS job (after `module load conda`). "
                         "Defaults to $PYTHON_BIN_VENV. Required.")
parser.add_argument("--disable_aws_ofi", action="store_true",
                    help="Disable the AWS OFI NCCL plugin. Default ON: measured 1.32x speedup "
                         "vs the stock NCCL config on Polaris for V-JEPA 2.1 (2026-05-20). "
                         "Set this flag if you see NCCL hangs (rare).")
parser.add_argument("--depends_on", type=str, default=None,
                    help="PBS job id this job should run after (afterok dependency).")
parser.add_argument("--dry_run", action="store_true",
                    help="Build PBS script and dump it, but don't qsub.")

# compute-node-only flags
parser.add_argument("--train_mode", action="store_true",
                    help="Internal: signals execution on a compute node.")
parser.add_argument("--local_data_root", type=str, default=None,
                    help="Internal: /local/scratch root to remap data.datasets onto.")
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


def build_pbs_script(folder, job_name, account, partition, qos, walltime_minutes,
                     nodes, tasks_per_node, cpus_per_task, filesystems, master_port,
                     params_path, code_folder, stage_data, dataset_paths, extra_env,
                     venv, use_aws_ofi=True):
    nnodes = int(nodes)
    nranks = int(tasks_per_node)
    ndepth = int(cpus_per_task)
    ntotranks = nnodes * nranks

    hours, mins = divmod(int(walltime_minutes), 60)
    walltime = f"{hours:02d}:{mins:02d}:00"

    polaris_env = [
        "ulimit -c unlimited",
        "module use /soft/modulefiles",
        "module load conda",
        "conda activate",
        f"source {venv}/bin/activate" if venv else "",
        "export MPICH_GPU_SUPPORT_ENABLED=1",
        "export FI_PROVIDER=cxi",
        "export NCCL_DEBUG=WARN",
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
        "export NCCL_LAUNCH_MODE=GROUP",
        "export PYTHONFAULTHANDLER=1",
    ]
    if use_aws_ofi:
        # AWS OFI NCCL plugin + CXI tuning per ALCF Polaris docs.
        # Measured 1.32x speedup on 2-node V-JEPA 2.1 ViT-L 256px (2026-05-20):
        # iter-time 1623ms -> 1230ms, backward 1014ms -> 609ms.
        polaris_env += [
            "export NCCL_NET_GDR_LEVEL=PHB",
            "export NCCL_CROSS_NIC=1",
            "export NCCL_COLLNET_ENABLE=1",
            'export NCCL_NET="AWS Libfabric"',
            "export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:${LD_LIBRARY_PATH:-}",
            "export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:${LD_LIBRARY_PATH:-}",
            "export FI_CXI_DISABLE_HOST_REGISTER=1",
            "export FI_MR_CACHE_MONITOR=userfaultfd",
            "export FI_CXI_DEFAULT_CQ_SIZE=131072",
            "export MPICH_OFI_NIC_POLICY=NUMA",
        ]
    else:
        polaris_env += [
            "unset NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_COLLNET_ENABLE NCCL_NET",
        ]
    polaris_env = [line for line in polaris_env if line]
    for kv in extra_env:
        polaris_env.append(f"export {kv}")

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
                        f'export LOCAL_DATA_DIR="/local/scratch/vjepa_data/{job_name}_${{PBS_JOBID}}"',
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
            f'mpiexec -n {nnodes} -ppn 1 --cpu-bind none {stage_script_path}',
            f'export LOCAL_DATA_DIR="/local/scratch/vjepa_data/{job_name}_${{PBS_JOBID}}"',
            'echo "--- staging complete ---"',
        ])
        local_data_flag = "--local_data_root $LOCAL_DATA_DIR"

    qos_line = f"#PBS -l qos={qos}" if qos else ""

    script = f"""#!/bin/bash -l
#PBS -N {job_name}
#PBS -l select={nnodes}
#PBS -l walltime={walltime}
#PBS -l filesystems={filesystems}
#PBS -q {partition}
#PBS -A {account}
{qos_line}

set -eo pipefail

echo "JOB START: $(date)"
echo "Nodes: {nnodes} Total ranks: {ntotranks} ppn: {nranks} cpus/task: {ndepth}"

cd {code_folder}

{chr(10).join(polaris_env)}

{chr(10).join(master)}

{staging_block}

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

mpiexec -envall -n {ntotranks} -ppn {nranks} \\
    --cpu-bind depth -d {ndepth} \\
    -genv OMP_NUM_THREADS 8 \\
    python -m app.main_dist_polaris --train_mode \\
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
    tasks_per_node = int(params.get("tasks_per_node", 4))
    cpus_per_task = int(params.get("cpus_per_task", args.cpus_per_task))

    # AWS OFI plugin segfaults on single-node DDP (no inter-node traffic for it
    # to handle). Auto-disable when nodes==1. Multi-node = always on by default.
    use_aws_ofi = not args.disable_aws_ofi and nodes > 1
    if nodes == 1 and not args.disable_aws_ofi:
        logger.info("Single-node run: auto-disabling AWS OFI plugin (causes "
                    "DDP init segfault on 1 node).")

    pbs = build_pbs_script(
        folder=folder, job_name=job_name, account=args.account,
        partition=args.partition, qos=args.qos, walltime_minutes=args.time,
        nodes=nodes, tasks_per_node=tasks_per_node, cpus_per_task=cpus_per_task,
        filesystems=args.filesystems, master_port=args.master_port,
        params_path=params_path, code_folder=code_folder,
        stage_data=args.stage_data, dataset_paths=dataset_paths,
        extra_env=args.extra_env, venv=args.venv,
        use_aws_ofi=use_aws_ofi,
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
        logger.error("qsub not found. Run from a Polaris login node.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"qsub failed: {e.stderr}")
        raise


# ----------------------------------------------------------------------------
# Compute-node: pin GPU, init dist, hand off to scaffold.main
# ----------------------------------------------------------------------------

def run_training(args):
    from src.utils.distributed import _get_pmi_env, init_distributed
    import torch

    # CUDA_VISIBLE_DEVICES was pinned at module load (before torch import) when
    # --train_mode was on sys.argv — see top-of-file note. Don't repin here.
    local_rank = _get_pmi_env("LOCAL_RANK") or "0"

    if not os.environ.get("MASTER_ADDR"):
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            with open(nodefile) as f:
                os.environ["MASTER_ADDR"] = f.readline().strip()
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

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

    world_size, rank = init_distributed()
    torch.cuda.set_device(0)

    if rank == 0:
        logger.info(f"World size: {world_size}; loaded params:")
        pprint.PrettyPrinter(indent=2).pprint(params)

    try:
        app_main(params["app"], args=params, resume_preempt=False)
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.train_mode:
        run_training(args)
    else:
        missing = []
        if not args.account:
            missing.append("--account (or env $VJEPA_ACCOUNT)")
        if not args.venv:
            missing.append("--venv (or env $PYTHON_BIN_VENV)")
        if missing:
            parser.error("missing required argument(s): " + ", ".join(missing))
        submit(args)
