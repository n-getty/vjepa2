# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from src.utils.logging import get_logger

logger = get_logger()


_PMI_VAR_CHAINS = {
    "RANK": ("PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "PALS_RANKID"),
    # Aurora's PALS does NOT export a global size — only PALS_LOCAL_SIZE per
    # node — so we resolve SIZE by either reading PMI_SIZE/PMIX_SIZE/OMPI_*
    # directly, or computing PALS_LOCAL_SIZE * num_nodes from PBS_NODEFILE.
    "SIZE": ("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE"),
    "LOCAL_RANK": (
        "PMI_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "PALS_LOCAL_RANKID",
        "LOCAL_RANK",
    ),
}


def _get_pmi_env(kind):
    for var in _PMI_VAR_CHAINS[kind]:
        if var in os.environ:
            return os.environ[var]
    if kind == "SIZE":
        # PALS fallback: total ranks = local_size * num_unique_nodes.
        local_size = os.environ.get("PALS_LOCAL_SIZE")
        nodefile = os.environ.get("PBS_NODEFILE")
        if local_size and nodefile and os.path.exists(nodefile):
            try:
                with open(nodefile) as f:
                    nodes = {line.strip() for line in f if line.strip()}
                return str(int(local_size) * len(nodes))
            except (OSError, ValueError):
                pass
        # Single-node fallback when no PBS_NODEFILE (e.g. SSH'd into the node).
        if local_size:
            return local_size
    return None


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)


def _select_backend():
    """Choose a torch.distributed backend appropriate for the local accelerator.

    Order of preference:
      1. XPU present (Aurora / Intel Max): prefer xccl if torch ships it, else
         oneCCL via `ccl` (requires `oneccl_bindings_for_pytorch` to be importable).
      2. CUDA present: nccl.
      3. CPU-only: gloo.
    The caller (the platform launcher) is responsible for setting the local
    accelerator device *before* this is reached (e.g. torch.xpu.set_device(local_rank)
    or CUDA_VISIBLE_DEVICES pin); we do not touch device state here.
    """
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    force_xpu = os.environ.get("VJEPA_FORCE_XPU") == "1"
    if xpu_available or force_xpu:
        if hasattr(dist, "is_xccl_available") and dist.is_xccl_available():
            return "xccl"
        try:
            import oneccl_bindings_for_pytorch  # noqa: F401
            return "ccl"
        except Exception as e:
            logger.warning(
                "XPU detected but neither xccl nor oneccl_bindings_for_pytorch "
                f"is available ({e}); falling back to gloo."
            )
            return "gloo"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def _try_mpi4py_broadcast_master_addr():
    """If MASTER_ADDR is missing and we're under MPI, broadcast rank-0's hostname.

    Only invoked when we have PMI/PALS env (i.e. we know we're under mpiexec)
    *and* the launcher hasn't already exported MASTER_ADDR. Safe no-op otherwise.
    """
    try:
        from mpi4py import MPI
    except Exception:
        return None
    comm = MPI.COMM_WORLD
    host = os.uname()[1] if comm.Get_rank() == 0 else None
    host = comm.bcast(host, root=0)
    return host


def init_distributed(port=37129, rank_and_world_size=(None, None)):
    # try to set all environment variables to avoid triggering a segfault
    # environment variables can be reallocated during the execution of torch.distributed.init_process_group
    # the idea is a race condition may trigger if init_progress_group is modifying an environment variable at
    # the same time as Python, so we try to set all environs before initializing distributed
    if "SLURM_JOB_ID" in os.environ:
        # Use the slurm_tmpdir (if it exists) instead of /tmp
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size

    if (rank is None) or (world_size is None):
        # 1) SLURM
        if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ.setdefault("MASTER_ADDR", os.environ.get("HOSTNAME", "localhost"))
        # 2) PMI / PMIX / PALS / OMPI  (PBS + mpiexec on Polaris and Aurora)
        elif _get_pmi_env("RANK") is not None and _get_pmi_env("SIZE") is not None:
            rank = int(_get_pmi_env("RANK"))
            world_size = int(_get_pmi_env("SIZE"))
            master_addr = os.environ.get("MASTER_ADDR")
            if not master_addr:
                nodefile = os.environ.get("PBS_NODEFILE")
                if nodefile and os.path.exists(nodefile):
                    with open(nodefile, "r") as f:
                        master_addr = f.readline().strip()
                else:
                    # last resort: only Aurora gets here; mpi4py is in the
                    # `frameworks` module so the import generally succeeds
                    master_addr = _try_mpi4py_broadcast_master_addr() or os.uname()[1]
                os.environ["MASTER_ADDR"] = master_addr
        # 3) Generic torchrun-style env://
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            os.environ.setdefault("MASTER_ADDR", "localhost")
        else:
            logger.info("No distributed env vars set (distributed training not available)")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            return 1, 0
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")

    if world_size <= 1:
        return 1, 0

    master_port = int(os.environ.get("MASTER_PORT", port))
    os.environ["MASTER_PORT"] = str(master_port)
    backend = _select_backend()
    timeout_seconds = int(os.environ.get("TORCH_DIST_TIMEOUT_SECONDS", "300"))
    logger.info(
        f"init_process_group backend={backend} world_size={world_size} rank={rank} "
        f"master={os.environ['MASTER_ADDR']}:{master_port}"
    )
    # NOTE: we deliberately do *not* pass `device_id=` here. On XPU multi-node
    # it hangs DataLoader workers (BaseMM_PRISM / torchtune both confirm). The
    # launcher pins the device per-rank before this call.
    try:
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=timeout_seconds),
        )
    except Exception:
        logger.exception(
            "Failed to initialize distributed process group "
            f"(rank={rank}, world_size={world_size}, master={os.environ['MASTER_ADDR']}:{master_port})"
        )
        raise

    return world_size, rank


def destroy_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
