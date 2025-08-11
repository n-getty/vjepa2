# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist

from src.utils.logging import get_logger

logger = get_logger()

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
except ImportError:
    ipex = None


def init_distributed(port=37129, rank_and_world_size=(None, None), backend=None):
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

    # Try MPI initialization if requested or available
    use_mpi = (backend == 'ccl') or (MPI is not None and 'SLURM_NTASKS' not in os.environ)
    if use_mpi:
        try:
            comm = MPI.COMM_WORLD
            world_size = comm.Get_size()
            rank = comm.Get_rank()

            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)

            master_addr = socket.gethostname() if rank == 0 else None
            master_addr = comm.bcast(master_addr, root=0)
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = str(port)

            dist_backend = 'ccl' if ipex is not None else 'gloo'
            if backend is not None:
                dist_backend = backend

            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            logger.info(f"Initialized distributed training via MPI: world_size={world_size}, rank={rank}, backend='{dist_backend}'")
            return world_size, rank

        except Exception as e:
            logger.info(f"MPI initialization failed, falling back. Error: {e}")
            world_size, rank = 1, 0


    # Fallback to SLURM or local
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ["MASTER_PORT"] = str(port)
        dist_backend = 'nccl' if backend is None else backend
        torch.distributed.init_process_group(backend=dist_backend, world_size=world_size, rank=rank)
        logger.info(f"Initialized distributed training via SLURM/local: world_size={world_size}, rank={rank}, backend='{dist_backend}'")
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"Rank: {rank}. Distributed training not available {e}")

    return world_size, rank


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
