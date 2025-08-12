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
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size

    # Check for MPI launch environment variables
    is_mpi_launch = 'OMPI_COMM_WORLD_RANK' in os.environ or 'PMI_RANK' in os.environ or 'PALS_LOCAL_RANKID' in os.environ

    if is_mpi_launch:
        if MPI is None:
            raise RuntimeError("MPI environment detected, but mpi4py is not installed.")

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

            dist_backend = 'ccl'
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method='env://'
            )
            logger.info(f"Initialized distributed training via MPI: world_size={world_size}, rank={rank}, backend='{dist_backend}'")
            return world_size, rank
        except Exception as e:
            logger.error(f"MPI initialization failed: {e}")
            raise e

    # Handle single-node, multi-process launch (from app/main.py) or SLURM
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(port))

    if rank is None or world_size is None:
        # Try to get from SLURM
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            if "MASTER_ADDR" not in os.environ and "HOSTNAME" in os.environ:
                 os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        except Exception:
            logger.info("SLURM vars not set, assuming single process.")
            world_size, rank = 1, 0
            return world_size, rank

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    try:
        if backend is None:
            if torch.cuda.is_available():
                backend = 'nccl'
            elif ipex is not None and ipex.xpu.is_available():
                backend = 'ccl'
            else:
                backend = 'gloo'

        torch.distributed.init_process_group(backend=backend, init_method='env://')
        logger.info(f"Initialized distributed training via SLURM/local: world_size={world_size}, rank={rank}, backend='{backend}'")

    except Exception as e:
        logger.error(f"Distributed training initialization failed: {e}")
        world_size, rank = 1, 0

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
