# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Unit test for the EMA target-encoder update under HSDP (FSDP1) sharding.

This is the single riskiest piece of the HSDP path (app/vjepa_2_1/hsdp.py +
train.py): the trainer updates the target encoder with

    torch._foreach_mul_(target_params, m)
    torch._foreach_add_(target_params, context_params, alpha=1 - m)

over `encoder.parameters()` / `target_encoder.parameters()`. Under FSDP with
use_orig_params=True those iterators yield each rank's LOCAL SHARD. The update is
only correct if encoder and target_encoder are wrapped with the SAME mesh + wrap
policy so their shards align element-for-element. This test proves that by
comparing the FSDP result (gathered to full) against an unsharded reference EMA.

HOW TO RUN — this is an MPI-NATIVE test (matches how the trainer actually inits
CCL on Aurora). It does NOT use torch.multiprocessing.spawn (which hangs under the
mpi CCL transport). Launch it with mpiexec on a compute node:

    ZE_AFFINITY... is set per-rank from PALS_LOCAL_RANKID before torch import,
    exactly like app/main_dist_aurora.py.

    mpiexec --pmi=pmix -n 2 -ppn 2 --cpu-bind depth --depth 16 \
        python tests/test_hsdp_ema.py

With <2 ranks (e.g. plain `python tests/test_hsdp_ema.py` on a login node) it
SKIPS cleanly. Under pytest it is skipped unless launched under MPI with an
accelerator (pytest can't provide the MPI world), so CI treats it as a
compute-node integration check driven by scripts/vitG384_hsdp_smoke.sh.
"""

import os
import sys

# --- per-rank XPU pin BEFORE torch import (mirrors app/main_dist_aurora.py) ---
for _v in ("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "MPI_LOCALRANKID",
           "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"):
    if _v in os.environ:
        os.environ["ZE_AFFINITY_MASK"] = os.environ[_v]
        break
os.environ.setdefault("MP_SOCKET_DIR", "/tmp")

import copy
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn

from src.models.utils.modules import Block


def _pmi(kind, default):
    # Aurora/PALS sets PMIX_RANK + PALS_RANKID but NOT PMI_SIZE/PMIX_SIZE; the
    # only size var present is PALS_LOCAL_SIZE (ranks/node). This test is
    # single-node (2 ranks) so PALS_LOCAL_SIZE == world size. (The production
    # trainer's init_distributed handles the multi-node size separately.)
    chains = {
        "RANK": ("PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "PALS_RANKID"),
        "SIZE": ("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE", "WORLD_SIZE",
                 "PALS_LOCAL_SIZE"),
    }[kind]
    for k in chains:
        if os.environ.get(k):
            return int(os.environ[k])
    return default


def _accel():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _toy_encoder(dim=32, depth=4, heads=4):
    # A stack of the REAL Block class (the class the HSDP wrap policy targets),
    # so the test exercises the same auto-wrap boundary as production.
    return nn.Sequential(
        *[
            Block(dim=dim, num_heads=heads, use_rope=False, use_sdpa=False)
            for _ in range(depth)
        ]
    )


def _reference_ema(enc_full_sd, tgt_full_sd, m):
    """Unsharded reference: new_target = m*target + (1-m)*context, per tensor."""
    return {k: m * tgt_full_sd[k] + (1.0 - m) * enc_full_sd[k] for k in tgt_full_sd}


def _run_mpi(dim=32, depth=4, heads=4, m=0.996, seed=1234):
    dev = _accel()
    if dev is None:
        print("SKIP: no XPU/CUDA accelerator (run on a compute node under mpiexec)")
        return None
    rank = _pmi("RANK", 0)
    world_size = _pmi("SIZE", 1)
    if world_size < 2:
        print(f"SKIP: need >=2 MPI ranks, got world_size={world_size} "
              "(launch with `mpiexec -n 2 python tests/test_hsdp_ema.py`)")
        return None

    if dev == "xpu":
        torch.xpu.set_device(0)  # ZE_AFFINITY_MASK already pinned one tile
        backend = "xccl" if dist.is_xccl_available() else "gloo"
    else:
        torch.cuda.set_device(0)
        backend = "nccl"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    result = None
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            ShardingStrategy,
            StateDictType,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        torch.manual_seed(seed)
        dev_t = torch.device(f"{dev}:0")
        encoder = _toy_encoder(dim, depth, heads).to(dev_t)
        target = copy.deepcopy(encoder)
        with torch.no_grad():
            for p in target.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Reference computed identically on every rank from the pre-wrap fulls.
        enc_ref = {k: v.clone() for k, v in encoder.state_dict().items()}
        tgt_ref = {k: v.clone() for k, v in target.state_dict().items()}
        expected = _reference_ema(enc_ref, tgt_ref, m)

        # 1D mesh + plain SHARD_GRAD_OP: validates EMA over aligned shards (the
        # shard axis is what the _foreach EMA runs over). Hybrid strategies need
        # a 2D mesh; production build_hsdp_mesh supplies that. Here 1D suffices.
        mesh = init_device_mesh(dev, (world_size,))
        wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Block}
        )
        common = dict(
            auto_wrap_policy=wrap_policy,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_mesh=mesh,
            use_orig_params=True,
            sync_module_states=False,  # keep our distinct init on each module
        )
        fenc = FSDP(encoder, **common)
        ftgt = FSDP(target, **common)
        for p in ftgt.parameters():
            p.requires_grad = False

        # The EXACT trainer EMA op (app/vjepa_2_1/train.py).
        params_k, params_q = [], []
        for pq, pk in zip(fenc.parameters(), ftgt.parameters()):
            params_q.append(pq)
            params_k.append(pk)
        with torch.no_grad():
            torch._foreach_mul_(params_k, m)
            torch._foreach_add_(params_k, params_q, alpha=1 - m)

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(ftgt, StateDictType.FULL_STATE_DICT, cfg):
            got_full = ftgt.state_dict()

        if rank == 0:
            max_err, missing = 0.0, []
            for k, exp in expected.items():
                cand = next((gv for gk, gv in got_full.items() if gk.endswith(k)), None)
                if cand is None:
                    missing.append(k)
                    continue
                # FULL_STATE_DICT with offload_to_cpu returns CPU tensors; the
                # reference `expected` is on the accelerator. Compare on CPU.
                max_err = max(
                    max_err,
                    (cand.float().cpu() - exp.float().cpu()).abs().max().item(),
                )
            assert missing == [], f"target keys missing after gather: {missing}"
            assert len(expected) > 0, "no parameters compared"
            assert max_err < 1e-5, (
                f"HSDP EMA diverged from unsharded reference: max_err={max_err:.2e}. "
                "encoder/target shards are misaligned or the EMA is a no-op."
            )
            result = max_err
    finally:
        dist.barrier()
        dist.destroy_process_group()
    return result


# pytest entry: only meaningful when already launched under MPI with an
# accelerator; otherwise skip (pytest can't spawn the MPI world itself).
try:
    import pytest

    @pytest.mark.skipif(
        _accel() is None or _pmi("SIZE", 1) < 2,
        reason="run via `mpiexec -n 2 python tests/test_hsdp_ema.py` on a compute node",
    )
    def test_hsdp_ema_matches_unsharded_reference():
        _run_mpi()
except ImportError:
    pass


if __name__ == "__main__":
    err = _run_mpi()
    if err is not None:
        print(f"PASS: HSDP EMA matches unsharded reference (max_err={err:.2e})")
        sys.exit(0)
