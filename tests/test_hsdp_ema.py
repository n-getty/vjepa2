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
comparing the FSDP result (gathered to full) against an unsharded reference EMA
computed on plain modules.

Run (needs the torch 2.13 XPU venv or any torch with gloo):
    python -m pytest tests/test_hsdp_ema.py -q
It spawns 2 gloo/CPU ranks via torch.multiprocessing; no GPU required.
"""

import copy
import os

try:
    import pytest
except ImportError:  # allow running as a plain script on Aurora venvs w/o pytest
    pytest = None
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from src.models.utils.modules import Block


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
    out = {}
    for k in tgt_full_sd:
        out[k] = m * tgt_full_sd[k] + (1.0 - m) * enc_full_sd[k]
    return out


def _accel():
    """Return ('xpu'|'cuda'|None). FSDP requires a real accelerator even with a
    CPU mesh, so the test only runs where one exists (a compute node)."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _worker(rank, world_size, dim, depth, heads, m, seed, ret):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29555"
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    dev = _accel()
    # gloo carries the tiny control/gather traffic fine; FSDP still requires the
    # accelerator for its param storage. Pin one device per rank.
    if dev == "xpu":
        torch.xpu.set_device(rank % torch.xpu.device_count())
    elif dev == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            ShardingStrategy,
            StateDictType,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial

        torch.manual_seed(seed)
        encoder = _toy_encoder(dim, depth, heads)
        # Distinct initial weights for the target so the EMA actually moves it.
        target = copy.deepcopy(encoder)
        with torch.no_grad():
            for p in target.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Reference (computed identically on every rank from the pre-wrap fulls).
        enc_ref = {k: v.clone() for k, v in encoder.state_dict().items()}
        tgt_ref = {k: v.clone() for k, v in target.state_dict().items()}
        expected = _reference_ema(enc_ref, tgt_ref, m)

        # 1D mesh (pure shard) is enough to prove shard-aligned EMA — what we're
        # validating is that the _foreach EMA acts on aligned local shards, which
        # is governed by the SHARD axis alone. Production uses the 2D (replicate,
        # shard) mesh with _HYBRID_SHARD_ZERO2, but hybrid strategies REQUIRE a
        # 2D mesh; here we use plain SHARD_GRAD_OP on the 1D mesh, which shards
        # params/grads identically along the shard axis (same shard boundaries
        # the EMA must respect). init_device_mesh with a single dim + no name.
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

        # Gather the updated target to full and compare on rank 0.
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(ftgt, StateDictType.FULL_STATE_DICT, cfg):
            got_full = ftgt.state_dict()

        if rank == 0:
            max_err = 0.0
            missing = []
            for k, exp in expected.items():
                # strip any FSDP/module prefixes to match reference keys
                cand = None
                for gk, gv in got_full.items():
                    if gk.endswith(k):
                        cand = gv
                        break
                if cand is None:
                    missing.append(k)
                    continue
                max_err = max(max_err, (cand.float() - exp.float()).abs().max().item())
            ret["max_err"] = max_err
            ret["missing"] = missing
            ret["n"] = len(expected)
    finally:
        dist.destroy_process_group()


def _run():
    if _accel() is None:
        print("SKIP: no XPU/CUDA accelerator (FSDP requires one; run on a compute node)")
        return None
    world_size = 2
    m = 0.996
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(
        _worker,
        args=(world_size, 32, 4, 4, m, 1234, ret),
        nprocs=world_size,
        join=True,
    )
    assert ret.get("missing") == [], f"target keys missing after gather: {ret.get('missing')}"
    assert ret.get("n", 0) > 0, "no parameters compared"
    # bf16-free CPU path: expect near-exact agreement.
    assert ret["max_err"] < 1e-5, (
        f"HSDP EMA diverged from unsharded reference: max_err={ret['max_err']:.2e}. "
        "This means encoder/target shards are misaligned or the EMA is a no-op."
    )
    return ret["max_err"]


if pytest is not None:

    @pytest.mark.skipif(
        _accel() is None
        or not hasattr(torch.distributed, "is_gloo_available")
        or not torch.distributed.is_gloo_available(),
        reason="FSDP requires an XPU/CUDA accelerator + gloo (run on a compute node)",
    )
    def test_hsdp_ema_matches_unsharded_reference():
        _run()


if __name__ == "__main__":
    err = _run()
    if err is not None:
        print(f"PASS: HSDP EMA matches unsharded reference (max_err={err:.2e})")
