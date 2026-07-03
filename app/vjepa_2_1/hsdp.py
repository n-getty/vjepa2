# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Optional HSDP (Hybrid Sharded Data Parallel, FSDP1) wrapping for the V-JEPA
2.1 trainer on Aurora XPU.

WHY THIS EXISTS
---------------
The ViT-G *2B* continued-pretrain wedges under DDP at 16 nodes: each tile holds
the full 2B fp32 params + grads + Adam m/v + a full fp32 target-encoder copy
(~43 GB fixed) and sits at ~90% of the 64 GB tile. At that occupancy Aurora's
CCL has no L0 headroom to absorb its ~10-30 MiB/step external-memory growth
(IPC handles + OFI fabric registrations), so the gradient collective stalls
every backward and the per-iter time climbs monotonically (9s -> 300s).
The 1B model (bs=1) had headroom and ran flat for thousands of iters.

HSDP shards params/grads/optimizer across the 12 intra-node tiles (replicate
across nodes), taking fixed state from ~43 GB -> ~20 GB/tile (~30% occupancy).
That restores the L0 headroom and removes the wedge at its cause. torchtune
measured that sharded parallelism WITHOUT weight-sync grows external memory only
~10 MiB over 100 steps (vs 3000 MiB with GRPO weight-sync); V-JEPA pretraining
has no weight-sync, so HSDP is expected to be stable for long runs with no
checkpoint-restart.

This module is ADDITIVE: it is only imported when VJEPA_DIST_STRATEGY=hsdp.
The DDP path in train.py is untouched and remains the default.

Design references (adapted, not imported):
  - BaseMM_PRISM .../scaling-study .../src/training/distributed.py:_wrap_hsdp
    (validated on Aurora XPU: _HYBRID_SHARD_ZERO2, 2D device_mesh, transformer
    auto-wrap, backward/forward prefetch).
"""

import os
from functools import partial

import torch

from src.models.utils.modules import Block

# ZERO2 (shard_grad_op) keeps params gathered after forward (no re-AllGather in
# backward) — less comm, a bit more memory than full_shard, but Aurora has ample
# headroom once fixed state drops to ~20 GB. This is PRISM's recommended default
# and the user-selected default here. FSDP_SHARDING=full_shard opts into
# HYBRID_SHARD (params sharded + re-AllGathered) if ever HBM-constrained.
_DEFAULT_FSDP_SHARDING = "shard_grad_op"


def _local_world_size():
    """Ranks per node (12 tiles on Aurora). Prefer an explicit export, then the
    PALS value mpiexec sets, then a safe default."""
    for k in ("LOCAL_WORLD_SIZE", "PALS_LOCAL_SIZE", "MPI_LOCALNRANKS"):
        v = os.environ.get(k)
        if v:
            try:
                n = int(v)
                if n > 0:
                    return n
            except ValueError:
                pass
    return 12


def build_hsdp_mesh(world_size, device_type="xpu", logger=None):
    """Build a 2D (replicate=nodes, shard=tiles/node) device mesh for HSDP.

    Returns (mesh, num_nodes, local_world_size). Raises if the topology is
    inconsistent (world_size not divisible by local_world_size) rather than
    silently producing a wrong mesh.
    """
    from torch.distributed.device_mesh import init_device_mesh

    lws = _local_world_size()
    if world_size % lws != 0:
        raise ValueError(
            f"HSDP: world_size={world_size} is not divisible by "
            f"local_world_size={lws}; set LOCAL_WORLD_SIZE correctly."
        )
    num_nodes = world_size // lws
    mesh = init_device_mesh(
        device_type,
        (num_nodes, lws),
        mesh_dim_names=("replicate", "shard"),
    )
    if logger is not None:
        logger.info(
            f"[HSDP] device mesh: {num_nodes} nodes x {lws} ranks/node "
            f"(replicate x shard), device_type={device_type}"
        )
    return mesh, num_nodes, lws


def _resolve_sharding(logger=None):
    from torch.distributed.fsdp import ShardingStrategy

    env = os.environ.get("FSDP_SHARDING", _DEFAULT_FSDP_SHARDING).lower()
    if env == "shard_grad_op":
        try:
            return ShardingStrategy._HYBRID_SHARD_ZERO2, "_HYBRID_SHARD_ZERO2 (shard_grad_op intra-node, DDP inter-node)"
        except AttributeError:
            if logger is not None:
                logger.warning(
                    "[HSDP] _HYBRID_SHARD_ZERO2 unavailable; using HYBRID_SHARD"
                )
            return ShardingStrategy.HYBRID_SHARD, "HYBRID_SHARD (fallback)"
    return ShardingStrategy.HYBRID_SHARD, "HYBRID_SHARD (full_shard intra-node, DDP inter-node)"


def _use_toplevel_wrap():
    """Whether to wrap each module as a SINGLE top-level FSDP unit (no
    auto_wrap_policy) vs. per-Block units.

    PRISM and torchtune both document that PER-MODULE (per-transformer-layer)
    FSDP wrapping causes CATASTROPHIC overhead on Aurora XPU: each wrapped unit
    issues its own inter-node collective, so a 72-Block ViT-G 2B mints ~72
    AllGather/ReduceScatter triples per step instead of ~1. On XPU that is not
    just latency — every collective registers Level-Zero IPC handles keyed by
    buffer VA, so 72x the collectives = 72x the IPC-handle churn rate, which is
    the accumulator behind the observed 16n iter-time drift (8.9s -> 32s over 80
    iters at flat torch-mem). torchtune CLAUDE.md:15 / PRISM distributed.py:458.

    For SHARD_GRAD_OP (_HYBRID_SHARD_ZERO2) params stay resident after forward,
    so a single top-level unit does ONE ReduceScatter per backward (like DDP's
    AllReduce) — this is PRISM's validated production default. Only FULL_SHARD /
    HYBRID_SHARD (params re-AllGathered) NEEDS per-layer units for memory, and
    PRISM's table shows that path is the slow one on XPU. Default: top-level for
    shard_grad_op, per-layer only for full_shard. Override with HSDP_WRAP=
    {toplevel,perlayer}.
    """
    override = os.environ.get("HSDP_WRAP", "").lower()
    if override in ("toplevel", "top-level", "top_level", "none"):
        return True
    if override in ("perlayer", "per-layer", "per_layer", "block"):
        return False
    env = os.environ.get("FSDP_SHARDING", _DEFAULT_FSDP_SHARDING).lower()
    return env == "shard_grad_op"


def wrap_hsdp(module, mesh, *, requires_grad=True, logger=None):
    """FSDP1-wrap an encoder/predictor/target_encoder for HSDP.

    - transformer_auto_wrap_policy on the shared `Block` class (encoder AND
      predictor blocks are the same class), so each attention block is its own
      FSDP unit and the inter-node grad ReduceScatter overlaps the next block's
      backward.
    - use_orig_params=True is REQUIRED: the optimizer builds param groups by
      named-parameter filtering (bias/1D -> no weight decay, app/vjepa_2_1/
      utils.py:init_opt). Flat-param mode cannot express per-name WD groups.
    - sync_module_states=True: all shards start from identical weights (matters
      for the deepcopy'd target_encoder and for resuming from a full checkpoint).
    - mixed_precision bf16 mirrors the trainer autocast; reduce_dtype=bf16 makes
      the gradient collective bf16 (same intent as the DDP bf16 comm hook, which
      we skip on the HSDP path).

    target_encoder MUST be wrapped with the SAME mesh + policy as encoder so the
    EMA `_foreach` ops act on aligned shards (verified by tests/test_hsdp_ema.py).
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    sharding, label = _resolve_sharding(logger=logger)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    toplevel = _use_toplevel_wrap()
    kwargs = dict(
        mixed_precision=mp_policy,
        sharding_strategy=sharding,
        device_mesh=mesh,
        use_orig_params=True,
        # NB: sync_module_states=True HANGS at FSDP construction on Aurora
        # xccl (its rank-0 broadcast never completes; isolated on a held node
        # 2026-07-03 — bare/device_id wrap in ~0.2s, sync_module_states hangs
        # >900s). We don't need it: the encoder loads a full checkpoint under
        # FULL_STATE_DICT (every rank applies identical weights) and
        # target_encoder is a deepcopy of encoder before wrap, so all ranks
        # already start from identical module states. PRISM's validated wrap
        # also omits it.
        limit_all_gathers=True,
    )
    if toplevel:
        # TOP-LEVEL wrap: one FSDP unit for the whole module -> ONE ReduceScatter
        # per backward (like DDP AllReduce). No auto_wrap_policy. This is the
        # PRISM/torchtune-validated XPU default for shard_grad_op: it avoids the
        # ~72x collective count (and ~72x IPC-handle churn) of per-Block wrapping
        # that drove the 16n iter-time drift. Prefetch is a no-op with a single
        # unit, so it is left off.
        wrap_desc = "top-level (single unit, no auto_wrap_policy)"
    else:
        # PER-LAYER wrap: only for FULL_SHARD/HYBRID_SHARD, where params are
        # re-AllGathered per unit and per-layer granularity is needed for memory.
        # PRISM's table flags this path as the slow one on XPU; use only when
        # HBM-constrained. Prefetch overlaps the many collectives here.
        kwargs["auto_wrap_policy"] = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Block}
        )
        from torch.distributed.fsdp import BackwardPrefetch

        if os.environ.get("HSDP_BACKWARD_PREFETCH", "1") == "1":
            kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
        if os.environ.get("HSDP_FORWARD_PREFETCH", "1") == "1":
            kwargs["forward_prefetch"] = True
        wrap_desc = "per-Block (transformer_auto_wrap_policy)"

    wrapped = FSDP(module, **kwargs)
    if not requires_grad:
        for p in wrapped.parameters():
            p.requires_grad = False
    if logger is not None:
        logger.info(
            f"[HSDP] wrapped module: {label}; wrap={wrap_desc} "
            f"(requires_grad={requires_grad})"
        )
    return wrapped


def full_state_dict_context(module):
    """Context manager to get/load a FULL (unsharded, rank0) state_dict from an
    FSDP module, so checkpoints stay DDP-compatible and small on disk.

    Usage:
        with full_state_dict_context(encoder):
            sd = encoder.state_dict()          # full on rank0, empty elsewhere
        ...
        with full_state_dict_context(encoder):
            encoder.load_state_dict(sd)        # broadcasts to shards
    """
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    return FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg)


def full_state_dict_context_multi(modules):
    """Combined FULL_STATE_DICT context over several FSDP modules at once.

    load_pretrained / load_checkpoint call .load_state_dict on encoder,
    predictor and target_encoder internally, so all three must be in the
    FULL_STATE_DICT state-dict type simultaneously for the load to broadcast
    correctly to shards. Uses a load-oriented config (rank0_only=False,
    broadcast) so every rank ends up with the loaded weights.
    """
    import contextlib as _ctx

    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    # rank0_only=False so the full dict is materialized/broadcast to every rank
    # during load_state_dict (offload_to_cpu keeps peak memory bounded).
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    es = _ctx.ExitStack()
    for m in modules:
        if m is not None:
            es.enter_context(
                FSDP.state_dict_type(m, StateDictType.FULL_STATE_DICT, cfg)
            )
    return es
