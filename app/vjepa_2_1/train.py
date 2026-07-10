# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import os
import socket

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
from app.vjepa_2_1.models.utils.modules import Lambda_LinearWarmupHold
from app.vjepa_2_1.transforms import make_transforms
from app.vjepa_2_1.utils import (
    init_opt,
    init_video_model,
    load_checkpoint,
    load_pretrained,
    normalize_nested,
)
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, PhaseTimer, get_logger, gpu_timer
from torch.nn.parallel import DistributedDataParallel


log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
MAX_REPEAT_COUNTS = 10

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", True)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    logger.info(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD')}")
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    is_causal = cfgs_model.get("is_causal", False)
    pred_is_causal = cfgs_model.get("pred_is_causal", False)
    init_type = cfgs_model.get("init_type", "default")
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size", None)
    n_registers = cfgs_model.get("n_registers", 0)
    has_cls_first = cfgs_model.get("has_cls_first", False)
    interpolate_rope = cfgs_model.get("interpolate_rope", False)
    lambda_value_img = cfgs_model.get("lambda_value_img", 0.0)
    lambda_value_vid = cfgs_model.get("lambda_value_vid", 0.0)
    n_registers_predictor = cfgs_model.get("n_registers_predictor", 0)
    lambda_progressive = cfgs_model.get("lambda_progressive", True)
    normalize_predictor = cfgs_model.get("normalize_predictor", False)
    modality_embedding = cfgs_model.get("modality_embedding", False)
    levels_predictor = cfgs_model.get("levels_predictor", 4)
    # Provisional encoder embed_dim by name (kept for the 3 canonical sizes); the AUTHORITATIVE value
    # is read from the built encoder right after init_video_model (below) so ANY ladder size works.
    # Without that, unknown sizes (tiny/small/base — used by the scaling sweep) left this unbound and
    # crashed at forward_target's `embed_dim=embed_dim_encoder` default (NameError). See scaling study.
    if model_name == "vit_large":
        embed_dim_encoder = 1024
    elif model_name == "vit_giant_xformers":
        embed_dim_encoder = 1408
    elif model_name == "vit_gigantic_xformers":
        embed_dim_encoder = 1664
    else:
        embed_dim_encoder = None  # set authoritatively from the built encoder below

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    # Per-sample source-selection temperature for WebDataset RandomMix.
    # 1.0 = size-proportional (uniform over true corpus); 0.5 = sqrt-size
    # (default, down-weights tiny sets without erasing them); 0.0 = the old
    # uniform-per-source behavior that catastrophically oversampled tiny sets.
    sampling_temperature = cfgs_data.get("sampling_temperature", 0.5)
    # Degenerate-clip reject floor (raw 0-255 per-pixel std). None -> use the
    # webdataset module default (env VJEPA_MIN_CLIP_STD, default 1.0). Drops
    # pure-black/frozen clips (e.g. surgvu24's ~19% byte-identical black mp4)
    # before they enter a batch. Set 0 in config to disable.
    min_clip_std = cfgs_data.get("min_clip_std")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    # Gradient accumulation (VJEPA_GRAD_ACCUM, default 1 = unchanged). Splits the
    # per-rank batch into `grad_accum` microbatches processed sequentially, with
    # DDP gradient sync deferred to the final microbatch (no_sync on the rest) so
    # the optimizer still sees the full-batch gradient in ONE allreduce per step.
    # On Aurora XPU this lowers the per-microbatch activation peak, buying back L0
    # headroom so CCL's ~10-30 MiB/step external-memory growth (the ViT-G 2B
    # backward-wedge; see docs/vitG_2B_spike_investigation.md) has room to breathe.
    # LR/wd/EMA/momentum still step ONCE per optimizer step, so schedules are
    # unchanged. Validated below to divide batch_size and keep micro-bs >= 1.
    grad_accum = int(os.environ.get("VJEPA_GRAD_ACCUM", "1"))
    # TRUE gradient accumulation (VJEPA_TRUE_ACCUM, default 1 = unchanged). Distinct
    # from VJEPA_GRAD_ACCUM above: that one SLICES a single loader batch into
    # microbatches (lowers activation peak, effective batch unchanged, same #collectives
    # /step). VJEPA_TRUE_ACCUM fetches `true_accum` SEPARATE loader batches per optimizer
    # step, runs fwd/bwd on each with the inter-node collective deferred (no_sync) to the
    # LAST one — so there is ONE ReduceScatter/AllReduce per optimizer step instead of
    # true_accum, halving (at N=2) the inter-node collective FREQUENCY. This is the real
    # fabric-contention lever (the §4g host-side collective stalls scale with collective
    # count). Effective global batch scales by N (192*bs2*N). LR kept unchanged (see
    # memory true-accum-lr-decision: a 2x-larger, less-noisy gradient at fixed LR is more
    # conservative per-sample, and LR-hotness is this model's known collapse mode).
    # no_sync ON is affordable here (2B bf16 grad ~4GB vs ~15GB free L0) — this DIVERGES
    # from PRISM's 7B (no_sync OFF because 14GB grad wouldn't fit); our 2B is fabric-bound
    # not memory-bound. LR/wd/EMA/momentum still step ONCE per optimizer step.
    true_accum = int(os.environ.get("VJEPA_TRUE_ACCUM", "1"))
    # Distributed strategy: "ddp" (default, unchanged) or "hsdp" (FSDP1
    # HYBRID_SHARD; shards params/grads/optimizer intra-node to buy back L0
    # headroom on Aurora and remove the 2B backward wedge). See app/vjepa_2_1/hsdp.py.
    dist_strategy = os.environ.get("VJEPA_DIST_STRATEGY", "ddp").lower()

    # SAFETY GUARD: activation-checkpointing-off is only affordable when HSDP
    # shards the optimizer state (2B ViT-g @ 384: ~22 GB HSDP baseline vs ~57 GB
    # DDP). Under DDP the full 48-layer activation set pushes the tile to ~82 GB
    # and OOMs (measured 2026-07-09, job 8659973). The active vitG384 configs ship
    # ckpt-off as the default recipe (they are always launched HSDP), so protect
    # any DDP launch of them from a guaranteed OOM by forcing ckpt back on.
    if (not use_activation_checkpointing) and dist_strategy == "ddp":
        logger.warning(
            "use_activation_checkpointing=false is unsafe under DDP (OOMs the 2B "
            "at ~82 GB); forcing it ON. Set VJEPA_DIST_STRATEGY=hsdp to keep it off."
        )
        use_activation_checkpointing = True

    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    grid_size = crop_size // patch_size
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    # Env override for the dataloader worker count. Needed for the HSDP path:
    # init_device_mesh creates inter-node xccl subgroup PGs, and forking
    # persistent DataLoader workers AFTER that can inherit broken xccl state and
    # deadlock on the first batch read (PRISM documents this failure mode). Set
    # VJEPA_NUM_WORKERS=0 to fork no workers on the HSDP path. Default: config.
    _nw_override = os.environ.get("VJEPA_NUM_WORKERS")
    if _nw_override is not None:
        num_workers = int(_nw_override)
    # Env override for pin_memory — a Mode-A (DataLoader shm-crash) isolation knob.
    # VJEPA_PIN_MEM=0 disables the pinned-host-memory staging buffer. (Note:
    # persistent_workers is NOT plumbed through init_data here, so it is already
    # effectively False regardless of config — not a Mode-A variable.)
    _pin_override = os.environ.get("VJEPA_PIN_MEM")
    if _pin_override is not None:
        pin_mem = _pin_override == "1"

    # -- IMG DATA
    cfgs_img_data = args.get("img_data")
    img_rank_ratio = 0.25
    img_mask = None
    if cfgs_img_data is not None:
        img_dataset_type = cfgs_img_data.get("dataset_type", "imagenet")
        img_dataset_paths = cfgs_img_data.get("datasets", [])
        img_dataset_weights = cfgs_img_data.get("datasets_weights", [])
        img_dataset_fpcs = cfgs_img_data.get("dataset_fpcs")
        img_dataset_batch_size = cfgs_img_data.get("batch_size")
        img_rank_ratio = cfgs_img_data.get("rank_ratio", img_rank_ratio)
        img_num_workers = cfgs_img_data.get("num_workers", num_workers)

        img_mask = args.get("img_mask", img_mask)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    shift_by_n = cfgs_loss.get("shift_by_n")
    predict_all = cfgs_loss.get("predict_all", True)
    weight_distance_loss = cfgs_loss.get("weight_distance_loss", False)
    offset_context_loss = cfgs_loss.get("offset_context_loss", False)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (
        is_anneal and resume_preempt
    )
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    use_radamw = cfgs_opt.get("use_radamw", False)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    loss_reg_std_mult = cfgs_opt.get("loss_reg_std_mult", None)
    loss_reg_num_tracking_steps = cfgs_opt.get("loss_reg_num_tracking_steps", 300)
    loss_reg_min_epoch = cfgs_opt.get("loss_reg_min_epoch", 50)
    if loss_reg_std_mult is not None:
        logger.info("Loss regulation activated")
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    data_world_size, data_rank = world_size, rank
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    img_world_size = 0

    # make adjustments to batch size for image data
    model_fpcs = dataset_fpcs
    model_cfgs_mask = cfgs_mask
    model_tubelet_size = tubelet_size
    if cfgs_img_data is not None:
        img_world_size = int(world_size * img_rank_ratio)
        num_video_ranks = world_size - img_world_size
        img_total_batch_size = img_dataset_batch_size * world_size
        video_total_batch_size = batch_size * world_size

        if img_total_batch_size % img_world_size != 0:
            raise ValueError(
                f"img_total_batch_size ({img_total_batch_size}) must be divisible by num_img_ranks ({img_world_size})"
            )
        if video_total_batch_size % num_video_ranks != 0:
            raise ValueError(
                f"video_total_batch_size ({video_total_batch_size}) must be divisible by num_video_ranks ({num_video_ranks})"
            )

        # img_dataset_batch_size = img_total_batch_size // img_world_size
        batch_size = video_total_batch_size // num_video_ranks

        if rank < int(world_size * img_rank_ratio):
            crop_size = cfgs_img_data.get("crop_size", 512)
            grid_size = crop_size // patch_size

        if rank < int(world_size * img_rank_ratio):
            logger.info(
                f"On rank {rank}, updating dataset with dataset type {img_dataset_type}"
            )
            if img_temporal_dim_size is not None:
                if img_dataset_fpcs[0] != 1:
                    raise NotImplementedError(
                        "Image loader only supports 1 frame per clip with img_temporal_dim_size=1"
                    )
                tubelet_size = 1
            else:
                tubelet_size = tubelet_size

            dataset_type = img_dataset_type
            dataset_paths = img_dataset_paths
            datasets_weights = img_dataset_weights
            dataset_fpcs = img_dataset_fpcs
            batch_size = img_dataset_batch_size
            num_workers = img_num_workers
            if img_mask is not None:
                logger.info("Using image mask")
                cfgs_mask = img_mask

            data_rank = rank
            data_world_size = img_world_size
            lambda_value = lambda_value_img  # We select a different lambda value depending on video vs. image
        else:
            data_rank = rank - img_world_size
            data_world_size = world_size - img_world_size
            lambda_value = lambda_value_vid  # We select a different lambda value depending on video vs. image

        logger.info(
            f"For rank {rank} with world size {world_size}, "
            f"we have total image batch size {img_total_batch_size}, total video batch size {video_total_batch_size}, "
            f"image ranks: {img_world_size}, video ranks: {num_video_ranks}, "
            f"using the following params: "
            f"dataset_type: {dataset_type}, "
            f"dataset_paths: {dataset_paths}, "
            f"datasets_weights: {datasets_weights}, "
            f"dataset_fpcs: {dataset_fpcs}, "
            f"batch_size: {batch_size}, "
            f"num_workers: {num_workers}, "
            f"data_rank: {data_rank}, "
            f"data_world_size: {data_world_size}"
            f"lambda_value for the context loss: {lambda_value}"
        )
    else:
        lambda_value = lambda_value_vid

    # -- set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # Aurora / Intel Max: ZE_AFFINITY_MASK was set by the launcher before
        # torch import, so torch.xpu only sees one tile (index 0).
        device = torch.device("xpu:0")
        torch.xpu.set_device(device)
    else:
        device = torch.device("cpu")

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)

    load_path = None
    if load_model:
        if is_anneal:
            if os.path.exists(latest_path) and resume_anneal:
                load_path = latest_path
            else:
                load_path = anneal_ckpt
                resume_anneal = False
        else:
            # Resume precedence: a latest.pth.tar in the run folder is an
            # in-progress resume and must WIN over read_checkpoint (r_file).
            # r_file is only a BOOTSTRAP (e.g. resume-from-another-run's e19);
            # if we keep preferring it, every chained slice reloads the bootstrap
            # and re-does the same epoch forever (observed: cresume stuck at ep21).
            if os.path.exists(latest_path):
                load_path = latest_path
            else:
                load_path = r_file if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        ("%.2f", "fwd-target-ms"),
        ("%.2f", "fwd-context-ms"),
        ("%.2f", "backward-ms"),
        ("%.2f", "opt-step-ms"),
        ("%.2f", "ema-ms"),
        ("%.5f", "loss-pred"),
        ("%.5f", "loss-context"),
        ("%.5f", "lambda"),
        # L0 free-memory probe (torchtune MEMPROBE): external CCL/OFI growth is
        # INVISIBLE to torch's reserved/allocated — it lives in the L0 driver's
        # free pool. external = l0_used - torch_alloc. If l0-free creeps DOWN as
        # the backward floor creeps UP -> inter-node CCL/OFI registration
        # accumulation (the static-buffer/registration fix path). If l0-free is
        # FLAT while backward spikes cohort-wide -> pure fabric contention
        # (grad_accum is then the legitimate mitigation, not a band-aid).
        ("%.1f", "l0-free-mib"),
        ("%.1f", "l0-ext-mib"),
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(model_cfgs_mask) * len(model_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=model_tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        n_registers_predictor=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    target_encoder = copy.deepcopy(encoder)

    # Authoritative encoder embed_dim from the built model (single source of truth; supports any ladder
    # size, not just the 3 hardcoded above). forward_target/forward_context use this as their default.
    embed_dim_encoder = encoder.backbone.embed_dim

    # -- scaling-law sidecar: if the config carries a `scaling:` stamp (written by
    # scaling/gen_configs.py), dump run identity + measured param counts to a rank-0
    # JSON next to the loss CSV. The collector (scaling/collect.py) joins this with
    # log_r0.csv; keeping it a static sidecar means ZERO edits to the training hot path.
    if rank == 0 and args.get("scaling") is not None:
        try:
            _sc = dict(args.get("scaling"))
            _sc["n_params_encoder"] = sum(p.numel() for p in encoder.parameters())
            _sc["n_params_predictor"] = sum(p.numel() for p in predictor.parameters())
            _sc["n_params_measured"] = _sc["n_params_encoder"] + _sc["n_params_predictor"]
            _sc["model_name"] = model_name
            _sc["world_size"] = world_size
            with open(os.path.join(folder, "scaling.json"), "w") as _f:
                json.dump(_sc, _f, indent=2)
        except Exception as _e:  # never let logging break training
            logger.warning(f"[scaling] sidecar write failed: {_e}")

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        # clip_len=clip_len,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=data_rank,
        world_size=data_world_size,
        datasets_weights=datasets_weights,
        sampling_temperature=sampling_temperature,
        min_clip_std=min_clip_std,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        try:
            _dlen = unsupervised_loader.num_batches
        except Exception:
            _dlen = -1
    if ipe is None:
        ipe = _dlen
    logger.info(f"Using batch size of {batch_size}, fpcs of {dataset_fpcs}")
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # zizi

    # -- distributed wrap + optimizer.
    # Two strategies, selected by VJEPA_DIST_STRATEGY (default "ddp" = unchanged):
    #   ddp  : build the optimizer on the RAW modules, THEN DDP-wrap (the original
    #          order; DDP keeps the same Parameter objects so this is equivalent
    #          and preserved byte-for-byte).
    #   hsdp : FSDP1 HYBRID_SHARD wrap FIRST, then build the optimizer on the
    #          wrapped modules. With use_orig_params=True the optimizer must see
    #          the FSDP-managed params, so ordering is reversed. See app/vjepa_2_1/
    #          hsdp.py for why (2B L0-headroom starvation under DDP on Aurora).

    def _make_opt():
        return init_opt(
            is_anneal=is_anneal,
            encoder=encoder,
            predictor=predictor,
            use_radamw=use_radamw,
            wd=wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs,
            ipe_scale=ipe_scale,
            mixed_precision=mixed_precision,
            dtype=dtype,
            device=device,
            betas=betas,
            eps=eps,
        )

    # HSDP loads model weights into the RAW (unwrapped) modules FIRST, then wraps.
    # Loading AFTER an FSDP wrap requires a FULL_STATE_DICT gather; with
    # rank0_only=False that all-gathers the full 22GB to every one of 192 ranks
    # with a redundant CPU copy per rank (torch warns of exactly this) — it wedged
    # a 16n run for 40min with 0 iters. Loading pre-wrap: every rank reads the .pt
    # from Lustre independently (as the DDP path effectively does), then FSDP
    # shards the already-loaded params. No cross-rank gather. `start_epoch` is
    # captured here for HSDP and the post-wrap load blocks below are skipped.
    start_epoch = 0
    _hsdp_loaded = False
    if dist_strategy == "hsdp":
        from app.vjepa_2_1.hsdp import build_hsdp_mesh, wrap_hsdp

        # 1) bootstrap from a pretrained checkpoint (e.g. Meta ViT-G e40)
        if p_file:
            encoder, predictor, target_encoder = load_pretrained(
                r_path=p_file,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                context_encoder_key=context_encoder_key,
                target_encoder_key=target_encoder_key,
                load_predictor=load_predictor,
                load_encoder=True,
            )
        # 2) resume model weights + epoch from an in-progress run (optimizer state
        #    is not restored under HSDP — opt=None; it is reinitialized below).
        if load_model or os.path.exists(latest_path):
            print("Loadind checkpoint from: ", load_path)
            (encoder, predictor, target_encoder, _, _, start_epoch) = load_checkpoint(
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=None,
                scaler=None,
                is_anneal=is_anneal and not resume_anneal,
            )
        _hsdp_loaded = True

        _hsdp_mesh, _hsdp_nodes, _hsdp_lws = build_hsdp_mesh(
            world_size, device_type=device.type, logger=logger
        )
        # target_encoder MUST use the SAME mesh + policy so EMA _foreach ops act
        # on aligned shards (verified by tests/test_hsdp_ema.py).
        encoder = wrap_hsdp(
            encoder, _hsdp_mesh, requires_grad=True, logger=logger
        )
        predictor = wrap_hsdp(
            predictor, _hsdp_mesh, requires_grad=True, logger=logger
        )
        target_encoder = wrap_hsdp(
            target_encoder, _hsdp_mesh, requires_grad=False, logger=logger
        )
        # Build optimizer AFTER wrapping (use_orig_params=True).
        optimizer, scaler, scheduler, wd_scheduler = _make_opt()
    else:
        # -- init optimizer and scheduler (on raw modules, as upstream) --
        optimizer, scaler, scheduler, wd_scheduler = _make_opt()
        # Allow tuning DDP gradient bucket size via env. Default 25MB; larger
        # buckets reduce collective count, helpful on Aurora xccl where backward
        # appears to be AllReduce-dominated.
        _bucket_mb = int(os.environ.get("VJEPA_DDP_BUCKET_MB", "25"))
        encoder = DistributedDataParallel(
            encoder, static_graph=True, bucket_cap_mb=_bucket_mb,
        )
        # CLAUDE.md flagged predictor static_graph=True as crashing on Polaris.
        # On Aurora this may behave differently — VJEPA_PRED_STATIC=1 opts in.
        _pred_static = os.environ.get("VJEPA_PRED_STATIC") == "1"
        predictor = DistributedDataParallel(
            predictor,
            static_graph=_pred_static,
            find_unused_parameters=not _pred_static,
            bucket_cap_mb=_bucket_mb,
        )
        # Optional bf16 gradient-compression comm hook (VJEPA_BF16_COMM=1). Casts
        # gradients to bf16 before the AllReduce and back to fp32 after — halves the
        # collective payload, which is the dominant cost in the AllReduce-bound
        # backward on Aurora xccl (esp. for ViT-g's ~3.3x params). Off by default so
        # baseline runs are bit-for-bit unchanged; enable only after a loss-curve
        # sanity check. Params are already bf16-autocast in compute, so the extra
        # precision loss is on the reduced gradients only.
        if os.environ.get("VJEPA_BF16_COMM") == "1":
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default_hooks as _ddp_hooks,
            )
            _pg = None  # default process group
            encoder.register_comm_hook(_pg, _ddp_hooks.bf16_compress_hook)
            predictor.register_comm_hook(_pg, _ddp_hooks.bf16_compress_hook)
            logger.info("DDP bf16_compress_hook registered (encoder+predictor)")
        target_encoder = DistributedDataParallel(target_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

    # DDP path: bootstrap from pretrained checkpoint (HSDP already loaded pre-wrap).
    if p_file and not _hsdp_loaded:
        encoder, predictor, target_encoder = load_pretrained(
            r_path=p_file,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            context_encoder_key=context_encoder_key,
            target_encoder_key=target_encoder_key,
            load_predictor=load_predictor,
            load_encoder=True,
        )

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs) + 1)
    )
    # Context-loss (lambda) warmup-hold schedule. Defaults match Meta's published
    # ~300k-iter run (ramp iters 15k->30k); for short CPT runs those bounds never
    # fire, so allow the YAML to override them (model.lambda_start_iter /
    # model.lambda_end_iter) to keep the same proportional ramp on fewer iters.
    _lambda_start = int(cfgs_model.get("lambda_start_iter", 15_000))
    _lambda_end = int(cfgs_model.get("lambda_end_iter", 30_000))
    lambda_sched = Lambda_LinearWarmupHold(
        lambda_value=lambda_value, start_iter=_lambda_start, end_iter=_lambda_end
    )
    logger.info(
        f"Lambda_LinearWarmupHold: value={lambda_value} ramp "
        f"[{_lambda_start}, {_lambda_end}] iters (progressive)"
    )

    # -- load training checkpoint (DDP path; HSDP already resumed pre-wrap above)
    if not _hsdp_loaded and (load_model or os.path.exists(latest_path)):
        print("Loadind checkpoint from: ", load_path)
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
        )
    if load_model or os.path.exists(latest_path):
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def _atomic_torch_save(obj, path):
        # Write to a temp file in the SAME directory, then os.replace() (atomic
        # rename on POSIX). Guarantees `path` is never a truncated half-write when
        # walltime SIGTERM lands mid-save. Without this, the chain resumes from a
        # corrupt latest.pth.tar ("failed finding central directory") and stalls.
        # Larger cells (bigger ckpt = longer write) hit the kill window more often.
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_checkpoint(epoch, path):
        if dist_strategy == "hsdp":
            # FULL_STATE_DICT is a COLLECTIVE: every rank must enter the context
            # and call .state_dict() (rank0 receives the full dict, others empty).
            # Only rank0 writes. Model + EMA target round-trip fully and remain
            # DDP-compatible / topology-independent. Optimizer (Adam m/v) is NOT
            # saved on the HSDP path — the shared optimizer spans two FSDP roots
            # so a correct sharded optim_state_dict is deferred; resume reinits
            # the optimizer (load_checkpoint already tolerates this). Acceptable
            # because HSDP has no weight-sync -> no external-memory growth ->
            # runs complete without frequent restarts. KNOWN LIMITATION.
            from app.vjepa_2_1.hsdp import full_state_dict_context

            with full_state_dict_context(encoder):
                enc_sd = encoder.state_dict()
            with full_state_dict_context(predictor):
                pred_sd = predictor.state_dict()
            with full_state_dict_context(target_encoder):
                tgt_sd = target_encoder.state_dict()
            if rank != 0:
                return
            save_dict = {
                "encoder": enc_sd,
                "predictor": pred_sd,
                "opt": None,  # see note above
                "scaler": None if scaler is None else scaler.state_dict(),
                "target_encoder": tgt_sd,
                "epoch": epoch,
                "loss": loss_meter.avg,
                "batch_size": batch_size,
                "world_size": world_size,
                "lr": lr,
                "dist_strategy": "hsdp",
            }
            try:
                _atomic_torch_save(save_dict, path)
            except Exception as e:
                logger.info(f"Encountered exception when saving checkpoint: {e}")
            return

        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            _atomic_torch_save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- validate grad accumulation config (fail loud, never silently wrong)
    if grad_accum > 1:
        if batch_size % grad_accum != 0:
            raise ValueError(
                f"VJEPA_GRAD_ACCUM={grad_accum} must divide per-rank batch_size="
                f"{batch_size}"
            )
        if loss_reg_std_mult is not None:
            raise ValueError(
                "VJEPA_GRAD_ACCUM>1 is not supported together with loss "
                "regulation (loss_reg_std_mult); the per-step skip logic is not "
                "threaded through microbatches. Disable one."
            )
        logger.info(
            f"Gradient accumulation ON: grad_accum={grad_accum}, "
            f"per-rank batch={batch_size} -> micro-batch={batch_size // grad_accum}"
        )

    # -- validate TRUE accumulation config (fail loud) --
    if true_accum > 1:
        if grad_accum > 1:
            raise ValueError(
                "VJEPA_TRUE_ACCUM>1 and VJEPA_GRAD_ACCUM>1 are mutually exclusive "
                "(one accumulates across loader batches, the other slices one batch). "
                "Set exactly one."
            )
        if loss_reg_std_mult is not None:
            raise ValueError(
                "VJEPA_TRUE_ACCUM>1 is not supported together with loss regulation "
                "(loss_reg_std_mult); the per-step skip logic is not threaded through "
                "accumulation. Disable one."
            )
        logger.info(
            f"TRUE gradient accumulation ON: true_accum={true_accum}, "
            f"per-rank batch={batch_size} -> effective per-rank batch="
            f"{batch_size * true_accum}; ONE inter-node collective per {true_accum} "
            f"backwards (no_sync on non-final). LR unchanged."
        )

    trailing_losses = []
    step_count = 0

    # -- Per-rank HANG WATCHDOG (diagnostic; env-gated, default OFF so the proven
    # recipe is untouched unless VJEPA_ITER_WATCHDOG_S is set). Ported/upgraded from
    # PRISM's step-watchdog (BaseMM_PRISM/src/training/trainer_native.py): each rank
    # arms a faulthandler timer at the top of every iter; if the iter exceeds the
    # timeout the offending rank DUMPS ITS OWN PYTHON STACK to stderr (showing the
    # exact line/collective it is blocked on) — the attribution we lacked when we
    # blindly pkill'd. Rank+host prefix lets us cross-map to a bad node. Motivation:
    # our 16n Mode-B hangs were killed blind with ZERO forensics; AGPT/torchtitan and
    # PRISM both instrument the stuck rank rather than guessing. This is the missing
    # evidence-collection, NOT a fix. See memory vitG-2b-allreduce-spikes.
    import faulthandler as _fh
    import signal as _sig
    _iter_watchdog_s = float(os.environ.get("VJEPA_ITER_WATCHDOG_S", "0"))
    _watchdog_on = _iter_watchdog_s > 0
    _dump_fh = None  # per-rank stack dump file; set below when watchdog is on
    if _watchdog_on:
        # Register SIGUSR1 -> dump ALL-THREAD stacks of THIS rank AND CONTINUE (does not
        # abort — verified). The shell watchdog (capacity.sh capture_hang_forensics) sends
        # SIGUSR1 to every rank on a true hang, so each of the 192 ranks prints exactly where
        # it is blocked (which collective/line) to stderr -> job log, then the shell pkill -9
        # does the actual kill. (SIGABRT cannot be faulthandler-registered on this build:
        # "signal 6 cannot be registered" — verified; SIGUSR1 is the correct trigger.)
        # Per-rank dump FILE. faulthandler writes raw stacks with NO rank/host prefix, so
        # when all 192 ranks dump into the shared job stdout they are un-attributable — we
        # cannot tell WHICH node the stalled ranks are on (the exact question ALCF needs to
        # answer whether the deadlock is a same-PG order divergence or a cross-PG straggler
        # cascade). Give each rank its own file so `grep -l _pre_forward_unshard` maps the
        # stalled ranks to hosts. Kept in hang_diag/ alongside the nodefile the shell captures.
        _hang_dir = os.path.join(folder, "hang_diag") if folder else None
        if _hang_dir:
            try:
                os.makedirs(_hang_dir, exist_ok=True)
                _dump_path = os.path.join(
                    _hang_dir, f"stack_rank{rank:04d}_{socket.gethostname()}.txt"
                )
                _dump_fh = open(_dump_path, "a", buffering=1)  # line-buffered, append
            except Exception as _e:
                logger.warning(f"[hang-watchdog] per-rank dump file open failed: {_e}")
        try:
            _fh.enable(all_threads=True)
            # SIGUSR1 dump goes to the per-rank file if we have one, else stderr.
            _fh.register(_sig.SIGUSR1, file=_dump_fh, all_threads=True, chain=False)
        except Exception as _e:  # never let diag setup break training
            logger.warning(f"[hang-watchdog] faulthandler.register(SIGUSR1) failed: {_e}")
        # dump_traceback_later prints ALL threads' stacks after the timeout unless
        # cancelled first; repeat=False so a single dump per arm. We re-arm each iter.
        logger.info(
            f"[hang-watchdog] per-rank iter watchdog ON: {_iter_watchdog_s}s "
            f"(rank={rank} host={socket.gethostname()}); dumps stack if an iter stalls, "
            f"and on SIGUSR1 from the shell watchdog."
            + (f" per-rank dumps -> {_hang_dir}/stack_rank*.txt" if _dump_fh else "")
        )

    def _watchdog_arm():
        if _watchdog_on:
            if _dump_fh is not None:
                _fh.dump_traceback_later(
                    _iter_watchdog_s, repeat=False, exit=False, file=_dump_fh
                )
            else:
                # file omitted -> faulthandler defaults to sys.stderr (-> job stdout)
                _fh.dump_traceback_later(_iter_watchdog_s, repeat=False, exit=False)

    def _watchdog_disarm():
        if _watchdog_on:
            _fh.cancel_dump_traceback_later()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        def fetch_sample():
            # One loader batch with the upstream StopIteration-refresh + retry logic.
            # Factored out of the loop so TRUE accumulation can pull N batches per step.
            nonlocal loader
            iter_retries = 0
            while True:
                try:
                    return next(loader)
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    if "airstore" in dataset_type.lower():
                        unsupervised_sampler.increase_epoch()
                    else:
                        unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(
                            f"Encountered exception when loading data (num retries {iter_retries}):\n{e}"
                        )
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        raise RuntimeError(
                            f"Exceeded max retries ({NUM_RETRIES}) when loading data."
                        ) from e

        for itr in range(ipe):
            itr_start_time = time.time()
            _watchdog_arm()  # per-rank hang stack-dump (no-op unless VJEPA_ITER_WATCHDOG_S set)

            sample = fetch_sample()

            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            def load_clips(_sample=None):
                _sample = sample if _sample is None else _sample
                all_clips, all_masks_enc, all_masks_pred = [], [], []
                for fpc_sample in _sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    all_clips += [udata[0][0].to(device, non_blocking=True)]
                    all_masks_enc += [
                        [m.to(device, non_blocking=True) for m in masks_enc]
                    ]
                    all_masks_pred += [
                        [m.to(device, non_blocking=True) for m in masks_pred]
                    ]
                return all_clips, all_masks_enc, all_masks_pred

            clips, masks_enc, masks_pred = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            phase_timer = PhaseTimer(enabled=True)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                phase_timer.mark("start")

                def forward_target(c, embed_dim=embed_dim_encoder):
                    with torch.no_grad():
                        h = target_encoder(c, gram_mode=False, training_mode=True)
                        new_h = []
                        for hi in h:
                            if levels_predictor > 1:
                                hi_0 = F.layer_norm(hi[:, :, :embed_dim], (embed_dim,))
                                hi_1 = F.layer_norm(
                                    hi[:, :, embed_dim : embed_dim * 2],
                                    (embed_dim,),
                                )
                                hi_2 = F.layer_norm(
                                    hi[:, :, embed_dim * 2 : embed_dim * 3],
                                    (embed_dim,),
                                )
                                hi_3 = F.layer_norm(hi[:, :, -embed_dim:], (embed_dim,))
                                hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=2)
                                new_h.append(hi_norm)
                            else:
                                new_h.append(F.layer_norm(hi, (hi.size(-1),)))
                        return new_h

                def forward_context(clips, masks_enc, masks_pred, embed_dim=embed_dim_encoder):
                    modality = "video"
                    if img_temporal_dim_size is not None:
                        if clips[0].shape[2] == img_temporal_dim_size:
                            modality = "image"
                    z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
                    z_pred, z_context = predictor(
                        z, masks_enc, masks_pred, mod=modality
                    )
                    if normalize_predictor:
                        z_pred = normalize_nested(z_pred, embed_dim)

                        if predict_all:
                            z_context = normalize_nested(z_context, embed_dim)
                    return z_pred, z_context

                def loss_fn(z, h, masks_to_apply, cls_loss, d_weights):
                    if cls_loss:
                        h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
                        h = [
                            apply_masks(hi[:, 1:], mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]
                        loss, n = 0, 0
                        for zi, hi, hi_cls in zip(z, h, h_cls):
                            for zij, hij in zip(zi, hi):
                                h_term = torch.cat([hi_cls, hij], dim=1)
                                loss += (
                                    torch.mean(torch.abs(zij - h_term) ** loss_exp)
                                    / loss_exp
                                )
                                n += 1

                        loss /= n
                        return loss
                    else:
                        h = [
                            apply_masks(hi, mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]

                        if d_weights is not None:
                            loss, n = 0, 0
                            for zi, hi, d_i in zip(z, h, d_weights):
                                for zij, hij, d_ij in zip(zi, hi, d_i):
                                    # clamp_min(1.0): d_ij is a grid distance; the smallest
                                    # NONZERO value is 1.0 (adjacent cell, offset_context_loss
                                    # False). d_ij CAN be 0 when an enc token shares a grid
                                    # (d,h,w) with a pred token (masks not disjoint) -> 1/0 = inf
                                    # -> NaN. Flooring at 1.0 treats a coincident token like a
                                    # nearest-neighbor (weight 1.0) and is a NO-OP for all
                                    # legitimate d_ij >= 1.0. Secondary backstop to the loader's
                                    # non-finite gate. NOTE: assumes offset_context_loss=False
                                    # (our config); if that's enabled, revisit the floor value.
                                    loss_n = torch.abs(zij - hij) ** loss_exp * (
                                        1 / d_ij.unsqueeze(2).clamp_min(1.0)
                                    )
                                    loss += torch.mean(loss_n) / loss_exp
                                    n += 1
                            loss /= n
                            return loss
                        else:
                            loss, n = 0, 0
                            for zi, hi in zip(z, h):
                                for zij, hij in zip(zi, hi):
                                    loss += (
                                        torch.mean(torch.abs(zij - hij) ** loss_exp)
                                        / loss_exp
                                    )
                                    n += 1
                            loss /= n
                            return loss

                # Step 1. Forward for one (micro)batch — device-agnostic autocast
                # so XPU/CUDA both take the bf16 path. torch.cuda.amp.autocast
                # silently disables autocast when CUDA is absent (measured on
                # Aurora), falling back to fp32 and tanking throughput.
                def _forward_losses(clips_mb, menc_mb, mpred_mb):
                    with torch.amp.autocast(device_type=device.type, dtype=dtype,
                                            enabled=mixed_precision):
                        h = forward_target(clips_mb)
                        z_pred, z_context = forward_context(
                            clips_mb, menc_mb, mpred_mb
                        )
                        loss_pred = loss_fn(
                            z_pred, h, mpred_mb, cls_loss=has_cls_first, d_weights=None
                        )
                        loss = loss_pred
                        loss_context = torch.zeros((), device=loss_pred.device)
                        lambda_value_step = 0.0
                        if predict_all:
                            distance_weights = compute_mask_distance(
                                mpred_mb, menc_mb, grid_size, offset_context_loss
                            )
                            d_weights = (
                                distance_weights if weight_distance_loss else None
                            )
                            loss_context = loss_fn(
                                z_context, h, menc_mb, cls_loss=False,
                                d_weights=d_weights,
                            )
                            if lambda_progressive:
                                lambda_value_step = lambda_sched.value(
                                    epoch * ipe + itr
                                )
                            else:
                                lambda_value_step = lambda_value
                            loss = loss + loss_context * lambda_value_step
                    return loss, loss_pred, loss_context, lambda_value_step

                # Step 2. Backward & step.
                run_step = True
                if true_accum > 1:
                    # -- TRUE accumulation path (VJEPA_TRUE_ACCUM>1) --
                    # Process `true_accum` SEPARATE loader batches; defer the inter-node
                    # collective (no_sync on encoder+predictor) to the LAST one, so there
                    # is ONE ReduceScatter/AllReduce per optimizer step instead of
                    # true_accum. This is the real fabric lever (fewer collectives =
                    # fewer §4g host-side stall opportunities). Each sub-batch's loss is
                    # /true_accum so the summed gradient is the mean over the enlarged
                    # effective batch. Batch 0 is the already-loaded `clips`; batches
                    # 1..N-1 are fetched here. no_sync ON is affordable (2B bf16 grad
                    # ~4GB << ~15GB free L0); DIVERGES from PRISM 7B (memory-bound).
                    def _accum_no_sync():
                        es = contextlib.ExitStack()
                        es.enter_context(encoder.no_sync())
                        es.enter_context(predictor.no_sync())
                        return es

                    loss_sum = loss_pred_sum = loss_context_sum = 0.0
                    lambda_value_step = 0.0
                    for j in range(true_accum):
                        if j == 0:
                            c_j, me_j, mp_j = clips, masks_enc, masks_pred
                        else:
                            c_j, me_j, mp_j = load_clips(fetch_sample())
                        l, lp, lc, lvs = _forward_losses(c_j, me_j, mp_j)
                        lambda_value_step = lvs
                        if j == 0:
                            phase_timer.mark("fwd_target_done")
                            phase_timer.mark("fwd_context_done")
                        l = l / true_accum
                        is_last = j == true_accum - 1
                        sync_ctx = (
                            contextlib.nullcontext() if is_last else _accum_no_sync()
                        )
                        with sync_ctx:
                            if scaler is not None:
                                scaler.scale(l).backward()
                            else:
                                l.backward()
                        loss_sum += float(l) * true_accum  # undo /true_accum for report
                        loss_pred_sum += float(lp)
                        loss_context_sum += float(lc)
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    phase_timer.mark("backward_done")
                    loss = loss_sum / true_accum
                    loss_pred = loss_pred_sum / true_accum
                    loss_context = loss_context_sum / true_accum
                elif grad_accum <= 1:
                    # -- single-batch path (default; behavior unchanged) --
                    loss, loss_pred, loss_context, lambda_value_step = _forward_losses(
                        clips, masks_enc, masks_pred
                    )
                    phase_timer.mark("fwd_target_done")
                    phase_timer.mark("fwd_context_done")
                    if loss_reg_std_mult is not None:
                        meanval = np.mean(trailing_losses)
                        stdval = np.std(trailing_losses)
                        max_bound = meanval + loss_reg_std_mult * stdval
                        if (
                            loss > max_bound
                            and epoch > loss_reg_min_epoch
                            and len(trailing_losses)
                            > int(0.5 * loss_reg_num_tracking_steps)
                        ):
                            run_step = False
                            loss.backward()
                            logger.info(
                                f"Loss {loss} is above bound {meanval} + {loss_reg_std_mult} * {stdval}. Skipping step."
                            )
                    if run_step:
                        # Branch on the scaler's presence, not on mixed_precision:
                        # scaler is None for bf16 (no loss scaling) and non-None
                        # only for fp16. Using `mixed_precision` would call
                        # scaler.scale() on None under bf16.
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                    phase_timer.mark("backward_done")
                    loss = float(loss)
                    loss_pred = float(loss_pred)
                    loss_context = float(loss_context)
                else:
                    # -- gradient-accumulation path (VJEPA_GRAD_ACCUM>1) --
                    # Slice each fpc slot's batch dim into grad_accum microbatches;
                    # defer DDP gradient sync (no_sync) until the final microbatch
                    # so the optimizer sees the full-batch gradient in ONE allreduce.
                    # Loss is divided by grad_accum so the summed gradient equals the
                    # full-batch mean gradient (bit-equivalent to bs=batch_size up to
                    # microbatch-boundary numerics). Timing note: fwd-* marks reflect
                    # microbatch 0; backward-ms covers all microbatches' fwd+bwd.
                    def _accum_no_sync():
                        es = contextlib.ExitStack()
                        es.enter_context(encoder.no_sync())
                        es.enter_context(predictor.no_sync())
                        return es

                    mb = batch_size // grad_accum
                    loss_sum = loss_pred_sum = loss_context_sum = 0.0
                    lambda_value_step = 0.0
                    for j in range(grad_accum):
                        sl = slice(j * mb, (j + 1) * mb)
                        clips_mb = [c[sl] for c in clips]
                        menc_mb = [[m[sl] for m in mm] for mm in masks_enc]
                        mpred_mb = [[m[sl] for m in mm] for mm in masks_pred]
                        l, lp, lc, lvs = _forward_losses(clips_mb, menc_mb, mpred_mb)
                        lambda_value_step = lvs
                        if j == 0:
                            phase_timer.mark("fwd_target_done")
                            phase_timer.mark("fwd_context_done")
                        l = l / grad_accum
                        is_last = j == grad_accum - 1
                        sync_ctx = (
                            contextlib.nullcontext() if is_last else _accum_no_sync()
                        )
                        with sync_ctx:
                            l.backward()
                        loss_sum += float(l) * grad_accum  # undo /grad_accum for report
                        loss_pred_sum += float(lp)
                        loss_context_sum += float(lc)
                    phase_timer.mark("backward_done")
                    loss = loss_sum / grad_accum
                    loss_pred = loss_pred_sum / grad_accum
                    loss_context = loss_context_sum / grad_accum

                # -- Symmetric non-finite guard (SURVIVABILITY, not a root-cause fix).
                # A rare bf16 transient can make ONE rank's loss non-finite (observed ~1/400
                # steps, always a distinct rank/host = stochastic). The old
                # `assert not np.isnan(loss)` hard-killed the whole job. We skip the opt step
                # on ALL ranks together instead (discard grads via zero_grad, EMA/schedule
                # advance, continue) — no crash, no collective mismatch. NO grad clipping
                # (Meta's recipe uses none; bounded L1/L2-to-EMA loss).
                #
                # PIGGYBACK DESIGN (2026-07-05): we do NOT add a per-iter all_reduce for this
                # (that was train.py:1212 in the old version — one hang blocked THERE; on a
                # fabric-deadlock-bound run every extra inter-node collective is another
                # deadlock surface, see mode-b-hang-investigation). Instead we exploit a
                # collective that ALREADY ran: after backward, FSDP does ReduceScatter+AllReduce
                # over the gradients, which SUMS every rank's contribution into every rank's
                # grad shard. So a non-finite loss on ANY rank → its whole grad is non-finite →
                # after the reduce, EVERY rank's local .grad shard is non-finite. A purely LOCAL
                # torch.isfinite scan of our own grads is therefore SYMMETRIC across all 192
                # ranks (validated: reduce-scatter propagates NaN to all shards) with ZERO added
                # collectives. Belt-and-suspenders: also OR in our own loss check (a NaN we
                # produced locally is known without looking at grads). See nan-crash-rootcause.
                local_bad = (not np.isfinite(loss))
                if not local_bad:
                    # Cheap single-scalar scan: sum each grad to a scalar (NaN/Inf propagates
                    # through sum), stack, ONE isfinite — one device sync instead of ~850
                    # per-tensor .all() calls. Grads are already reduced across ranks here, so
                    # this is symmetric globally.
                    with torch.no_grad():
                        _gsums = [
                            _p.grad.sum()
                            for _p in (list(encoder.parameters()) + list(predictor.parameters()))
                            if _p.grad is not None
                        ]
                        if _gsums:
                            local_bad = not bool(torch.isfinite(torch.stack(_gsums)).all())
                all_finite = not local_bad  # symmetric: NaN grad reduced onto every rank
                if not all_finite:
                    run_step = False
                    # Every rank sees it now (grad is globally NaN), so gate the log to rank 0
                    # + the loss-origin rank to keep it to a couple lines, not 192.
                    if not np.isfinite(loss) or rank == 0:
                        logger.warning(
                            "NON-FINITE guard FIRED epoch=%d itr=%d rank=%d host=%s "
                            "local_loss_finite=%s — SKIPPING optimizer step (grads discarded, "
                            "EMA/schedule advance). Survivability guard; root-cause TBD. "
                            "loss=%s loss_pred=%s loss_ctx=%s"
                            % (epoch + 1, itr, rank, socket.gethostname(),
                               np.isfinite(loss), loss, loss_pred, loss_context)
                        )

                if run_step:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()
                phase_timer.mark("opt_step_done")

                # Step 3. momentum update of target encoder
                m = min(next(momentum_scheduler), ema[1])
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(
                        encoder.parameters(), target_encoder.parameters()
                    ):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)
                phase_timer.mark("ema_done")

                return (
                    float(loss),
                    float(loss_pred),
                    float(loss_context),
                    float(lambda_value_step),
                    _new_lr,
                    _new_wd,
                    run_step,
                )

            (
                loss,
                loss_pred_val,
                loss_context_val,
                lambda_value_step_val,
                _new_lr,
                _new_wd,
                run_step,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            phase_times = phase_timer.to_dict()
            phase_fwd_target = phase_times.get("start->fwd_target_done", 0.0)
            phase_fwd_context = phase_times.get("fwd_target_done->fwd_context_done", 0.0)
            phase_backward = phase_times.get("fwd_context_done->backward_done", 0.0)
            phase_opt_step = phase_times.get("backward_done->opt_step_done", 0.0)
            phase_ema = phase_times.get("opt_step_done->ema_done", 0.0)
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            if loss_reg_std_mult is not None:
                if run_step:
                    trailing_losses.append(loss)
                    if len(trailing_losses) > loss_reg_num_tracking_steps:
                        trailing_losses = trailing_losses[1:]
                else:
                    step_count += 1
                    if step_count > MAX_REPEAT_COUNTS:
                        raise RuntimeError(
                            "Loss is above bound for too many tries. Exiting."
                        )

            # -- Logging
            def log_stats():
                # -- L0 free-memory probe (per rank, per iter). external =
                # l0_used - torch_alloc catches CCL/OFI growth reserved can't see.
                l0_free_mib = -1.0
                l0_ext_mib = -1.0
                if device.type == "xpu":
                    try:
                        free_b, total_b = torch.xpu.mem_get_info()
                        l0_free_mib = free_b / 1024.0**2
                        l0_ext_mib = (
                            (total_b - free_b) - torch.xpu.memory_allocated()
                        ) / 1024.0**2
                    except Exception:
                        pass  # NotImplementedError under pluggable allocator, etc.

                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                    phase_fwd_target,
                    phase_fwd_context,
                    phase_backward,
                    phase_opt_step,
                    phase_ema,
                    loss_pred_val,
                    loss_context_val,
                    lambda_value_step_val,
                    l0_free_mib,
                    l0_ext_mib,
                )
                if (
                    (itr % log_freq == 0)
                    or (itr == ipe - 1)
                    or np.isnan(loss)
                    or np.isinf(loss)
                ):
                    logger.info(
                        "[%d, %5d] loss: %.3f (pred=%.3f ctx=%.3f λ=%.3f) "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] [resv: %.2e] "
                        "[l0free: %.0f] [l0ext: %.0f] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            loss_pred_val,
                            loss_context_val,
                            lambda_value_step_val,
                            "["
                            + ", ".join(
                                [
                                    f"{k}: " + "%.1f" % mask_meters[k].avg
                                    for k in mask_meters
                                ]
                            )
                            + "]",
                            _new_wd,
                            _new_lr,
                            (torch.xpu.max_memory_allocated() if device.type == "xpu"
                             else torch.cuda.max_memory_allocated()) / 1024.0**2,
                            # RESERVED (segment pool) — the counter that actually tests
                            # the "allocator touches new segments -> CCL mints new MRs"
                            # hypothesis. max_memory_allocated (above) is BYTES in use and
                            # says nothing about segment count. If resv is flat -> segment-
                            # growth story is FALSE; if it climbs -> allocator implicated.
                            (torch.xpu.memory_reserved() if device.type == "xpu"
                             else torch.cuda.memory_reserved()) / 1024.0**2,
                            l0_free_mib,
                            l0_ext_mib,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            _watchdog_disarm()  # iter completed within the deadline — cancel the stack-dump timer
            # NOTE: the old hard `assert not np.isnan(loss)` is intentionally removed.
            # A non-finite loss is now handled SYMMETRICALLY above (all-reduce MIN finite
            # flag -> every rank skips the step together) instead of crashing the job. The
            # event is logged with rank/host/itr for later root-cause. Re-adding an assert
            # here would reintroduce the fatal single-rank crash the guard exists to prevent.

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_every_file = f"e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
            # On a short-walltime chained slice (debug-scaling), we only ever get
            # ~one epoch per slice. Exiting right after the checkpoint avoids
            # burning the rest of the walltime on a partial next epoch that will
            # be discarded (next slice resumes from this same checkpoint). The
            # chain/watchdog relaunches; the successor resumes from latest.pth.tar.
            # Opt-in via VJEPA_EXIT_AFTER_CKPT=1 so the capacity/long runs (which
            # SHOULD keep going) are unaffected.
            if os.environ.get("VJEPA_EXIT_AFTER_CKPT") == "1" and (epoch + 1) < num_epochs:
                logger.info(
                    f"VJEPA_EXIT_AFTER_CKPT: saved epoch {epoch + 1}, exiting "
                    f"slice cleanly (chain will resume from latest.pth.tar)."
                )
                return
