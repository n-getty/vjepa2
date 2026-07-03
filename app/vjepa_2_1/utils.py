# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1.models.vision_transformer as video_vit
import torch
import torch.nn.functional as F
import yaml
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import (
    CosineWDSchedule,
    LinearDecaySchedule,
    WarmupCosineSchedule,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, val in state_dict.items():
        if key.startswith("module."):
            key = key.removeprefix("module.")
        if key.startswith("backbone."):
            key = key.removeprefix("backbone.")
        normalized[key] = val
    return normalized


def _target_state_dict_prefix(module):
    first_key = next(iter(module.state_dict()), "")
    if first_key.startswith("module.backbone."):
        return "module.backbone."
    if first_key.startswith("backbone."):
        return "backbone."
    if first_key.startswith("module."):
        return "module."
    return ""


def _prepare_state_dict_for_module(module, pretrained_dict):
    prefix = _target_state_dict_prefix(module)
    normalized = _normalize_state_dict_keys(pretrained_dict)
    return {f"{prefix}{k}": v for k, v in normalized.items()}


def _load_pretrained_module(
    module, pretrained_dict, module_name, epoch, min_match_frac=0.5
):
    prepared_dict = _prepare_state_dict_for_module(module, pretrained_dict)
    model_keys = module.state_dict()
    missing, shape_mismatch, matched = [], [], 0
    for key, value in model_keys.items():
        if key not in prepared_dict:
            missing.append(key)
        elif prepared_dict[key].shape != value.shape:
            shape_mismatch.append(key)
            prepared_dict[key] = value  # keep model's init tensor on mismatch
        else:
            matched += 1

    n_total = max(1, len(model_keys))
    match_frac = matched / n_total
    # A wrong key-prefix guess or an unexpected checkpoint layout makes (almost)
    # every key "missing"; load_state_dict(strict=False) then swallows it and
    # the module silently TRAINS FROM INIT while the loss curve still looks
    # plausible (EMA self-distillation from init falls smoothly but wrongly).
    # Fail loud instead of logging at INFO and moving on. This is exactly the
    # silent-corruption class that the SDPA bug taught us to refuse.
    if match_frac < min_match_frac:
        raise RuntimeError(
            f"load_pretrained: only {matched}/{n_total} keys "
            f"({match_frac:.1%}) of '{module_name}' matched the checkpoint "
            f"(epoch {epoch}); {len(missing)} missing, "
            f"{len(shape_mismatch)} shape-mismatched. This almost always means "
            f"a wrong checkpoint key or prefix — the module would train from "
            f"init. First few missing: {missing[:5]}. "
            f"First few checkpoint keys: {list(prepared_dict)[:5]}."
        )

    if missing:
        logger.warning(
            "load_pretrained[%s]: %d/%d keys missing from checkpoint "
            "(kept at init). First few: %s",
            module_name, len(missing), n_total, missing[:5],
        )
    if shape_mismatch:
        logger.warning(
            "load_pretrained[%s]: %d keys shape-mismatched (kept at init): %s",
            module_name, len(shape_mismatch), shape_mismatch[:5],
        )
    msg = module.load_state_dict(prepared_dict, strict=False)
    logger.info(
        "loaded pretrained %s from epoch %s: matched %d/%d keys (%.1f%%); "
        "load_state_dict msg: %s",
        module_name, epoch, matched, n_total, 100.0 * match_frac, msg,
    )


def normalize_and_concat(tensor, embed_dim):
    """Split tensor into 4 chunks of size embed_dim along the last axis,
    apply LayerNorm to each chunk, then concatenate back."""
    chunks = [
        F.layer_norm(tensor[:, :, i * embed_dim : (i + 1) * embed_dim], (embed_dim,))
        for i in range(4)
    ]
    return torch.cat(chunks, dim=2)


def normalize_nested(nested, embed_dim):
    """Apply normalize_and_concat recursively over nested lists."""
    return [
        [[normalize_and_concat(z, embed_dim) for z in inner] for inner in outer]
        for outer in nested
    ]


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    import warnings

    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = (
                _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            )
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn(
                    "Configs for online evals must use same number of nodes for slurm-batch processing"
                )

            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu
            _args["pretrain"]["frames_per_clip"] = num_frames
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    pretrained_dict = checkpoint["encoder"]
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    pretrained_dict = checkpoint["predictor"]
    for k, v in predictor.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    if target_encoder is not None:
        pretrained_dict = checkpoint["target_encoder"]
        for k, v in target_encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(
                    f'key "{k}" is of different shape in model and loaded state dict'
                )
                pretrained_dict[k] = v
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(
            f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}"
        )

    # Skip optimizer restore when either (a) the caller passed opt=None (the HSDP
    # path builds the optimizer AFTER FSDP-wrapping, so load runs with no optimizer
    # object yet — it is reinitialized post-wrap), or (b) the checkpoint has no opt
    # state. Guarding on the local `opt` object too avoids derefing None when
    # resuming a checkpoint that DOES contain opt state on the HSDP path.
    if opt is None:
        print("[warn] opt=None passed to load_checkpoint (HSDP: optimizer reinit post-wrap); skipping opt restore.")
    elif checkpoint.get("opt") is None:
        print("[warn] No optimizer state in checkpoint (HSDP or fresh); keeping current optimizer.")
    else:
        try:
            opt.load_state_dict(checkpoint["opt"])
        except ValueError:
            print("[warn] Optimizer groups mismatch; reinitializing optimizer.")
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def load_pretrained(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=True,
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint.get("epoch", "unknown")

    if load_encoder and encoder is not None:
        _load_pretrained_module(
            encoder,
            checkpoint[context_encoder_key],
            "encoder",
            epoch,
        )

    if load_predictor and predictor is not None and "predictor" in checkpoint:
        _load_pretrained_module(
            predictor,
            checkpoint["predictor"],
            "predictor",
            epoch,
        )

    if load_encoder and target_encoder is not None:
        checkpoint_key = (
            target_encoder_key
            if target_encoder_key in checkpoint
            else context_encoder_key
        )
        _load_pretrained_module(
            target_encoder,
            checkpoint[checkpoint_key],
            "target encoder",
            epoch,
        )

    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    chop_last_n_tokens=0,
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    interpolate_rope=False,
    modality_embedding=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=(
            encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads
        ),
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        chop_last_n_tokens=chop_last_n_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    use_radamw=False,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    dtype=None,
    device=None,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    if use_radamw:
        from src.utils.adamw import AdamW as RAdamW

        logger.info("Using Rescaled-AdamW")
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    # Loss scaling is only meaningful for fp16; bf16 has fp32-range exponent and
    # needs none. The old `torch.cuda.amp.GradScaler()` is silently DISABLED on
    # a CUDA-less box (XPU): scale/unscale/step/update all become no-ops with
    # only a warning. That's fine for bf16 (the only dtype we run today) but a
    # silent footgun for fp16 — gradients would underflow with no scaling and
    # no error. Make the scaler device-aware and only enable it for fp16, then
    # assert it is actually enabled so an fp16 run can't silently train unscaled.
    scaler = None
    if mixed_precision and dtype == torch.float16:
        _dev = device.type if device is not None else (
            "cuda" if torch.cuda.is_available()
            else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                  else "cpu")
        )
        scaler = torch.amp.GradScaler(device=_dev)
        if not scaler.is_enabled():
            raise RuntimeError(
                f"float16 training requires an enabled GradScaler, but it is "
                f"disabled on device '{_dev}'. Loss scaling would be a no-op "
                f"and fp16 gradients would underflow. Use bfloat16 or a device "
                f"with GradScaler support."
            )
    return optimizer, scaler, scheduler, wd_scheduler
