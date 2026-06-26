# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
# On CUDA we pin one GPU per process via CUDA_VISIBLE_DEVICES. On Aurora/XPU
# the launcher (app/main_dist_aurora) already pins a single tile per rank via
# ZE_AFFINITY_MASK before import, so we must NOT set CUDA_VISIBLE_DEVICES there.
try:
    local_rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID"))
    _xpu_pinned = bool(os.environ.get("ZE_AFFINITY_MASK"))
    if local_rank is not None and not _xpu_pinned:
        os.environ["CUDA_VISIBLE_DEVICES"] = local_rank
except Exception:
    pass

import json
import logging
import math
import pprint
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.backbone_feature_cache import make_backbone_feature_cache
from src.datasets.data_manager import init_data
# MambaHead requires mamba_ssm (CUDA-only custom kernel); import lazily inside
# the classifier_name == "mamba" branch so the asformer path works on XPU.
from src.models.asformer_head import ASFormerHead
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = int(os.environ.get("VJEPA_PROBE_SEED", "0"))
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def unwrap_module(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def is_main_process():
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


def maybe_sync(device, enabled):
    if not enabled:
        return
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif dev.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize(device)


def max_mem_mb(device):
    """Peak allocated memory in MiB for the active accelerator, 0.0 on CPU."""
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024.0**2
    if dev.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.max_memory_allocated() / 1024.0**2
    return 0.0


def _compute_metrics(preds, labels, num_classes):
    """Sklearn-free per-class precision/recall/F1 + macro/weighted.

    Mirrors `per_class_metrics` in eval_perclass_f1.py so training-time
    numbers line up with the standalone evaluator. Returns a dict with
    scalar keys 'accuracy', 'macro_f1', 'weighted_f1', 'macro_recall'.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if preds.size:
        # Vectorised confusion-matrix accumulation.
        valid = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
        if valid.any():
            np.add.at(cm, (labels[valid], preds[valid]), 1)

    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    true_sum = cm.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        recall = np.where(true_sum > 0, tp / true_sum, 0.0)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    accuracy = tp.sum() / max(cm.sum(), 1)
    macro_f1 = float(f1.mean())
    weighted_f1 = float((f1 * true_sum).sum() / max(true_sum.sum(), 1))
    macro_recall = float(recall.mean())
    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_recall": macro_recall,
    }


def _gather_preds_labels(preds_np, labels_np, world_size, rank):
    """All-gather per-rank numpy arrays of preds/labels onto every rank.

    Returns concatenated (preds, labels) on rank 0, and (None, None) on others.
    Uses all_gather_object so the arrays don't have to share shape.
    """
    if world_size <= 1 or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return preds_np, labels_np
    gathered_preds = [None for _ in range(world_size)]
    gathered_labels = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_preds, preds_np)
    torch.distributed.all_gather_object(gathered_labels, labels_np)
    if rank == 0:
        return np.concatenate(gathered_preds), np.concatenate(gathered_labels)
    return None, None


def adapt_state_dict_for_model(model, state_dict):
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if not model_keys or not ckpt_keys:
        return state_dict

    model_has_module_prefix = all(k.startswith("module.") for k in model_keys)
    ckpt_has_module_prefix = all(k.startswith("module.") for k in ckpt_keys)

    if ckpt_has_module_prefix and not model_has_module_prefix:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if model_has_module_prefix and not ckpt_has_module_prefix:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    classifier_name = args_classifier.get("name", "attentive")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    asformer_kwargs = args_classifier.get("asformer_kwargs", {})
    mamba_kwargs = args_classifier.get("mamba_kwargs", {})

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    sequence_labels = args_data.get("sequence_labels", False)
    class_weights = args_data.get("class_weights", None)
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    train_cache_root = args_data.get("train_cache_root", None)
    val_cache_root = args_data.get("val_cache_root", None)
    cache_num_workers = args_data.get("cache_num_workers", 1)
    cache_require_complete = args_data.get("cache_require_complete", True)
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)
    # Predict-on-subwindow variant: when set to [s, e] the trainer keeps the
    # full clip sequence flowing through the head (so the SSM/ASFormer sees
    # the same causal context as the unrestricted run) but slices both
    # `outputs[:, s:e, :]` and `labels[:, s:e]` BEFORE the loss / smoothing /
    # accuracy / F1 pipeline. Default None preserves every existing config.
    supervise_token_range = args_data.get("supervise_token_range", None)
    if supervise_token_range is not None:
        if not sequence_labels:
            raise ValueError(
                "supervise_token_range requires experiment.data.sequence_labels=true"
            )
        if (
            not isinstance(supervise_token_range, (list, tuple))
            or len(supervise_token_range) != 2
        ):
            raise ValueError(
                f"supervise_token_range must be [start, end], got {supervise_token_range!r}"
            )
        _stok, _etok = int(supervise_token_range[0]), int(supervise_token_range[1])
        _ttot = int(num_segments) * (int(frames_per_clip) // 2)
        if not (0 <= _stok < _etok <= _ttot):
            raise ValueError(
                f"supervise_token_range=[{_stok},{_etok}] out of bounds for "
                f"num_segments*frames_per_clip//2={_ttot}"
            )
        supervise_token_range = (_stok, _etok)
        logger.info(
            "supervise_token_range=[%d,%d] applied (T %d -> %d)",
            _stok, _etok, _ttot, _etok - _stok,
        )

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    profile_timing = args_opt.get("profile_timing", False)
    profile_log_interval = args_opt.get("profile_log_interval", 10)
    profile_warmup_iters = args_opt.get("profile_warmup_iters", 5)
    profile_cuda_sync = args_opt.get("profile_cuda_sync", True)
    # ASFormer/MS-TCN-style truncated-MSE smoothing loss on adjacent log-
    # probabilities. Zero (the default) is a no-op, so behaviour is unchanged
    # for every existing config. Enable for dense per-frame segmentation
    # tasks like SAR_RARP50 actions, where it reduces over-segmentation.
    loss_smoothing_weight = float(args_opt.get("loss_smoothing_weight", 0.0))
    loss_smoothing_threshold = float(args_opt.get("loss_smoothing_threshold", 4.0))

    # Opt-in early stop on val macro-F1 (already computed natively). None disables.
    # log_val_f1 is accepted for harness symmetry; F1 is always logged here so it's a no-op.
    early_stop_patience = args_eval.get("early_stop_patience", None)
    _ = args_eval.get("log_val_f1", False)
    # Sub-epoch save: persist head + opt every N iters so a preemption mid-epoch
    # doesn't wipe all progress. Resume re-trains the same epoch from iter 0 with
    # the loaded weights (true iter-resume isn't cheap). 0 / None = disabled.
    save_every_iters = args_eval.get("save_every_iters", None)
    # -- RUN DISCIPLINE (opt-in). When true, the eval writes a STATUS.json
    # under `folder`, saves a single rotating `best.pt` instead of separate
    # `best_val_acc.pt` + `best_val_f1.pt`. Default false preserves prior
    # behavior (existing best_val_*.pt files keep being written too in the
    # legacy path).
    write_status_json = args_eval.get("write_status_json", False)
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # Device selection: prefer XPU (Aurora) then CUDA then CPU. On XPU the
    # launcher pins one tile per rank via ZE_AFFINITY_MASK before import, so the
    # only valid index is 0 (xpu:0), NOT local_rank — mirror app/main_dist_aurora.
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu:0")
        torch.xpu.set_device(0)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best.pt")

    # -- STATUS.json + run banner (opt-in via write_status_json).
    status_io = None
    if write_status_json:
        try:
            import status_io as _status_io
            status_io = _status_io
        except Exception as _e:
            logger.warning("write_status_json=true but status_io import failed (%s); skipping", _e)
    if status_io is not None and rank == 0:
        try:
            if not resume_checkpoint:
                _archived = status_io.archive_dir_on_fresh_start(folder)
                if _archived:
                    logger.info("Auto-archived prior run dir -> %s", _archived)
            status_io.init_status(
                run_dir=folder,
                stage=f"asformer_origsplit/{eval_tag or 'main'}",
                backbone=eval_tag or "?",
                total_epochs_target=int(num_epochs),
                best_metric_name="val_macro_f1",
            )
        except Exception as _e:
            logger.warning("status_io init failed (%s); STATUS.json updates disabled", _e)
            status_io = None

    # -- make csv_logger
    # Extended schema (newsplit run, 2026-06): also track macro F1, weighted F1
    # and macro recall on both train and val, and which epoch produced the
    # current best val_acc / best val_macro_f1 checkpoints.
    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.5f", "train_loss"),
            ("%.5f", "train_acc"),
            ("%.5f", "train_macro_f1"),
            ("%.5f", "train_weighted_f1"),
            ("%.5f", "train_macro_recall"),
            ("%.5f", "val_loss"),
            ("%.5f", "val_acc"),
            ("%.5f", "val_macro_f1"),
            ("%.5f", "val_weighted_f1"),
            ("%.5f", "val_macro_recall"),
            ("%d", "best_val_acc_epoch"),
            ("%d", "best_val_f1_epoch"),
            ("%.5f", "best_val_acc"),
            ("%.5f", "best_val_f1"),
        )

    # Initialize model

    # -- init models
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    # -- EXPORT FEATURE CACHE mode: run frozen encoder once over train+val,
    # write a backbone-feature cache, then exit (no head training). Enables fast
    # subsequent probes that read cached features. Opt-in via export_cache: true.
    if args_eval.get("export_cache", False):
        norm = args_data.get("normalization", None)
        for split, dpath, croot in (
            ("train", train_data_path, args_data.get("train_cache_root")),
            ("val", val_data_path, args_data.get("val_cache_root")),
        ):
            if not croot:
                raise ValueError(f"export_cache: missing {split}_cache_root in config")
            logger.info(f"[export] {split}: {dpath} -> {croot}")
            export_feature_cache(
                encoder=encoder,
                cache_root=croot,
                dataset_type=dataset_type,
                root_path=dpath,
                resolution=resolution,
                frames_per_clip=frames_per_clip,
                frame_step=frame_step,
                num_segments=num_segments,
                num_views_per_segment=num_views_per_segment,
                allow_segment_overlap=args_data.get("allow_segment_overlap", True),
                sequence_labels=sequence_labels,
                normalization=norm,
                batch_size=batch_size,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                device=device,
                use_bfloat16=args_opt.get("use_bfloat16", True),
            )
        logger.info("[export] feature cache export complete; exiting.")
        return

    # -- init classifier
    if classifier_name == "asformer":
        # ASFormer expects [B, num_clips, T_clip*S, D] from the encoder wrapper.
        # Validate that the YAML asked for the matching multi-clip layout.
        if not args_wrapper or not args_wrapper.get("preserve_clip_dim", False):
            raise ValueError(
                "classifier.name='asformer' requires "
                "model_kwargs.wrapper_kwargs.preserve_clip_dim: true"
            )
        if not sequence_labels:
            logger.warning(
                "classifier.name='asformer' but experiment.data.sequence_labels "
                "is False; ASFormer is designed for per-frame supervision."
            )

        def _build_head():
            return ASFormerHead(
                embed_dim=encoder.embed_dim,
                num_classes=num_classes,
                num_clips=asformer_kwargs.get("num_clips", num_segments),
                tokens_per_clip=asformer_kwargs.get(
                    "tokens_per_clip", frames_per_clip // 2
                ),
                num_layers=asformer_kwargs.get("num_layers", 10),
                num_heads=asformer_kwargs.get("num_heads", 8),
                mlp_ratio=asformer_kwargs.get("mlp_ratio", 4.0),
                dropout=asformer_kwargs.get("dropout", 0.0),
                return_sequence=asformer_kwargs.get("return_sequence", True),
                temporal_tokens=asformer_kwargs.get("temporal_tokens", None),
            ).to(device)

        classifiers = [_build_head() for _ in opt_kwargs]
    elif classifier_name == "mamba":
        # Lazy import: mamba_ssm is a CUDA-only custom kernel, absent on XPU.
        from evals.video_classification_frozen.modelcustom.mamba_head import MambaHead

        # Mamba-head expects the same per-clip token layout as ASFormer.
        if not args_wrapper or not args_wrapper.get("preserve_clip_dim", False):
            raise ValueError(
                "classifier.name='mamba' requires "
                "model_kwargs.wrapper_kwargs.preserve_clip_dim: true"
            )
        if not sequence_labels:
            logger.warning(
                "classifier.name='mamba' but experiment.data.sequence_labels "
                "is False; MambaHead is designed for per-frame supervision."
            )

        def _build_head():
            return MambaHead(
                embed_dim=encoder.embed_dim,
                num_classes=num_classes,
                num_clips=mamba_kwargs.get("num_clips", num_segments),
                tokens_per_clip=mamba_kwargs.get(
                    "tokens_per_clip", frames_per_clip // 2
                ),
                num_layers=mamba_kwargs.get("num_layers", 10),
                d_state=mamba_kwargs.get("d_state", 16),
                d_conv=mamba_kwargs.get("d_conv", 4),
                expand=mamba_kwargs.get("expand", 2),
                num_stages=mamba_kwargs.get("num_stages", 4),
                mlp_ratio=mamba_kwargs.get("mlp_ratio", 4.0),
                dropout=mamba_kwargs.get("dropout", 0.0),
                return_sequence=mamba_kwargs.get("return_sequence", True),
                temporal_tokens=mamba_kwargs.get("temporal_tokens", None),
            ).to(device)

        classifiers = [_build_head() for _ in opt_kwargs]
    elif classifier_name in ("attentive", None):
        classifiers = [
            AttentiveClassifier(
                embed_dim=encoder.embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes,
                use_activation_checkpointing=True,
            ).to(device)
            for _ in opt_kwargs
        ]
    else:
        raise ValueError(
            f"Unknown classifier.name={classifier_name!r} "
            "(expected 'attentive', 'asformer', or 'mamba')"
        )
    use_ddp = world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized()
    if use_ddp:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
        logger.info("Using DistributedDataParallel for probe classifiers")
    else:
        logger.info("Running probe classifiers without DistributedDataParallel")
    print(classifiers[0])

    train_loader, train_sampler, train_uses_cached_features = make_probe_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
        cache_root=train_cache_root,
        cache_num_workers=cache_num_workers,
        cache_require_complete=cache_require_complete,
        sequence_labels=sequence_labels,
    )
    val_loader, _, val_uses_cached_features = make_probe_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
        cache_root=val_cache_root,
        cache_num_workers=cache_num_workers,
        cache_require_complete=cache_require_complete,
        sequence_labels=sequence_labels,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")
    logger.info(f"Offline train cache enabled: {train_uses_cached_features}")
    logger.info(f"Offline val cache enabled: {val_uses_cached_features}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- load training checkpoint
    # Best-checkpoint bookkeeping (rank 0 only writes; all ranks track to keep
    # logic simple). On a fresh run start_epoch=0 and both bests start at -inf.
    best_val_acc = float("-inf")
    best_val_acc_epoch = 0
    best_val_f1 = float("-inf")
    best_val_f1_epoch = 0
    best_acc_path = os.path.join(folder, "best_val_acc.pt")
    best_f1_path = os.path.join(folder, "best_val_f1.pt")

    start_epoch = 0
    # Pick best of {latest.pt, best.pt} by epoch field on resume.
    def _epoch_of(p):
        if not p or not os.path.exists(p) or os.path.getsize(p) == 0:
            return -1
        try:
            return int(torch.load(p, map_location="cpu", weights_only=False).get("epoch", -1))
        except Exception:
            return -1
    resume_path = latest_path
    if resume_checkpoint and write_status_json and os.path.exists(best_path):
        best_ep = _epoch_of(best_path)
        latest_ep = _epoch_of(latest_path)
        if best_ep > latest_ep:
            logger.info("Resume: best.pt(ep=%d) > latest.pt(ep=%d); resuming from best.pt", best_ep, latest_ep)
            resume_path = best_path
    if resume_checkpoint and os.path.exists(resume_path):
        # Snapshot fresh state before load -- load_checkpoint can partially
        # mutate classifiers (head N raises after heads 0..N-1 loaded). On
        # failure we restore so the fresh-start path keeps clean weights.
        import copy as _copy
        _pre_clf = [_copy.deepcopy(c.state_dict()) for c in classifiers]
        _pre_opt = [_copy.deepcopy(o.state_dict()) for o in optimizer]
        _pre_scaler = [_copy.deepcopy(s.state_dict()) if s is not None else None for s in (scaler or [])]
        try:
            classifiers, optimizer, scaler, start_epoch, ckpt_bests = load_checkpoint(
                device=device,
                r_path=resume_path,
                classifiers=classifiers,
                opt=optimizer,
                scaler=scaler,
                val_only=val_only,
            )
            if ckpt_bests is not None:
                best_val_acc, best_val_acc_epoch, best_val_f1, best_val_f1_epoch = ckpt_bests
            for _ in range(start_epoch * ipe):
                [s.step() for s in scheduler]
                [wds.step() for wds in wd_scheduler]
        except Exception as e:
            # Loader has quarantined the corrupt file. Restore the pre-load
            # snapshot to undo any partial mutation, then start fresh.
            logger.warning(
                "Resume from %s failed (%s); restoring fresh state and starting from epoch 0.",
                resume_path,
                e,
            )
            for c, sd in zip(classifiers, _pre_clf):
                c.load_state_dict(sd)
            for o, sd in zip(optimizer, _pre_opt):
                o.load_state_dict(sd)
            if scaler is not None:
                for s, sd in zip(scaler, _pre_scaler):
                    if s is not None and sd is not None:
                        s.load_state_dict(sd)
            start_epoch = 0

    # RUN START banner (after resume so start_epoch is known).
    if status_io is not None and rank == 0:
        try:
            status_io.write_run_banner(
                run_dir=folder,
                jid=os.environ.get("PBS_JOBID", "?"),
                start_epoch=start_epoch,
                resume_from=(resume_path if (resume_checkpoint and os.path.exists(resume_path)) else "fresh"),
            )
        except Exception as _e:
            logger.warning("write_run_banner failed: %s", _e)

    def _build_save_dict(epoch):
        all_classifier_dicts = [unwrap_module(c).state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]
        return {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            # Persist best-tracking state so resumed runs don't lose history.
            "best_val_acc": best_val_acc,
            "best_val_acc_epoch": best_val_acc_epoch,
            "best_val_f1": best_val_f1,
            "best_val_f1_epoch": best_val_f1_epoch,
        }

    def _atomic_torch_save(obj, path):
        # Atomic save with rotating backup. tmp + rename so a kill mid-write
        # (PBS preemption) can't leave a 0-byte file. Existing target is
        # rotated to .bak first so robust_checkpoint_loader can fall back if
        # the next save is also killed mid-write.
        tmp = path + ".tmp"
        bak = path + ".bak"
        torch.save(obj, tmp)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            os.replace(path, bak)
        os.replace(tmp, path)

    def save_checkpoint(epoch):
        if rank != 0:
            return
        _atomic_torch_save(_build_save_dict(epoch), latest_path)

    def save_best_acc(epoch):
        if rank != 0:
            return
        sd = _build_save_dict(epoch)
        sd["best_metric"] = "val_acc"
        sd["best_metric_value"] = best_val_acc
        _atomic_torch_save(sd, best_acc_path)
        logger.info(f"  ↳ new best val_acc={best_val_acc:.4f} at epoch {epoch} -> {best_acc_path}")

    def save_best_f1(epoch):
        if rank != 0:
            return
        sd = _build_save_dict(epoch)
        sd["best_metric"] = "val_macro_f1"
        sd["best_metric_value"] = best_val_f1
        _atomic_torch_save(sd, best_f1_path)
        logger.info(f"  ↳ new best val_macro_f1={best_val_f1:.4f} at epoch {epoch} -> {best_f1_path}")

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        if val_only:
            train_acc = -1.0
            train_loss = -1.0
            train_per_head_preds_local = [np.zeros(0, dtype=np.int64) for _ in classifiers]
            train_labels_local = np.zeros(0, dtype=np.int64)
        else:
            train_acc, train_loss, train_per_head_preds_local, train_labels_local = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                use_cached_features=train_uses_cached_features,
                profile_timing=profile_timing,
                profile_log_interval=profile_log_interval,
                profile_warmup_iters=profile_warmup_iters,
                profile_cuda_sync=profile_cuda_sync,
                sequence_labels=sequence_labels,
                class_weights=class_weights,
                loss_smoothing_weight=loss_smoothing_weight,
                loss_smoothing_threshold=loss_smoothing_threshold,
                supervise_token_range=supervise_token_range,
                sub_epoch_save_fn=(lambda: save_checkpoint(epoch)) if save_every_iters else None,
                save_every_iters=save_every_iters,
            )

        val_acc, val_loss, val_per_head_preds_local, val_labels_local = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            use_cached_features=val_uses_cached_features,
            profile_timing=profile_timing,
            profile_log_interval=profile_log_interval,
            profile_warmup_iters=profile_warmup_iters,
            profile_cuda_sync=profile_cuda_sync,
            sequence_labels=sequence_labels,
            class_weights=class_weights,
            loss_smoothing_weight=loss_smoothing_weight,
            loss_smoothing_threshold=loss_smoothing_threshold,
            supervise_token_range=supervise_token_range,
        )

        # -- gather preds/labels across ranks and compute F1/recall on rank 0
        # Labels are identical across heads; gather them once.
        _, train_labels_all = _gather_preds_labels(
            np.zeros(0, dtype=np.int64), train_labels_local, world_size, rank
        )
        _, val_labels_all = _gather_preds_labels(
            np.zeros(0, dtype=np.int64), val_labels_local, world_size, rank
        )
        # Then gather each head's predictions.
        train_per_head_preds_all = []
        val_per_head_preds_all = []
        for hi in range(len(classifiers)):
            tp, _ = _gather_preds_labels(
                train_per_head_preds_local[hi], np.zeros(0, dtype=np.int64), world_size, rank
            )
            vp, _ = _gather_preds_labels(
                val_per_head_preds_local[hi], np.zeros(0, dtype=np.int64), world_size, rank
            )
            train_per_head_preds_all.append(tp)
            val_per_head_preds_all.append(vp)

        # Default fallbacks (used on non-rank-0 OR if no predictions accumulated).
        train_metrics = {"accuracy": train_acc / 100.0, "macro_f1": 0.0,
                         "weighted_f1": 0.0, "macro_recall": 0.0}
        val_metrics = {"accuracy": val_acc / 100.0, "macro_f1": 0.0,
                       "weighted_f1": 0.0, "macro_recall": 0.0}
        if rank == 0:
            # Compute per-head metrics, then pick the head with the best val
            # macro-F1 — same "pick the best head" convention the existing
            # accuracy logger uses with `_agg_top1.max()`.
            best_head = 0
            best_head_f1 = -1.0
            val_metrics_per_head = []
            for hi in range(len(classifiers)):
                vm = _compute_metrics(val_per_head_preds_all[hi], val_labels_all, num_classes) \
                    if (val_per_head_preds_all[hi] is not None and val_per_head_preds_all[hi].size) \
                    else dict(val_metrics)
                val_metrics_per_head.append(vm)
                if vm["macro_f1"] > best_head_f1:
                    best_head_f1 = vm["macro_f1"]
                    best_head = hi
            val_metrics = val_metrics_per_head[best_head]
            if train_per_head_preds_all[best_head] is not None and train_per_head_preds_all[best_head].size:
                train_metrics = _compute_metrics(
                    train_per_head_preds_all[best_head], train_labels_all, num_classes
                )

            # Update best trackers using the FRESH val_metrics.
            cur_val_acc = 100.0 * val_metrics["accuracy"]   # match CSV %-scale
            cur_val_f1 = 100.0 * val_metrics["macro_f1"]
            improved_f1 = cur_val_f1 > best_val_f1
            if cur_val_acc > best_val_acc:
                best_val_acc = cur_val_acc
                best_val_acc_epoch = epoch + 1
                if not write_status_json:
                    save_best_acc(epoch + 1)
            if improved_f1:
                best_val_f1 = cur_val_f1
                best_val_f1_epoch = epoch + 1
                if not write_status_json:
                    save_best_f1(epoch + 1)
            if write_status_json and improved_f1:
                # Single rotating best.pt replaces best_val_acc.pt + best_val_f1.pt.
                _atomic_torch_save(_build_save_dict(epoch + 1), best_path)
                logger.info("Saved best ckpt -> best.pt: val_macro_f1=%.4f at epoch %d", best_val_f1, epoch + 1)

        logger.info(
            "[%5d] train: acc %.3f%% f1 %.3f%% rec %.3f%% (loss: %.3f) | "
            "val: acc %.3f%% f1 %.3f%% rec %.3f%% (loss: %.3f) | "
            "best_val_acc %.3f%% @ep%d  best_val_f1 %.3f%% @ep%d"
            % (epoch + 1,
               100.0 * train_metrics["accuracy"],
               100.0 * train_metrics["macro_f1"],
               100.0 * train_metrics["macro_recall"],
               train_loss,
               100.0 * val_metrics["accuracy"],
               100.0 * val_metrics["macro_f1"],
               100.0 * val_metrics["macro_recall"],
               val_loss,
               best_val_acc, best_val_acc_epoch,
               best_val_f1, best_val_f1_epoch)
        )
        if rank == 0:
            csv_logger.log(
                epoch + 1,
                train_loss,
                100.0 * train_metrics["accuracy"],
                100.0 * train_metrics["macro_f1"],
                100.0 * train_metrics["weighted_f1"],
                100.0 * train_metrics["macro_recall"],
                val_loss,
                100.0 * val_metrics["accuracy"],
                100.0 * val_metrics["macro_f1"],
                100.0 * val_metrics["weighted_f1"],
                100.0 * val_metrics["macro_recall"],
                best_val_acc_epoch,
                best_val_f1_epoch,
                best_val_acc,
                best_val_f1,
            )

        if val_only:
            return

        save_checkpoint(epoch + 1)

        # STATUS.json per-epoch update (rank 0 only; best_val_* set above).
        if status_io is not None and rank == 0:
            try:
                status_io.update_epoch(
                    run_dir=folder,
                    epoch=epoch + 1,
                    val_f1=float(best_val_f1),
                    val_acc=float(best_val_acc),
                    saved_best=(best_val_f1_epoch == epoch + 1),
                    best_ckpt_path=best_path if (best_val_f1_epoch == epoch + 1) else None,
                )
            except Exception as _e:
                logger.warning("status_io.update_epoch failed: %s", _e)

        if early_stop_patience is not None:
            # Best-epoch tracking lives only on rank 0 (see above). Broadcast it
            # to every rank so the early-stop decision is unanimous and DDP
            # doesn't deadlock with some ranks looping and others breaking.
            import torch.distributed as dist
            best_ep_t = torch.tensor([best_val_f1_epoch], dtype=torch.long, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(best_ep_t, src=0)
            cur_best_ep = int(best_ep_t.item())
            no_improve = (epoch + 1) - cur_best_ep
            if no_improve >= early_stop_patience:
                if rank == 0:
                    logger.info(
                        "Early stop at epoch %d: val_macro_f1 did not improve for %d epochs "
                        "(best=%.3f%% @ep%d)",
                        epoch + 1,
                        early_stop_patience,
                        best_val_f1,
                        cur_best_ep,
                    )
                if status_io is not None and rank == 0:
                    try:
                        status_io.mark_done(folder, f"Early stop at epoch {epoch+1}")
                    except Exception:
                        pass
                break
    else:
        # Loop completed all num_epochs without early-stop break.
        if status_io is not None and rank == 0:
            try:
                status_io.mark_done(folder, "DONE all epochs")
            except Exception:
                pass


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    use_cached_features=False,
    profile_timing=False,
    profile_log_interval=10,
    profile_warmup_iters=5,
    profile_cuda_sync=True,
    sequence_labels=False,
    class_weights=None,
    loss_smoothing_weight=0.0,
    loss_smoothing_threshold=4.0,
    supervise_token_range=None,
    sub_epoch_save_fn=None,
    save_every_iters=None,
):

    for c in classifiers:
        c.train(mode=training)

    use_smoothing = sequence_labels and loss_smoothing_weight > 0.0

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(list(class_weights), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    top1_meters = [AverageMeter() for _ in classifiers]
    loss_meters = [AverageMeter() for _ in classifiers]
    # Per-rank, per-classifier flat int64 accumulators of predicted vs true
    # token labels. We track every classifier (not just the first) because
    # the existing `_agg_top1.max()` already picks the best head — keeping F1
    # parallel means the F1 we report and use for best-checkpoint selection
    # corresponds to the SAME head whose accuracy is logged.
    preds_chunks: list[list] = [[] for _ in classifiers]
    labels_chunks: list = []
    timing_meters = {
        "data": AverageMeter(),
        "transfer": AverageMeter(),
        "encoder": AverageMeter(),
        "head": AverageMeter(),
        "loss": AverageMeter(),
        "backward": AverageMeter(),
        "total": AverageMeter(),
    }
    prev_iter_end_time = time.perf_counter()
    for itr, data in enumerate(data_loader):
        iter_start_time = time.perf_counter()
        data_wait_time = iter_start_time - prev_iter_end_time
        timed_iteration = profile_timing and itr >= profile_warmup_iters

        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        maybe_sync(device, timed_iteration and profile_cuda_sync)
        transfer_start_time = time.perf_counter()
        with torch.amp.autocast(
            device.type,
            dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
            enabled=use_bfloat16,
        ):
            if use_cached_features:
                features = data[0].to(device, non_blocking=True)
                clips = None
                clip_indices = None
            else:
                clips = [
                    [dij.to(device, non_blocking=True) for dij in di]
                    for di in data[0]
                ]
                clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)
        maybe_sync(device, timed_iteration and profile_cuda_sync)
        transfer_time = time.perf_counter() - transfer_start_time

        if use_cached_features:
            outputs = [features[:, view_idx] for view_idx in range(features.shape[1])]
            encoder_time = 0.0
        else:
            maybe_sync(device, timed_iteration and profile_cuda_sync)
            encoder_start_time = time.perf_counter()
            with torch.amp.autocast(
                device.type,
                dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
                enabled=use_bfloat16,
            ):
                with torch.no_grad():
                    outputs = encoder(clips, clip_indices)
            maybe_sync(device, timed_iteration and profile_cuda_sync)
            encoder_time = time.perf_counter() - encoder_start_time

        maybe_sync(device, timed_iteration and profile_cuda_sync)
        head_start_time = time.perf_counter()
        with torch.amp.autocast(
            device.type,
            dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
            enabled=use_bfloat16,
        ):
            if not training:
                outputs = [[c(o) for o in outputs] for c in classifiers]
            if training:
                outputs = [[c(o) for o in outputs] for c in classifiers]
        maybe_sync(device, timed_iteration and profile_cuda_sync)
        head_time = time.perf_counter() - head_start_time

        # Predict-on-subwindow variant: slice both `outputs` and `labels` to
        # [s, e) along the temporal axis BEFORE loss/smoothing/acc/F1. The
        # head still ran on the full T (so causal SSM context is preserved);
        # only the supervised window changes. Default None = no-op.
        if supervise_token_range is not None and sequence_labels:
            _s, _e = supervise_token_range
            outputs = [[o[:, _s:_e, :] for o in coutputs] for coutputs in outputs]
            labels = labels[:, _s:_e]

        maybe_sync(device, timed_iteration and profile_cuda_sync)
        loss_start_time = time.perf_counter()
        # Compute loss. For sequence labels the head returns logits of shape
        # [B, T, C] and `labels` is shaped [B, T]; flatten both before CE.
        if sequence_labels:
            losses = [
                [
                    # Cast logits to fp32 for loss; matches `class_weights`
                    # dtype (always fp32) and is the standard AMP pattern
                    # for numerical stability of softmax/log in CE.
                    criterion(o.reshape(-1, o.shape[-1]).float(), labels.reshape(-1))
                    for o in coutputs
                ]
                for coutputs in outputs
            ]
            if use_smoothing:
                # Truncated-MSE smoothing on adjacent log-probabilities
                # (ASFormer / MS-TCN, "Multi-Stage Temporal Convolutional
                # Network", Farha & Gall 2019, eq. 2). Penalises high-frequency
                # flips between neighbouring tokens; clamping at threshold^2
                # keeps genuine action boundaries from being overpenalised.
                #
                # Build smoothing PER-CLASSIFIER so that each classifier's
                # total loss owns its own smoothing-loss subgraph. Sharing a
                # single smooth_loss across classifiers causes backward() on
                # the first classifier to free the shared smoothing subgraph,
                # crashing the second classifier's backward ("backward through
                # the graph a second time").
                tau2 = loss_smoothing_threshold ** 2
                new_losses = []
                for ci, coutputs in enumerate(outputs):
                    per_clf_smooth_terms = []
                    for o in coutputs:
                        # o: [B, T, C]
                        logp = F.log_softmax(o.float(), dim=-1)
                        delta = (logp[:, 1:, :] - logp[:, :-1, :]) ** 2
                        delta = torch.clamp(delta, max=tau2)
                        per_clf_smooth_terms.append(delta.mean())
                    per_clf_smooth_loss = torch.stack(per_clf_smooth_terms).mean()
                    new_losses.append([
                        l + loss_smoothing_weight * per_clf_smooth_loss
                        for l in losses[ci]
                    ])
                losses = new_losses
        else:
            # Cast logits to fp32 for loss (see comment in sequence_labels branch).
            losses = [[criterion(o.float(), labels) for o in coutputs] for coutputs in outputs]
        with torch.no_grad():
            if sequence_labels:
                # Average softmax across views, then per-token accuracy.
                outputs = [
                    sum([F.softmax(o, dim=-1) for o in coutputs]) / len(coutputs)
                    for coutputs in outputs
                ]
                n_tokens = float(labels.numel())
                top1_accs = [
                    100.0
                    * coutputs.argmax(dim=-1).eq(labels).sum()
                    / max(n_tokens, 1.0)
                    for coutputs in outputs
                ]
            else:
                outputs = [
                    sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs)
                    for coutputs in outputs
                ]
                top1_accs = [
                    100.0 * coutputs.max(dim=1).indices.eq(labels).sum() / batch_size
                    for coutputs in outputs
                ]
            top1_accs = [float(AllReduce.apply(t1a)) for t1a in top1_accs]
            for t1m, t1a in zip(top1_meters, top1_accs):
                t1m.update(t1a)
            loss_vals = [sum([float(AllReduce.apply(lij)) for lij in li]) / len(li) for li in losses]
            for lm, lv in zip(loss_meters, loss_vals):
                lm.update(lv)
            # Collect predictions + labels per-classifier so we can compute
            # macro/weighted F1 + macro recall at epoch end. `outputs` is
            # already softmax-averaged across views above. Stays on CPU as
            # int64 to keep the buffer compact (these are class ids, not logits).
            for ci, coutputs in enumerate(outputs):
                preds_chunks[ci].append(
                    coutputs.argmax(dim=-1).detach().cpu().reshape(-1).numpy().astype(np.int64)
                )
            labels_chunks.append(
                labels.detach().cpu().reshape(-1).numpy().astype(np.int64)
            )
        maybe_sync(device, timed_iteration and profile_cuda_sync)
        loss_time = time.perf_counter() - loss_start_time

        backward_time = 0.0
        if training:
            maybe_sync(device, timed_iteration and profile_cuda_sync)
            backward_start_time = time.perf_counter()
            if use_bfloat16:
                # ASFormer/MS-TCN: per-stage CE losses share the encoder
                # forward graph. Sum the per-stage losses first, then call
                # backward once — calling .backward() on each stage loss
                # individually frees the shared subgraph after the first
                # call and crashes on the second ("backward through the
                # graph a second time"). Standard MS-TCN training sums
                # stage losses with equal weight.
                [s.scale(sum(li)).backward() for s, li in zip(scaler, losses)]
                [s.step(o) for s, o in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [sum(li).backward() for li in losses]
                [o.step() for o in optimizer]
            [o.zero_grad() for o in optimizer]
            maybe_sync(device, timed_iteration and profile_cuda_sync)
            backward_time = time.perf_counter() - backward_start_time

        iter_total_time = time.perf_counter() - iter_start_time

        if timed_iteration:
            timing_meters["data"].update(data_wait_time)
            timing_meters["transfer"].update(transfer_time)
            timing_meters["encoder"].update(encoder_time)
            timing_meters["head"].update(head_time)
            timing_meters["loss"].update(loss_time)
            timing_meters["backward"].update(backward_time)
            timing_meters["total"].update(iter_total_time)

        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        _agg_loss = np.array([lm.avg for lm in loss_meters])
        if itr % 10 == 0:
            logger.info(
                "[%5d] %.3f%% [%.3f%% %.3f%%] loss: %.3f [mem: %.2e]"
                % (
                    itr,
                    _agg_top1.max(),
                    _agg_top1.mean(),
                    _agg_top1.min(),
                    _agg_loss.max(),
                    max_mem_mb(device),
                )
            )
        if profile_timing and timed_iteration and (itr % profile_log_interval == 0) and is_main_process():
            total_avg = max(timing_meters["total"].avg, 1e-9)
            logger.info(
                (
                    "TIMING[%5d][%s] data=%.3fs (%.1f%%) h2d=%.3fs (%.1f%%) "
                    "encoder=%.3fs (%.1f%%) head=%.3fs (%.1f%%) loss=%.3fs (%.1f%%) "
                    "backward=%.3fs (%.1f%%) total=%.3fs"
                )
                % (
                    itr,
                    "train" if training else "val",
                    timing_meters["data"].avg,
                    100.0 * timing_meters["data"].avg / total_avg,
                    timing_meters["transfer"].avg,
                    100.0 * timing_meters["transfer"].avg / total_avg,
                    timing_meters["encoder"].avg,
                    100.0 * timing_meters["encoder"].avg / total_avg,
                    timing_meters["head"].avg,
                    100.0 * timing_meters["head"].avg / total_avg,
                    timing_meters["loss"].avg,
                    100.0 * timing_meters["loss"].avg / total_avg,
                    timing_meters["backward"].avg,
                    100.0 * timing_meters["backward"].avg / total_avg,
                    timing_meters["total"].avg,
                )
            )

        if (training and sub_epoch_save_fn is not None and save_every_iters
                and (itr + 1) % save_every_iters == 0):
            sub_epoch_save_fn()
            if itr == save_every_iters - 1 or itr % (10 * save_every_iters) == 0:
                logger.info("Sub-epoch checkpoint saved at iter %d", itr + 1)

        prev_iter_end_time = time.perf_counter()

    # Concatenate per-iter chunks. Returns per-classifier preds (list of
    # numpy arrays, one per head) and a single labels array shared by all
    # heads. The caller is responsible for all-gather + metric computation
    # on rank 0.
    per_head_preds = [
        np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
        for chunks in preds_chunks
    ]
    labels_np = np.concatenate(labels_chunks) if labels_chunks else np.zeros(0, dtype=np.int64)
    return _agg_top1.max(), _agg_loss.max(), per_head_preds, labels_np


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading encoder
    pretrained_dict = checkpoint["classifiers"]
    msg = [
        c.load_state_dict(adapt_state_dict_for_model(c, pd))
        for c, pd in zip(classifiers, pretrained_dict)
    ]

    # Best-tracking tuple: (best_val_acc, best_val_acc_epoch,
    # best_val_f1, best_val_f1_epoch). Older checkpoints (pre-newsplit) won't
    # have these keys; default to -inf so the first new epoch always wins.
    bests = (
        float(checkpoint.get("best_val_acc", float("-inf"))),
        int(checkpoint.get("best_val_acc_epoch", 0)),
        float(checkpoint.get("best_val_f1", float("-inf"))),
        int(checkpoint.get("best_val_f1_epoch", 0)),
    )

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
        return classifiers, opt, scaler, 0, bests

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch, bests


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    sequence_labels=False,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        sequence_labels=sequence_labels,
    )
    return data_loader, data_sampler


def export_feature_cache(
    *,
    encoder,
    cache_root,
    dataset_type,
    root_path,
    resolution,
    frames_per_clip,
    frame_step,
    num_segments,
    num_views_per_segment,
    allow_segment_overlap,
    sequence_labels,
    normalization,
    batch_size,
    num_workers,
    world_size,
    rank,
    device,
    use_bfloat16,
    shard_size=64,
):
    """Run the frozen encoder once over a dataset and write a backbone-feature
    cache that BackboneFeatureCacheDataset can read.

    The stored per-sample feature is the EXACT live-path output stacked over
    views: features[i] has shape [num_views, *encoder_out], so the reader's
    `features[:, view_idx]` equals the live `encoder(clips, clip_indices)[view_idx]`.
    Deterministic (training=False transform, no shuffle); spatial pooling is NOT
    applied here -- it is a learnable part of the head -- so we cache full tokens.
    """
    from src.datasets.video_dataset import make_videodataset

    device = torch.device(device) if isinstance(device, str) else device
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    out_dir = Path(cache_root) / f"rank_{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic (val-style) transform; no augmentation. return_sample_path so
    # the cache carries clip paths. shuffle off so row_index == enumeration order.
    transform = make_transforms(
        training=False,
        num_views_per_clip=num_views_per_segment,
        crop_size=resolution,
        normalize=normalization,
    )
    _dataset, data_loader, _sampler = make_videodataset(
        data_paths=root_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        transform=transform,
        world_size=world_size,
        rank=rank,
        drop_last=False,
        num_workers=num_workers,
        sequence_labels=sequence_labels,
        return_sample_path=True,
    )

    encoder.eval()
    shards_meta = []
    buf_feats, buf_labels, buf_rows, buf_paths = [], [], [], []
    feature_shape = None
    row_counter = 0
    shard_idx = 0

    def _flush():
        nonlocal shard_idx
        if not buf_feats:
            return
        feats = torch.stack(buf_feats, dim=0)  # [N, num_views, *enc_out]
        # Labels: stack so scalar->[N], per-clip sequence vectors->[N, T]. The
        # reader returns each row as-is; default collate then matches the live
        # `labels` shape ([B] or [B, T]).
        labels_t = torch.stack([torch.as_tensor(l, dtype=torch.long) for l in buf_labels], dim=0)
        shard_name = f"shard_{shard_idx:05d}.pt"
        torch.save(
            {
                "features": feats,
                "labels": labels_t,
                "row_indices": torch.tensor(buf_rows, dtype=torch.long),
                "sample_paths": list(buf_paths),
            },
            out_dir / shard_name,
        )
        shards_meta.append({"path": shard_name, "num_samples": feats.shape[0]})
        shard_idx += 1
        buf_feats.clear(); buf_labels.clear(); buf_rows.clear(); buf_paths.clear()

    with torch.no_grad():
        for data in data_loader:
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            labels = data[1]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            paths = data[3]
            with torch.amp.autocast(device.type,
                                    dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
                                    enabled=use_bfloat16):
                outputs = encoder(clips, clip_indices)  # list[num_views] of [B, ...]
            # Stack views -> [B, num_views, *enc_out]; store bf16 on CPU.
            feats = torch.stack(outputs, dim=1).to(torch.bfloat16).cpu()
            if feature_shape is None:
                feature_shape = list(feats.shape[1:])  # [num_views, *enc_out]
            # sequence_labels -> label is a per-clip tensor; store as-is via list.
            for i in range(feats.shape[0]):
                buf_feats.append(feats[i])
                buf_labels.append(labels[i])  # scalar or [T] tensor; stacked at flush
                buf_rows.append(row_counter); row_counter += 1
                buf_paths.append(paths[i])
            if len(buf_feats) >= shard_size:
                _flush()
    _flush()

    manifest = {
        "status": "completed",
        "world_size": int(world_size),
        "rank": int(rank),
        "feature_shape_per_sample": feature_shape,
        "shards": shards_meta,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        f"[export rank {rank}] wrote {len(shards_meta)} shards, "
        f"{row_counter} samples -> {out_dir} (feature_shape={feature_shape})"
    )
    return row_counter


def make_probe_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    normalization=None,
    cache_root=None,
    cache_num_workers=1,
    cache_require_complete=True,
    sequence_labels=False,
):
    if cache_root is not None:
        _, data_loader, data_sampler = make_backbone_feature_cache(
            cache_root=cache_root,
            batch_size=batch_size,
            training=training,
            rank=rank,
            world_size=world_size,
            num_workers=cache_num_workers,
            pin_mem=True,
            persistent_workers=True,
            require_complete_export=cache_require_complete,
        )
        return data_loader, data_sampler, True

    data_loader, data_sampler = make_dataloader(
        root_path=root_path,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        dataset_type=dataset_type,
        img_size=img_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=eval_duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=allow_segment_overlap,
        training=training,
        num_workers=num_workers,
        normalization=normalization,
        sequence_labels=sequence_labels,
    )
    return data_loader, data_sampler, False


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        # The training loop always routes through the scaler when use_bfloat16
        # (s.scale/step/update), so we need a valid object — but bf16 needs no
        # loss scaling. A *disabled* device-generic GradScaler is a pass-through
        # (scale=identity, step=optimizer.step, update=no-op) that does no
        # device-specific work, so it is XPU-safe and numerically matches the
        # CUDA bf16 path within noise.
        scalers += [torch.amp.GradScaler(enabled=False) if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
