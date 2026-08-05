# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# ---------------------------------------------------------------------------
# Frozen-feature DENSE tool-segmentation probe for V-JEPA2 (SAR_RARP50).
#
# This is the spatial-localization counterpart to
# `evals.video_classification_frozen` (temporal action recognition). The
# backbone is frozen; a light conv decoder (src/models/segmentation_head.py) is
# trained on the ViT token grid to predict a per-pixel instrument-class map.
#
# Registered by directory name via evals/scaffold.py: set
#     eval_name: video_segmentation_frozen
# in the config YAML and this module's main() is imported and called.
#
# Metric: mean IoU (primary, used for best-checkpoint selection) + pixel
# accuracy, both computed from an all-reduced confusion matrix (dense pixel
# preds are far too large to gather like the classification probe does).
# ---------------------------------------------------------------------------

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

import logging
import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import math

from evals.video_segmentation_frozen.models import init_module
from src.datasets.video_seg_dataset import make_videosegdataset
from src.models.segmentation_head import SegmentationDecoderHead
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger

# NOTE: helpers below (init_opt, schedules, unwrap_module,
# adapt_state_dict_for_model) are copied verbatim from
# evals/video_classification_frozen/eval.py rather than imported. That module
# imports mamba_head at load time, which pulls in triton/mamba_ssm and needs a
# live CUDA driver — importing it would make this eval un-importable on a
# CPU/login node. Keeping the ~40 lines local decouples the two evals.


def unwrap_module(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def adapt_state_dict_for_model(model, state_dict):
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not model_keys or not ckpt_keys:
        return state_dict
    model_has = all(k.startswith("module.") for k in model_keys)
    ckpt_has = all(k.startswith("module.") for k in ckpt_keys)
    if ckpt_has and not model_has:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if model_has and not ckpt_has:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


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
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(final_lr, final_lr + (ref_lr - final_lr)
                             * 0.5 * (1.0 + math.cos(math.pi * progress)))
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
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        # A *disabled* device-generic GradScaler is a pass-through (bf16 needs
        # no loss scaling); torch.cuda.amp.GradScaler is CUDA-only and would
        # crash on XPU.
        scalers += [torch.amp.GradScaler(enabled=False) if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = int(os.environ.get("VJEPA_SEED", "0"))
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def max_mem_mb(device):
    """Peak allocated memory in MiB for the active accelerator, 0.0 on CPU."""
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024.0**2
    if dev.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.max_memory_allocated() / 1024.0**2
    return 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _confusion_from_preds(preds, labels, num_classes, ignore_index=None):
    """Accumulate a [C, C] confusion matrix (rows=true, cols=pred) on device.

    preds, labels: flat int64 tensors of identical shape. Returns a float64
    tensor so cross-rank all-reduce doesn't overflow on large pixel counts.
    """
    valid = (labels >= 0) & (labels < num_classes)
    if ignore_index is not None:
        valid &= labels != ignore_index
    if not valid.any():
        return torch.zeros((num_classes, num_classes), dtype=torch.float64,
                           device=labels.device)
    t = labels[valid].to(torch.int64)
    p = preds[valid].clamp_(0, num_classes - 1).to(torch.int64)
    idx = t * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).to(torch.float64)


def _metrics_from_confusion(cm, ignore_background=False):
    """mIoU + pixel accuracy + per-class IoU from a confusion matrix.

    cm: numpy [C, C], rows=true, cols=pred.
    ignore_background: exclude class 0 from the mIoU mean (common in surgical
    seg since background dominates). Per-class IoU still reported for all.
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    pixel_acc = tp.sum() / max(cm.sum(), 1.0)
    # Mean over classes that actually appear (denom>0). Optionally drop bg.
    start = 1 if ignore_background else 0
    present = ~np.isnan(iou[start:])
    miou = float(np.nanmean(iou[start:])) if present.any() else 0.0
    # Dice / F1 per class -> macro
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    macro_dice = float(np.nanmean(dice[start:])) if present.any() else 0.0
    return {
        "miou": miou,
        "pixel_acc": float(pixel_acc),
        "macro_dice": macro_dice,
        "per_class_iou": np.nan_to_num(iou).tolist(),
    }


def _all_reduce_cm(cm_tensor):
    """Sum a confusion-matrix tensor across ranks (in place-safe)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(cm_tensor, op=torch.distributed.ReduceOp.SUM)
    return cm_tensor


def _dice_loss(logits, labels, num_classes, ignore_index=None, eps=1.0):
    """Soft multi-class Dice loss. logits [N, C], labels [N]."""
    valid = (labels >= 0) & (labels < num_classes)
    if ignore_index is not None:
        valid &= labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    logits = logits[valid]
    labels = labels[valid]
    probs = F.softmax(logits.float(), dim=-1)
    onehot = F.one_hot(labels, num_classes).float()
    inter = (probs * onehot).sum(dim=0)
    union = probs.sum(dim=0) + onehot.sum(dim=0)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args_eval, resume_preempt=False):

    val_only = args_eval.get("val_only", False)

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 8)

    # -- PRETRAIN (frozen backbone)
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER (segmentation head)
    args_classifier = args_exp.get("classifier")
    classifier_name = args_classifier.get("name", "segmentation")
    seg_kwargs = args_classifier.get("segmentation_kwargs", {})

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoSegDataset")
    num_classes = args_data.get("num_classes")
    class_weights = args_data.get("class_weights", None)
    ignore_index = args_data.get("ignore_index", None)
    ignore_background_in_miou = args_data.get("ignore_background_in_miou", False)
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 384)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 1)
    out_hw = args_data.get("out_hw", [resolution, resolution])
    normalization = args_data.get("normalization", None)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16", True)
    dice_weight = float(args_opt.get("dice_weight", 0.0))

    early_stop_patience = args_eval.get("early_stop_patience", None)
    save_every_iters = args_eval.get("save_every_iters", None)

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

    # -- log/checkpointing paths. Subdir MUST match this eval_name so the
    # launcher's resume-check finds it.
    folder = os.path.join(pretrain_folder, "video_segmentation_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best.pt")

    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.5f", "train_loss"),
            ("%.5f", "train_miou"),
            ("%.5f", "train_pixel_acc"),
            ("%.5f", "val_loss"),
            ("%.5f", "val_miou"),
            ("%.5f", "val_pixel_acc"),
            ("%.5f", "val_macro_dice"),
            ("%d", "best_val_miou_epoch"),
            ("%.5f", "best_val_miou"),
        )

    # -- init frozen backbone
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # -- init segmentation head(s)
    if classifier_name != "segmentation":
        raise ValueError(
            f"video_segmentation_frozen expects classifier.name='segmentation', "
            f"got {classifier_name!r}"
        )
    if not args_wrapper or not args_wrapper.get("preserve_clip_dim", False):
        raise ValueError(
            "segmentation head requires model_kwargs.wrapper_kwargs.preserve_clip_dim: true"
        )
    patch_size = args_model["encoder"].get("patch_size", 16)
    tubelet_size = args_model["encoder"].get("tubelet_size", 2)
    grid_hw = seg_kwargs.get("grid_hw", [resolution // patch_size, resolution // patch_size])
    tokens_per_clip = frames_per_clip // tubelet_size

    def _build_head():
        return SegmentationDecoderHead(
            embed_dim=encoder.embed_dim,
            num_classes=num_classes,
            num_clips=num_segments,
            tokens_per_clip=tokens_per_clip,
            grid_hw=grid_hw,
            out_hw=out_hw,
            decoder_channels=seg_kwargs.get("decoder_channels", 256),
            num_conv_blocks=seg_kwargs.get("num_conv_blocks", 2),
            dropout=seg_kwargs.get("dropout", 0.1),
            supervise_all_frames=seg_kwargs.get("supervise_all_frames", False),
        ).to(device)

    classifiers = [_build_head() for _ in opt_kwargs]

    use_ddp = world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized()
    if use_ddp:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
        logger.info("Using DistributedDataParallel for probe heads")
    else:
        logger.info("Running probe heads without DistributedDataParallel")
    print(classifiers[0])

    # -- dataloaders
    _, train_loader, train_sampler = make_videosegdataset(
        data_paths=train_data_path,
        batch_size=batch_size,
        resolution=resolution,
        out_hw=out_hw,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_segments,
        normalization=normalization,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
    )
    _, val_loader, _ = make_videosegdataset(
        data_paths=val_data_path,
        batch_size=batch_size,
        resolution=resolution,
        out_hw=out_hw,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_segments,
        normalization=normalization,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer / scheduler (one per head), reused from classification eval
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    best_val_miou = float("-inf")
    best_val_miou_epoch = 0
    start_epoch = 0

    if resume_checkpoint and os.path.exists(latest_path) and os.path.getsize(latest_path) > 0:
        try:
            classifiers, optimizer, scaler, start_epoch, ckpt_best = load_checkpoint(
                device=device, r_path=latest_path, classifiers=classifiers,
                opt=optimizer, scaler=scaler, val_only=val_only,
            )
            if ckpt_best is not None:
                best_val_miou, best_val_miou_epoch = ckpt_best
            for _ in range(start_epoch * ipe):
                [s.step() for s in scheduler]
                [wds.step() for wds in wd_scheduler]
        except Exception as e:
            logger.warning("Resume from %s failed (%s); starting fresh.", latest_path, e)
            start_epoch = 0

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(list(class_weights), dtype=torch.float32, device=device)

    def _build_save_dict(epoch):
        return {
            "classifiers": [unwrap_module(c).state_dict() for c in classifiers],
            "opt": [o.state_dict() for o in optimizer],
            "scaler": None if scaler is None else [
                (s.state_dict() if s is not None else None) for s in scaler
            ],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "best_val_miou": best_val_miou,
            "best_val_miou_epoch": best_val_miou_epoch,
        }

    def _atomic_torch_save(obj, path):
        tmp = path + ".tmp"
        bak = path + ".bak"
        torch.save(obj, tmp)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            os.replace(path, bak)
        os.replace(tmp, path)

    def save_checkpoint(epoch):
        if rank == 0:
            _atomic_torch_save(_build_save_dict(epoch), latest_path)

    # -- TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)

        if val_only:
            train_loss, train_metrics = -1.0, {"miou": 0.0, "pixel_acc": 0.0}
        else:
            train_loss, train_cms = run_one_epoch(
                device=device, training=True, encoder=encoder,
                classifiers=classifiers, scaler=scaler, optimizer=optimizer,
                scheduler=scheduler, wd_scheduler=wd_scheduler,
                data_loader=train_loader, use_bfloat16=use_bfloat16,
                num_classes=num_classes, weight_tensor=weight_tensor,
                ignore_index=ignore_index, dice_weight=dice_weight,
                sub_epoch_save_fn=(lambda: save_checkpoint(epoch)) if save_every_iters else None,
                save_every_iters=save_every_iters,
            )
            # Best head by train mIoU used only for the logged train number.
            train_metrics, _ = _best_head_metrics(train_cms, ignore_background_in_miou)

        val_loss, val_cms = run_one_epoch(
            device=device, training=False, encoder=encoder,
            classifiers=classifiers, scaler=scaler, optimizer=optimizer,
            scheduler=scheduler, wd_scheduler=wd_scheduler,
            data_loader=val_loader, use_bfloat16=use_bfloat16,
            num_classes=num_classes, weight_tensor=weight_tensor,
            ignore_index=ignore_index, dice_weight=dice_weight,
        )
        val_metrics, best_head = _best_head_metrics(val_cms, ignore_background_in_miou)

        cur_val_miou = 100.0 * val_metrics["miou"]
        improved = cur_val_miou > best_val_miou
        if improved:
            best_val_miou = cur_val_miou
            best_val_miou_epoch = epoch + 1
            if rank == 0:
                _atomic_torch_save(_build_save_dict(epoch + 1), best_path)
                logger.info("Saved best -> best.pt: val_mIoU=%.4f @ep%d",
                            best_val_miou, epoch + 1)

        logger.info(
            "[%5d] train: mIoU %.3f%% pixacc %.3f%% (loss %.3f) | "
            "val: mIoU %.3f%% pixacc %.3f%% dice %.3f%% (loss %.3f) | "
            "best_val_mIoU %.3f%% @ep%d",
            epoch + 1,
            100.0 * train_metrics["miou"], 100.0 * train_metrics["pixel_acc"], train_loss,
            100.0 * val_metrics["miou"], 100.0 * val_metrics["pixel_acc"],
            100.0 * val_metrics.get("macro_dice", 0.0), val_loss,
            best_val_miou, best_val_miou_epoch,
        )
        if rank == 0:
            csv_logger.log(
                epoch + 1, train_loss,
                100.0 * train_metrics["miou"], 100.0 * train_metrics["pixel_acc"],
                val_loss,
                100.0 * val_metrics["miou"], 100.0 * val_metrics["pixel_acc"],
                100.0 * val_metrics.get("macro_dice", 0.0),
                best_val_miou_epoch, best_val_miou,
            )
            # Log per-class IoU of the best head for diagnostics.
            logger.info("val per-class IoU (best head %d): %s", best_head,
                        ["%.3f" % v for v in val_metrics.get("per_class_iou", [])])

        if val_only:
            return
        save_checkpoint(epoch + 1)

        if early_stop_patience is not None:
            no_improve = (epoch + 1) - best_val_miou_epoch
            if no_improve >= early_stop_patience:
                logger.info("Early stop at epoch %d (no val mIoU improvement for %d).",
                            epoch + 1, early_stop_patience)
                logger.info("DONE segmentation (early stop) best_val_mIoU=%.4f @ep%d",
                            best_val_miou, best_val_miou_epoch)
                return

    # Completed all num_epochs without early-stop. This explicit marker lets an
    # external preempt-watcher tell a clean finish from a preemption (qstat-gone
    # with no marker => resubmit; marker present => done).
    logger.info("DONE segmentation (all epochs) best_val_mIoU=%.4f @ep%d",
                best_val_miou, best_val_miou_epoch)


def _best_head_metrics(cms, ignore_background):
    """Pick the head with the highest mIoU; return (metric_dict, head_index)."""
    best_m, best_h, best_miou = None, 0, -1.0
    for hi, cm in enumerate(cms):
        m = _metrics_from_confusion(cm, ignore_background=ignore_background)
        if m["miou"] > best_miou:
            best_miou, best_m, best_h = m["miou"], m, hi
    if best_m is None:
        best_m = {"miou": 0.0, "pixel_acc": 0.0, "macro_dice": 0.0, "per_class_iou": []}
    return best_m, best_h


def run_one_epoch(
    device, training, encoder, classifiers, scaler, optimizer, scheduler,
    wd_scheduler, data_loader, use_bfloat16, num_classes, weight_tensor=None,
    ignore_index=None, dice_weight=0.0, sub_epoch_save_fn=None, save_every_iters=None,
):
    for c in classifiers:
        c.train(mode=training)

    ce_ignore = -100 if ignore_index is None else int(ignore_index)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ce_ignore)
    loss_meters = [AverageMeter() for _ in classifiers]
    # One [C,C] confusion matrix per head, accumulated on device.
    cms = [torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
           for _ in classifiers]

    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)  # [B, N_pix]

            with torch.no_grad():
                outputs = encoder(clips, clip_indices)  # list over spatial views
            # head per view: each -> [B, N_pix, C]
            head_outputs = [[c(o) for o in outputs] for c in classifiers]

        # -- loss (dense CE + optional Dice), flattened like the seq-label path
        losses = []
        for coutputs in head_outputs:
            per_view = []
            for o in coutputs:
                logits = o.reshape(-1, o.shape[-1]).float()
                lab = labels.reshape(-1)
                loss = criterion(logits, lab)
                if dice_weight > 0.0:
                    loss = loss + dice_weight * _dice_loss(
                        logits, lab, num_classes, ignore_index=ignore_index
                    )
                per_view.append(loss)
            losses.append(per_view)

        # -- predictions -> confusion matrix (avg softmax across views)
        with torch.no_grad():
            for ci, coutputs in enumerate(head_outputs):
                prob = sum(F.softmax(o.float(), dim=-1) for o in coutputs) / len(coutputs)
                preds = prob.argmax(dim=-1).reshape(-1)
                cms[ci] += _confusion_from_preds(
                    preds, labels.reshape(-1), num_classes, ignore_index=ignore_index
                )
            for lm, li in zip(loss_meters, losses):
                lm.update(float(sum(l.detach() for l in li) / len(li)))

        if training:
            if use_bfloat16:
                [s.scale(sum(li)).backward() for s, li in zip(scaler, losses)]
                [s.step(o) for s, o in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [sum(li).backward() for li in losses]
                [o.step() for o in optimizer]
            [o.zero_grad() for o in optimizer]

        if itr % 10 == 0:
            _agg = np.array([lm.avg for lm in loss_meters])
            logger.info("[%5d] loss: %.3f [mem: %.2e]", itr, _agg.max(), max_mem_mb(device))

        if (training and sub_epoch_save_fn is not None and save_every_iters
                and (itr + 1) % save_every_iters == 0):
            sub_epoch_save_fn()

    # All-reduce confusion matrices across ranks, return as numpy.
    out_cms = []
    for cm in cms:
        _all_reduce_cm(cm)
        out_cms.append(cm.cpu().numpy())
    return max(lm.avg for lm in loss_meters), out_cms


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")
    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(adapt_state_dict_for_model(c, pd))
           for c, pd in zip(classifiers, pretrained_dict)]
    best = (
        float(checkpoint.get("best_val_miou", float("-inf"))),
        int(checkpoint.get("best_val_miou_epoch", 0)),
    )
    if val_only:
        logger.info(f"loaded classifier (val_only) with msg: {msg}")
        return classifiers, opt, scaler, 0, best
    epoch = checkpoint["epoch"]
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]
    if scaler is not None and checkpoint.get("scaler") is not None:
        for s, pd in zip(scaler, checkpoint["scaler"]):
            if s is not None and pd is not None:
                s.load_state_dict(pd)
    logger.info(f"loaded classifier+opt from epoch {epoch} with msg: {msg}")
    return classifiers, opt, scaler, epoch, best
