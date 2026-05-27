# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    local_rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID"))
    if local_rank is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = local_rank
except Exception:
    pass

import logging
import math
import pprint
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.backbone_feature_cache import make_backbone_feature_cache
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
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
    if enabled and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


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
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
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

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    profile_timing = args_opt.get("profile_timing", False)
    profile_log_interval = args_opt.get("profile_log_interval", 10)
    profile_warmup_iters = args_opt.get("profile_warmup_iters", 5)
    profile_cuda_sync = args_opt.get("profile_cuda_sync", True)
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

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

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

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.5f", "train_loss"),
            ("%.5f", "train_acc"),
            ("%.5f", "val_loss"),
            ("%.5f", "val_acc"),
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
    # -- init classifier
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
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        all_classifier_dicts = [unwrap_module(c).state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        if val_only:
            train_acc = -1.0
            train_loss = -1.0
        else:
            train_acc, train_loss = run_one_epoch(
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
            )

        val_acc, val_loss = run_one_epoch(
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
        )

        logger.info(
            "[%5d] train: %.3f%% (loss: %.3f) test: %.3f%% (loss: %.3f)"
            % (epoch + 1, train_acc, train_loss, val_acc, val_loss)
        )
        if rank == 0:
            csv_logger.log(epoch + 1, train_loss, train_acc, val_loss, val_acc)

        if val_only:
            return

        save_checkpoint(epoch + 1)


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
):

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    top1_meters = [AverageMeter() for _ in classifiers]
    loss_meters = [AverageMeter() for _ in classifiers]
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
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
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
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                with torch.no_grad():
                    outputs = encoder(clips, clip_indices)
            maybe_sync(device, timed_iteration and profile_cuda_sync)
            encoder_time = time.perf_counter() - encoder_start_time

        maybe_sync(device, timed_iteration and profile_cuda_sync)
        head_start_time = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            if not training:
                outputs = [[c(o) for o in outputs] for c in classifiers]
            if training:
                outputs = [[c(o) for o in outputs] for c in classifiers]
        maybe_sync(device, timed_iteration and profile_cuda_sync)
        head_time = time.perf_counter() - head_start_time

        maybe_sync(device, timed_iteration and profile_cuda_sync)
        loss_start_time = time.perf_counter()
        # Compute loss
        losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]
        with torch.no_grad():
            outputs = [sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]
            top1_accs = [100.0 * coutputs.max(dim=1).indices.eq(labels).sum() / batch_size for coutputs in outputs]
            top1_accs = [float(AllReduce.apply(t1a)) for t1a in top1_accs]
            for t1m, t1a in zip(top1_meters, top1_accs):
                t1m.update(t1a)
            loss_vals = [sum([float(AllReduce.apply(lij)) for lij in li]) / len(li) for li in losses]
            for lm, lv in zip(loss_meters, loss_vals):
                lm.update(lv)
        maybe_sync(device, timed_iteration and profile_cuda_sync)
        loss_time = time.perf_counter() - loss_start_time

        backward_time = 0.0
        if training:
            maybe_sync(device, timed_iteration and profile_cuda_sync)
            backward_start_time = time.perf_counter()
            if use_bfloat16:
                [[s.scale(lij).backward() for lij in li] for s, li in zip(scaler, losses)]
                [s.step(o) for s, o in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [[lij.backward() for lij in li] for li in losses]
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
                    torch.cuda.max_memory_allocated() / 1024.0**2,
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

        prev_iter_end_time = time.perf_counter()

    return _agg_top1.max(), _agg_loss.max()


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading encoder
    pretrained_dict = checkpoint["classifiers"]
    msg = [
        c.load_state_dict(adapt_state_dict_for_model(c, pd))
        for c, pd in zip(classifiers, pretrained_dict)
    ]

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
        return classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch


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
    )
    return data_loader, data_sampler


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
        scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
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
