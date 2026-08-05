# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# ---------------------------------------------------------------------------
# Dense tool-segmentation dataset for the frozen V-JEPA2 probe (SAR_RARP50).
#
# Unlike VideoDataset (which returns a whole-clip int / temporal-sequence label)
# this returns a DENSE per-pixel class map for the center frame of each clip,
# geometrically aligned to the RGB frames so the segmentation head can be
# supervised pixel-for-pixel.
#
# CSV format (space-delimited, no header), one row per annotated frame:
#     <mask_png> <rgb_dir> <center_num> <rgb_step> <max_num>
#   mask_png    absolute path to the SAR_RARP50 segmentation PNG (HxWx3, the
#               class id 0..num_classes-1 replicated across the 3 channels)
#   rgb_dir     absolute path to the sibling rgb/ dir of extracted frames
#               named <9-digit original-frame-number>.png
#   center_num  original frame number of the annotated frame (also the rgb
#               filename stem, zero-padded to 9 digits)
#   rgb_step    spacing (in original-frame units) between consecutive rgb PNGs
#               (6 for the 10 Hz export of the 60 Hz source)
#   max_num     largest rgb frame number available for that video (for clamping)
#
# The clip is built by sampling `frames_per_clip` rgb frames centered on
# `center_num` with stride `frame_step` (in rgb-frame units), so the head's
# center temporal token lines up with the annotated frame.
# ---------------------------------------------------------------------------

import os
from logging import getLogger

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

logger = getLogger()

_GLOBAL_SEED = 0
DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_videosegdataset(
    data_paths,
    batch_size,
    resolution=384,
    out_hw=None,
    frames_per_clip=16,
    frame_step=1,
    num_clips=1,
    normalization=None,
    world_size=1,
    rank=0,
    training=False,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    drop_last=False,
    collator=None,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION
    if out_hw is None:
        out_hw = (resolution, resolution)

    dataset = VideoSegDataset(
        data_paths=data_paths,
        resolution=resolution,
        out_hw=out_hw,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        normalization=normalization,
        training=training,
    )
    logger.info("VideoSegDataset created (%d samples)", len(dataset))

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=training
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("VideoSegDataset data loader created")
    return dataset, data_loader, dist_sampler


class VideoSegDataset(torch.utils.data.Dataset):
    """Dense tool-segmentation dataset (frame clip -> center-frame pixel mask)."""

    def __init__(
        self,
        data_paths,
        resolution=384,
        out_hw=(384, 384),
        frames_per_clip=16,
        frame_step=1,
        num_clips=1,
        normalization=DEFAULT_NORMALIZATION,
        training=False,
    ):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.data_paths = data_paths
        self.resolution = int(resolution)
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.frames_per_clip = int(frames_per_clip)
        self.frame_step = int(frame_step)
        self.num_clips = int(num_clips)
        self.mean = normalization[0]
        self.std = normalization[1]
        self.training = training

        rows = []
        for p in self.data_paths:
            if p is None:
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        logger.warning("Skipping malformed seg CSV row: %r", line)
                        continue
                    mask_png, rgb_dir, center_num, rgb_step, max_num = parts[:5]
                    rows.append(
                        (mask_png, rgb_dir, int(center_num), int(rgb_step), int(max_num))
                    )
        if not rows:
            raise ValueError(f"No usable rows in seg CSV(s): {self.data_paths}")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _frame_path(self, rgb_dir, num):
        return os.path.join(rgb_dir, f"{num:09d}.png")

    def _clip_frame_numbers(self, center_num, rgb_step, max_num):
        """Frame numbers for one clip centered on `center_num`.

        Stride in original-frame units = frame_step * rgb_step. Positions are
        clamped to [0, max_num] and snapped to a multiple of rgb_step so the
        computed filename actually exists.
        """
        fpc = self.frames_per_clip
        stride = max(1, self.frame_step) * rgb_step
        center_pos = fpc // 2
        nums = []
        for i in range(fpc):
            n = center_num + (i - center_pos) * stride
            n = int(round(n / rgb_step) * rgb_step)
            n = max(0, min(n, max_num))
            nums.append(n)
        return nums

    def _load_rgb_clip(self, rgb_dir, frame_nums):
        """Return normalized clip tensor [C, T, H, W] at self.resolution."""
        frames = []
        for n in frame_nums:
            fp = self._frame_path(rgb_dir, n)
            img = Image.open(fp).convert("RGB")
            img = self._resize_center_crop(img, Image.BILINEAR)
            t = TF.to_tensor(img)  # [C,H,W] in [0,1]
            t = TF.normalize(t, mean=self.mean, std=self.std)
            frames.append(t)
        clip = torch.stack(frames, dim=1)  # [C, T, H, W]
        return clip

    def _resize_center_crop(self, pil_img, interp):
        """Aspect-preserving short-side resize to `resolution`, then center crop.

        Matches the backbone's eval transform geometry so frames stay
        in-distribution AND the mask (transformed identically) stays aligned.
        """
        r = self.resolution
        w, h = pil_img.size
        short = min(w, h)
        new_w, new_h = int(round(w * r / short)), int(round(h * r / short))
        pil_img = pil_img.resize((new_w, new_h), interp)
        left = (new_w - r) // 2
        top = (new_h - r) // 2
        return pil_img.crop((left, top, left + r, top + r))

    def _load_mask(self, mask_png):
        """Return LongTensor [H_out*W_out] of class ids in raster order."""
        m = Image.open(mask_png)
        if m.mode != "L":
            # SAR_RARP50 masks store the class id replicated across RGB; take
            # the first channel.
            m = m.split()[0]
        # Same geometry as the RGB frames, but NEAREST to preserve class ids.
        m = self._resize_center_crop(m, Image.NEAREST)
        if (m.size[1], m.size[0]) != self.out_hw:
            m = m.resize((self.out_hw[1], self.out_hw[0]), Image.NEAREST)
        arr = np.asarray(m, dtype=np.int64)  # [H_out, W_out]
        return torch.from_numpy(arr.reshape(-1))  # [H_out*W_out]

    def __getitem__(self, index):
        mask_png, rgb_dir, center_num, rgb_step, max_num = self.rows[index]

        # Build num_clips clips. For dense single-frame supervision num_clips is
        # normally 1; if >1 the clips are temporally offset windows around the
        # same annotated frame (extra temporal context for the backbone).
        buffer, clip_indices = [], []
        for c in range(self.num_clips):
            frame_nums = self._clip_frame_numbers(center_num, rgb_step, max_num)
            clip = self._load_rgb_clip(rgb_dir, frame_nums)  # [C,T,H,W]
            buffer.append([clip])  # one spatial view
            clip_indices.append(np.array(frame_nums, dtype=np.int64))

        # Label: dense mask for the center (annotated) frame. The head returns
        # T_total = num_clips (center token per clip); tile the same mask per
        # clip so shapes line up, flattened in (T_total, H, W) raster order.
        mask = self._load_mask(mask_png)  # [H*W]
        if self.num_clips > 1:
            label = mask.repeat(self.num_clips)  # [num_clips*H*W]
        else:
            label = mask
        return buffer, label, clip_indices


__all__ = ["VideoSegDataset", "make_videosegdataset"]
