# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import math
import os
import pathlib
import tarfile
import tempfile
from logging import getLogger

import numpy as np
import torch
import torchvision
import webdataset as wds
from decord import VideoReader, cpu

_GLOBAL_SEED = 0
logger = getLogger()


def compute_mixing_probs(sample_counts, datasets_weights=None, temperature=0.5):
    """Per-sample source-selection probabilities for ``wds.RandomMix``.

    ``RandomMix`` picks a *source* per emitted sample with probability
    ``probs[i] / sum(probs)`` — independent of how many samples each source
    holds (every stream here is ``resampled=True``, i.e. infinite/with-
    replacement). So passing uniform ``datasets_weights`` makes an 8-clip set
    contribute the same fraction of training as a 136k-clip set, oversampling
    the tiny one by ~10^4 and collapsing the effective corpus onto a handful of
    memorized clips. (This silently corrupted all prior surgical pretraining.)

    Semantics (mirrors the standard multilingual temperature-sampling recipe):

      base_i  = sample_count_i ** temperature   (× datasets_weights_i if given)
      prob_i  = base_i / sum_j base_j

    - ``temperature == 1.0``: size-proportional → the realized sample
      distribution is *uniform over the true corpus* (no over/under-sampling).
    - ``temperature == 0.5`` (default): sqrt-size sampling — down-weights tiny
      sets without erasing them; keeps a surgical-heavy mix without memorization.
    - ``temperature == 0.0``: recovers the old uniform-per-source behavior
      (each source 1/N regardless of size) — kept only for explicit opt-in.
    - ``0 < temperature < 1``: gently up-weights small sources without the
      catastrophic blow-up of full uniform (e.g. 0.5 ≈ sqrt-size sampling).
    - ``datasets_weights`` (optional): per-source multipliers applied on top,
      for deliberate domain up/down-weighting (e.g. surgical vs kinetics).

    Returns a list of probabilities summing to 1.0.
    """
    n = len(sample_counts)
    if n == 0:
        return []
    counts = [max(1.0, float(c or 0)) for c in sample_counts]
    if datasets_weights is None:
        datasets_weights = [1.0] * n
    if len(datasets_weights) != n:
        raise ValueError("datasets_weights length must match sample_counts")
    base = [w * (c ** float(temperature)) for w, c in zip(datasets_weights, counts)]
    s = float(sum(base))
    if s <= 0:
        # Degenerate (all weights zero) — fall back to uniform-per-source.
        return [1.0 / n] * n
    return [b / s for b in base]


class VideoDecoder:
    """
    Custom WebDataset decoder to replicate the logic from VideoDataset.
    It decodes the video from bytes, samples frames, and applies transforms.

    Args:
        frames_per_clip (int): Number of frames to sample.
        frame_step (int): Step between sampled frames (from original VideoDataset).
        num_clips (int): Number of clips to extract (default 1).
        random_clip_sampling (bool): Whether to sample randomly.
        allow_clip_overlap (bool): Whether to allow overlap.
        filter_short_videos (bool): If True, skip videos shorter than clip_len.
        filter_long_videos (int): Maximum video file size in bytes.
        transform (callable): The transform to apply to the final clip(s).
        shared_transform (callable): Transform applied *before* splitting into clips.
    """
    def __init__(
        self,
        frames_per_clip=16,
        frame_step=4,
        duration=None,
        fps=None,
        num_clips=1,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        transform=None,
        shared_transform=None,
    ):
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.duration = duration
        self.fps = fps
        self.num_clips = num_clips
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.transform = transform
        self.shared_transform = shared_transform

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")

    def loadvideo_decord(self, video_bytes, fpc):
        """
        Load video content using Decord from a byte buffer.
        Replicates the logic from `VideoDataset.loadvideo_decord`.
        """
        try:
            vr = VideoReader(io.BytesIO(video_bytes), num_threads=1, ctx=cpu(0))
        except Exception as e:
            try:
                logger.info(f"Fallback: Writing {len(video_bytes)} bytes to temp file")
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_name = tmp.name
                    tmp.write(video_bytes)
                    tmp.flush()
                vr = VideoReader(tmp_name, num_threads=1, ctx=cpu(0))
                os.remove(tmp_name)
            except Exception as e2:
                if 'tmp_name' in locals() and os.path.exists(tmp_name):
                    os.remove(tmp_name)
                logger.warning(f"Failed to open video with Decord (BytesIO and TempFile): {e} | {e2}")
                return [], None

        fstp = self.frame_step

        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)
                return [], None

            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = max(1, video_fps // self.fps)

        assert fstp is not None and fstp > 0, "frame_step must be set"
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            logger.warning(f"skipping video of length {len(vr)}")
            return [], None

        vr.seek(0)

        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - partition_len // fstp) * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - sample_len // fstp) * sample_len,
                        )
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def loadimage(self, image_bytes, fpc):
        try:
            image_tensor = torchvision.io.decode_image(
                torch.frombuffer(image_bytes, dtype=torch.uint8),
                mode=torchvision.io.ImageReadMode.RGB,
            )
        except Exception as e:
            print(f"Failed to decode image: {e}", flush=True)
            return [], None

        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))
        buffer = buffer.numpy()
        return buffer, clip_indices

    def decode(self, sample, frames_per_clip):
        """Decode a webdataset sample dict into (clips, label, clip_indices)."""
        try:
            # 1. label
            label_bytes = None
            for key in ("label.txt", "label", "txt", "cls", "class.txt", "class"):
                if key in sample:
                    label_bytes = sample[key]
                    break
            if label_bytes is None:
                available_keys = [k for k in sample.keys() if not k.startswith("__")]
                logger.warning(
                    f"No label for key {sample.get('__key__', 'N/A')}. Available: {available_keys}"
                )
                return None
            try:
                label = int(label_bytes.decode("utf-8").strip())
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to decode label for key {sample.get('__key__', 'N/A')}: {e}")
                return None

            # 2. media (video preferred, then image)
            media_bytes = None
            is_image = False
            for key in ("video.mp4", "video.avi", "video.mov", "video.webm", "video.mkv",
                        "mp4", "avi", "mov", "webm", "mkv", "flv"):
                if key in sample:
                    media_bytes = sample[key]
                    break
            if media_bytes is None:
                for key in ("image.jpg", "image.jpeg", "image.png", "image.bmp",
                            "jpg", "jpeg", "png", "bmp", "gif"):
                    if key in sample:
                        media_bytes = sample[key]
                        is_image = True
                        break
            if media_bytes is None:
                available_keys = [k for k in sample.keys() if not k.startswith("__")]
                logger.warning(
                    f"No media for key {sample.get('__key__', 'N/A')}. Available: {available_keys}"
                )
                return None

            # 3. decode
            if is_image:
                buffer, clip_indices = self.loadimage(media_bytes, frames_per_clip)
            else:
                buffer, clip_indices = self.loadvideo_decord(media_bytes, frames_per_clip)
            if len(buffer) == 0:
                return None

            # 4. transforms
            def split_into_clips(video):
                fpc = frames_per_clip
                nc = self.num_clips
                return [video[i * fpc:(i + 1) * fpc] for i in range(nc)]

            if self.shared_transform is not None:
                buffer = self.shared_transform(buffer)
            if not is_image:
                buffer = split_into_clips(buffer)
            else:
                buffer = [buffer]
            if self.transform is not None:
                buffer = [self.transform(clip) for clip in buffer]

            return buffer, label, clip_indices
        except Exception as e:
            logger.error(f"Error processing sample {sample.get('__key__', 'N/A')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


def is_not_none(x):
    return x is not None


class _NoOpSampler:
    """Stub returned in place of a DistributedSampler to satisfy trainer code
    that calls ``set_epoch`` / ``increase_epoch`` on the sampler."""

    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def increase_epoch(self):
        self.epoch += 1


def _count_samples_in_tar(tar_path):
    """Count distinct sample keys in one tar (members sharing a key but
    differing by extension count once)."""
    seen = set()
    with tarfile.open(tar_path, "r|") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            dot = name.find(".")
            key = name[:dot] if dot > 0 else name
            seen.add(key)
    return len(seen)


def _load_or_build_metadata(path):
    """Return metadata dict (with ``shard_urls`` populated). Reads
    ``metadata.json`` if present; otherwise estimates sample_count from one
    shard and writes a cache file for next time.

    The cache is best-effort: if the dir is read-only we skip the write and
    keep the in-memory result.
    """
    meta_path = os.path.join(path, "metadata.json")
    shard_files = sorted(f for f in os.listdir(path) if f.endswith(".tar"))
    if not shard_files:
        raise FileNotFoundError(f"No .tar shards found under {path}")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # Patch in current shard listing in case it has changed since the cache.
        meta["shard_count"] = len(shard_files)
        meta["shard_urls"] = shard_files
        meta.setdefault("name", os.path.basename(os.path.normpath(path)))
        return meta

    name = os.path.basename(os.path.normpath(path))
    probe_shard = os.path.join(path, shard_files[0])
    samples_in_probe = _count_samples_in_tar(probe_shard)
    sample_count = samples_in_probe * len(shard_files)
    logger.info(
        f"[{name}] no metadata.json; estimated sample_count={sample_count} "
        f"from {len(shard_files)} shards x {samples_in_probe} samples/shard"
    )
    meta = {
        "name": name,
        "shard_count": len(shard_files),
        "sample_count": int(sample_count),
        "shard_urls": shard_files,
        "estimated": True,
    }
    try:
        with open(meta_path, "w") as f:
            json.dump({k: v for k, v in meta.items() if k != "shard_urls"}, f, indent=2)
    except OSError as e:
        logger.warning(f"[{name}] could not cache metadata.json: {e}")
    return meta


class _PerSampleDecode:
    """Picklable wrapper binding (decoder, fpc) for use in WDS .map().

    A closure inside _make_stream would not survive multiprocessing.spawn
    (DataLoader workers re-execute the parent script's __main__ on Aurora
    where 'fork' is unsafe), so we use a regular class instead.
    """

    def __init__(self, decoder, fpc):
        self.decoder = decoder
        self.fpc = fpc

    def __call__(self, sample):
        return self.decoder.decode(sample, self.fpc)


def _make_stream(meta, dataset_dir, decoder, fpc, shuffle_buffer=1000,
                 rank=None, world_size=None):
    """Build one WDS pipeline for a single dataset that yields decoded samples.

    When (rank, world_size) is provided, we slice the URL list to this rank's
    shards BEFORE handing it to WebDataset, and disable WDS's own nodesplitter.
    Combined with per-node shard staging (each node holds the union of its
    LOCAL_WORLD_SIZE ranks' slices), this guarantees every rank only ever opens
    a shard that lives on its local node's /tmp.

    Per-node staging override: when WDS_LOCAL_SLICING=1 is set, we slice by
    *local* rank/world_size instead of global. The local node's data dir holds
    exactly the union of shards its LOCAL_WORLD_SIZE ranks need, so slicing
    across local ranks gives disjoint coverage with no cross-node misses.

    Datasets with fewer shards than world_size keep the full URL list per rank
    (the alternative would starve most ranks of any shard at all).
    """
    if os.environ.get("WDS_LOCAL_SLICING", "0") == "1":
        # Prefer the launcher-exported local ids; fall back to common PMI vars.
        local_rank = int(os.environ.get(
            "LOCAL_RANK", os.environ.get("PALS_LOCAL_RANKID",
            os.environ.get("PMI_LOCAL_RANK", "0"))))
        local_world = int(os.environ.get(
            "LOCAL_WORLD_SIZE", os.environ.get("PALS_LOCAL_SIZE",
            os.environ.get("PMI_LOCAL_SIZE", "1"))))
        slice_rank, slice_ws = local_rank, local_world
    else:
        slice_rank, slice_ws = rank, world_size

    urls = [os.path.join(dataset_dir, u) for u in meta["shard_urls"]]
    if slice_rank is not None and slice_ws is not None and slice_ws > 1:
        if len(urls) >= slice_ws:
            urls = urls[slice_rank::slice_ws]
        # else: keep full list — tiny dataset, every rank uses all shards.
        nodesplitter = None
    else:
        nodesplitter = wds.split_by_node
    stream = wds.WebDataset(
        urls,
        resampled=True,
        shardshuffle=True,
        nodesplitter=nodesplitter,
        workersplitter=wds.split_by_worker,
        handler=wds.warn_and_continue,
    ).shuffle(shuffle_buffer).map(_PerSampleDecode(decoder, fpc)).select(is_not_none)
    return stream


def make_webdataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    sampling_temperature=0.5,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    ipe=None,
    shuffle_buffer=1000,
):
    """Create a WebDataset-based DataLoader honoring rank/world_size via
    ``wds.split_by_node`` and per-dataset frames_per_clip.

    Returns ``(stream, data_loader, dummy_sampler)`` matching the 3-tuple
    contract of the other loaders in this package.
    """
    if not isinstance(data_paths, (list, tuple)):
        data_paths = [data_paths]

    if dataset_fpcs is None:
        dataset_fpcs = [frames_per_clip for _ in data_paths]
    elif len(dataset_fpcs) != len(data_paths):
        raise ValueError("dataset_fpcs length must match data_paths")

    if datasets_weights is not None and len(datasets_weights) != len(data_paths):
        raise ValueError("datasets_weights length must match data_paths")

    decoder = VideoDecoder(
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        duration=duration,
        fps=fps,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        transform=transform,
        shared_transform=shared_transform,
    )

    streams = []
    total_samples = 0
    total_shards = 0
    summary = []
    per_dataset_counts = []
    for path, fpc in zip(data_paths, dataset_fpcs):
        meta = _load_or_build_metadata(path)
        streams.append(_make_stream(
            meta, path, decoder, fpc, shuffle_buffer,
            rank=rank, world_size=world_size,
        ))
        total_samples += int(meta.get("sample_count", 0))
        total_shards += int(meta.get("shard_count", 0))
        per_dataset_counts.append(int(meta.get("sample_count", 0)))
        summary.append((meta["name"], meta.get("shard_count"), meta.get("sample_count")))

    logger.info(
        f"WebDataset: {len(streams)} stream(s), {total_shards} shards, "
        f"~{total_samples} samples; per-dataset: {summary}"
    )

    if len(streams) == 1:
        mixed = streams[0]
    else:
        sample_counts = [int(m or 0) for m in per_dataset_counts]
        probs = compute_mixing_probs(
            sample_counts,
            datasets_weights=datasets_weights,
            temperature=sampling_temperature,
        )
        # Log the REALIZED training mixture vs the true corpus fractions so an
        # oversampling regression (like the old uniform-per-source default that
        # gave an 8-clip set 9% of all training) is visible on iteration 0.
        total = float(sum(sample_counts)) or 1.0
        logger.info(
            "WebDataset mixing (temperature=%.3f): realized per-source sample "
            "fractions vs true corpus fractions:", sampling_temperature
        )
        # Tripwire: expected #times each *individual* clip is shown per epoch =
        # (samples drawn from this source per epoch) / (unique clips it has).
        # This is the quantity that maps directly to memorization, independent
        # of temperature. The old uniform default drove this into the hundreds
        # for tiny sets (e.g. 8-clip endovis15 each clip ~hundreds of times per
        # epoch) while kinetics clips were seen <1x.
        names = [s[0] for s in summary]
        epoch_samples = total  # ~one pass over the corpus per nominal epoch
        reps_per_clip = []
        for name, cnt, p in zip(names, sample_counts, probs):
            reps = (p * epoch_samples / cnt) if cnt > 0 else float("inf")
            reps_per_clip.append((name, cnt, p, reps))
            logger.info(
                "  %-24s n=%-8d true=%6.2f%%  realized=%6.2f%%  "
                "reps/clip/epoch=%.2f",
                name, cnt, 100.0 * cnt / total, 100.0 * p, reps,
            )
        # Two-level guard on per-clip repetition (the memorization signature).
        # WARN: any tiny set whose clips repeat a lot — visible but allowed,
        #   since keeping micro-datasets under temperature sampling is a
        #   deliberate choice (e.g. endovis15's 8 clips ~80x/epoch at T=0.5).
        # FAIL: an egregious level that only the broken uniform-per-source
        #   default produced (tiny sets in the thousands of reps/epoch) — that
        #   is never intentional and indicates a temperature/weights regression.
        warn_reps = float(os.environ.get("VJEPA_WARN_REPS_PER_CLIP", "50"))
        fail_reps = float(os.environ.get("VJEPA_MAX_REPS_PER_CLIP", "1000"))
        worst = max(reps_per_clip, key=lambda r: r[3])
        if worst[3] > fail_reps:
            raise RuntimeError(
                f"Dataset mixing tripwire: source '{worst[0]}' ({worst[1]} "
                f"unique clips) would show each clip ~{worst[3]:.0f} times per "
                f"epoch (realized share {100*worst[2]:.1f}%), exceeding the "
                f"hard limit {fail_reps:.0f}. This is the oversampling/"
                f"memorization regression (the old uniform default did this). "
                f"Fix sampling_temperature/datasets_weights, drop the micro-"
                f"dataset, or raise VJEPA_MAX_REPS_PER_CLIP if truly intended."
            )
        for name, cnt, p, reps in reps_per_clip:
            if reps > warn_reps:
                logger.warning(
                    "Dataset mixing: '%s' (%d clips) repeats each clip ~%.0fx "
                    "per epoch (share %.1f%%) — memorization risk; kept by "
                    "config (temperature=%.2f).",
                    name, cnt, reps, 100.0 * p, sampling_temperature,
                )
        mixed = wds.RandomMix(streams, probs=probs)

    if ipe is None:
        if total_samples > 0 and world_size > 0 and batch_size > 0:
            samples_per_rank = max(1, total_samples // world_size)
            ipe = max(1, samples_per_rank // batch_size)
        else:
            ipe = 1
    ipe = int(ipe)

    # Build the WebLoader WITHOUT .with_epoch(ipe). The trainer's outer loop
    # (`for itr in range(ipe)`) drives epoch boundaries; the loader is purely
    # infinite (sources are resampled=True). Adding .with_epoch makes the
    # iterator raise StopIteration after ipe batches; at 192 ranks, the
    # subsequent `iter(unsupervised_loader)` re-entry hangs after the first
    # successful re-iter — phase-1 job 8527315 (2026-06-06) hung permanently
    # at the 2nd epoch boundary and wasted ~4h of walltime.
    data_loader = wds.WebLoader(
        mixed,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    class _LenWrapper:
        def __init__(self, loader, length):
            self.loader = loader
            self.length = length

        def __iter__(self):
            return iter(self.loader)

        def __len__(self):
            return self.length

        def __getattr__(self, name):
            return getattr(self.loader, name)

    data_loader = _LenWrapper(data_loader, ipe)
    logger.info(f"WebDataset loader ready: batches_per_rank={ipe}, batch_size={batch_size}")

    sampler = _NoOpSampler()
    return mixed, data_loader, sampler
