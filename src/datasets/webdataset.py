# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import math
import os
import pathlib
import warnings
from logging import getLogger
import json
import random

import numpy as np
import torch
import torchvision
from decord import VideoReader, cpu

import webdataset as wds
from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


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
        
        # Validation (same as original)
        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")
        
        # This will be set by the dataloader when it maps over the dataset
        self.dataset_fpcs = None

    def loadvideo_decord(self, video_bytes, fpc):
        """
        Load video content using Decord from a byte buffer.
        Replicates the logic from `VideoDataset.loadvideo_decord`.
        """
        
        # Check file size filter
        #if len(video_bytes) > self.filter_long_videos:
        #    logger.warning(f"skipping long video of size {len(video_bytes)} bytes")
        #    return [], None
        
        # Use io.BytesIO to treat the byte buffer as a file
        try:
            vr = VideoReader(io.BytesIO(video_bytes), num_threads=1, ctx=cpu(0))
        except Exception as e:
            logger.warning(f"Failed to open video with Decord: {e}")
            return [], None

        fstp = self.frame_step
        
        # Calculate frame step based on duration or fps if specified
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
                fstp = video_fps // self.fps
        
        assert fstp is not None and fstp > 0, "frame_step must be set"
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            logger.warning(f"skipping video of length {len(vr)}")
            return [], None

        vr.seek(0)

        # Partition video into equal sized segments and sample each clip
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
        """
        Load image content from bytes.
        Replicates the logic from `VideoDataset.get_item_image`.
        """
        try:
            # Decode image from bytes
            image_tensor = torchvision.io.decode_image(
                torch.frombuffer(image_bytes, dtype=torch.uint8),
                mode=torchvision.io.ImageReadMode.RGB
            )
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return [], None

        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expanding the input image [3, H, W] ==> [T, 3, H, W]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))  # [T, 3, H, W] ==> [T H W 3]
        buffer = buffer.numpy()

        return buffer, clip_indices

    def _get_dataset_name_from_url(self, url):
        """
        Extract dataset name from the shard URL.
        Assumes the URL structure is: /path/to/dataset_name/shard_xxxx.tar
        """
        if url is None:
            return None
        
        # Get the parent directory name
        path_obj = pathlib.Path(url)
        dataset_name = path_obj.parent.name
        return dataset_name

    def __call__(self, sample):
        """
        WebDataset map function.
        `sample` is a dict from webdataset, e.g.,
        { "__key__": "...", "video.mp4": b"...", "label.txt": b"0", "__url__": "..." }
        """
        try:
            # 1. Get dataset name from the shard URL
            shard_url = sample.get("__url__")
            if shard_url is None:
                logger.warning(f"No __url__ found in sample with key {sample.get('__key__', 'N/A')}")
                return None
            
            dataset_name = self._get_dataset_name_from_url(shard_url)
            
            if self.dataset_fpcs is None:
                logger.error("VideoDecoder.dataset_fpcs was not initialized by the loader!")
                return None

            if dataset_name not in self.dataset_fpcs:
                logger.warning(
                    f"Dataset name '{dataset_name}' from URL '{shard_url}' not found in fpc map. "
                    f"Available: {list(self.dataset_fpcs.keys())}"
                )
                return None
                
            frames_per_clip = self.dataset_fpcs[dataset_name]
            
            # 2. Decode label - check all possible label key formats
            label_bytes = None
            # Check for compound extensions first, then simple ones
            label_keys = ["label.txt", "label", "txt", "cls", "class.txt", "class"]
            for key in label_keys:
                if key in sample:
                    label_bytes = sample[key]
                    break
            
            if label_bytes is None:
                # Debug: print all available keys
                available_keys = [k for k in sample.keys() if not k.startswith("__")]
                logger.warning(
                    f"No label found for key {sample['__key__']}. "
                    f"Available keys: {available_keys}"
                )
                return None
            
            try:
                label = int(label_bytes.decode('utf-8').strip())
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to decode label for key {sample['__key__']}: {e}")
                return None

            # 3. Determine if this is an image or video and get media bytes
            is_image = False
            media_bytes = None
            media_key = None
            
            # Check for video formats (compound extensions first)
            video_keys = [
                "video.mp4", "video.avi", "video.mov", "video.webm", "video.mkv",
                "mp4", "avi", "mov", "webm", "mkv", "flv"
            ]
            for key in video_keys:
                if key in sample:
                    media_bytes = sample[key]
                    media_key = key
                    is_image = False
                    break
            
            # Check for image formats if no video found
            if media_bytes is None:
                image_keys = [
                    "image.jpg", "image.jpeg", "image.png", "image.bmp",
                    "jpg", "jpeg", "png", "bmp", "gif"
                ]
                for key in image_keys:
                    if key in sample:
                        media_bytes = sample[key]
                        media_key = key
                        is_image = True
                        break
            
            if media_bytes is None:
                # Debug: print all available keys (excluding metadata keys)
                available_keys = [k for k in sample.keys() if not k.startswith("__")]
                logger.warning(
                    f"No media (video/image) found for key {sample['__key__']}. "
                    f"Available keys: {available_keys}"
                )
                return None

            # 4. Load media
            if is_image:
                buffer, clip_indices = self.loadimage(media_bytes, frames_per_clip)
            else:
                buffer, clip_indices = self.loadvideo_decord(media_bytes, frames_per_clip)
            
            if len(buffer) == 0:
                logger.warning(f"Failed to load {media_key} media: {sample['__key__']}")
                return None

            # 5. Apply transforms (mirroring VideoDataset.get_item_video)
            def split_into_clips(video):
                fpc = frames_per_clip
                nc = self.num_clips
                return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

            if self.shared_transform is not None:
                buffer = self.shared_transform(buffer)
            
            # Only split into clips for videos, not images
            if not is_image:
                buffer = split_into_clips(buffer)
            else:
                buffer = [buffer]  # Wrap in list for consistency
            
            if self.transform is not None:
                buffer = [self.transform(clip) for clip in buffer]

            return buffer, label, clip_indices
        
        except Exception as e:
            logger.error(f"Error processing sample {sample.get('__key__', 'N/A')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None




class _WeightedShardSampler(torch.utils.data.IterableDataset):
    """
    Infinite Weighted Shard Sampler.
    Yields shards indefinitely based on dataset weights.
    """
    def __init__(self, dataset_metas, dataset_weights, rank, world_size, seed=0):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.dataset_names = [meta["name"] for meta in dataset_metas]
        self.dataset_shard_urls = [meta["urls"] for meta in dataset_metas]
        self.dataset_weights = dataset_weights
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # Worker isolation
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        # Create a distinct seed per rank AND per worker to ensuring unique shard coverage
        # epoch is included to shift the RNG state every training epoch
        rng_seed = self.seed + (self.rank * 1000) + worker_id + (self.epoch * 10000)
        rng = random.Random(rng_seed)

        while True:
            # 1. Pick a dataset based on weights
            # random.choices is fast and handles weights natively
            ds_idx = rng.choices(range(len(self.dataset_names)), weights=self.dataset_weights, k=1)[0]
            
            # 2. Pick a random shard from that dataset
            shards = self.dataset_shard_urls[ds_idx]
            if not shards: 
                continue
            
            url = rng.choice(shards)
            yield {"url": url}


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
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
):
    """
    Create a WebDataset-based data loader that replicates VideoDataset functionality.
    
    Note: MonitoredDataset is not supported with WebDataset as it's designed for
    map-style datasets. Resource monitoring should be done at the training loop level.
    """
    
    if not isinstance(data_paths, (list, tuple)):
        data_paths = [data_paths]
        
    if dataset_fpcs is None:
        dataset_fpcs = [frames_per_clip for _ in data_paths]
    else:
        if len(dataset_fpcs) != len(data_paths):
            raise ValueError("Frames per clip not properly specified for data paths")

    # 1. Load metadata for all datasets
    dataset_metas = []
    dataset_fpcs_map = {}
    total_shards = 0
    total_samples = 0

    for i, path in enumerate(data_paths):
        meta_path = os.path.join(path, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"metadata.json not found in {path}. "
                "Please run your WebDataset creation script to generate metadata."
            )
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Find all .tar files in the directory
        shard_urls = sorted([
            os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tar')
        ])
        meta["urls"] = shard_urls
        
        if len(shard_urls) != meta["shard_count"]:
            logger.warning(
                f"Metadata for {meta['name']} says {meta['shard_count']} shards, "
                f"but found {len(shard_urls)} .tar files. Using actual count."
            )
            meta["shard_count"] = len(shard_urls)
            
        dataset_metas.append(meta)
        total_shards += meta["shard_count"]
        total_samples += meta["sample_count"]
        
        # Map the dataset name to its frames_per_clip
        dataset_fpcs_map[meta['name']] = dataset_fpcs[i]

    logger.info(
        f"Loaded {len(dataset_metas)} WebDataset manifests. "
        f"Total shards: {total_shards}, Total samples: {total_samples}"
    )
    logger.info(f"Dataset FPC mapping: {dataset_fpcs_map}")
    
    # Estimate total number of shards to sample per epoch
    # Use a multiplier for better shuffling
    nshards_per_epoch = total_shards * 2
    logger.info(f"Setting nshards_per_epoch to {nshards_per_epoch}")

    # 2. Create the video decoder
    video_decoder = VideoDecoder(
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
    # Give the decoder the FPC map
    video_decoder.dataset_fpcs = dataset_fpcs_map

    # 3. Create the shard sampler and processing pipeline
    if datasets_weights is not None:
        logger.info("Using weighted sampling (_WeightedShardSampler)...")
        dist_sampler = _WeightedShardSampler(
            dataset_metas=dataset_metas,
            dataset_weights=datasets_weights,
            rank=rank,
            world_size=world_size,
            seed=_GLOBAL_SEED,
        )
    else:
        logger.info("Using uniform sampling (simple shard list)...")
        all_shard_urls = [url for meta in dataset_metas for url in meta["urls"]]
        
        # Create a simple iterable that yields shard URLs
        # For uniform sampling, we just cycle through all shards
        class UniformShardSampler(torch.utils.data.IterableDataset):
            def __init__(self, urls, rank, world_size, seed=0):
                self.urls = urls
                self.rank = rank
                self.world_size = world_size
                self.seed = seed
                self.epoch = 0
            
            def set_epoch(self, epoch):
                self.epoch = epoch
            
            def __iter__(self):
                # Shuffle URLs based on epoch
                rng = random.Random(self.seed + self.epoch)
                shuffled_urls = self.urls.copy()
                rng.shuffle(shuffled_urls)
                
                # Distribute across ranks
                for i, url in enumerate(shuffled_urls):
                    if i % self.world_size == self.rank:
                        yield {"url": url}
        
        dist_sampler = UniformShardSampler(
            urls=all_shard_urls,
            rank=rank,
            world_size=world_size,
            seed=_GLOBAL_SEED,
        )
    
    pipeline = wds.DataPipeline(
        dist_sampler,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(1000),  # <--- NEW: Buffer 1000 samples to smooth IO/randomness
        wds.map(video_decoder),
        wds.select(lambda x: x is not None),
    )

    # 4. Create the final DataLoader
    data_loader = wds.WebLoader(
        pipeline,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=False,  # Shuffling is handled by the shard sampler
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        prefetch_factor=2, # Load 4 batches ahead per worker (adjust as needed)
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    
    batches_per_rank = 1000 
    
    data_loader = data_loader.with_length(batches_per_rank)
    
    logger.info(f"WebDataset data loader created. Virtual Epoch Steps: {batches_per_rank}")

    return dist_sampler, data_loader, dist_sampler
