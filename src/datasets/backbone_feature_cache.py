import json
import math
from logging import getLogger
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = getLogger()


class BackboneFeatureCacheDataset(Dataset):
    def __init__(self, cache_root, require_complete_export=True):
        self.cache_root = Path(cache_root)
        self.manifest_paths = self._discover_manifest_paths(self.cache_root)
        self.manifests = [self._read_manifest(path) for path in self.manifest_paths]
        self._validate_manifests(require_complete_export=require_complete_export)

        self.shards = []
        self.sample_index = []
        self.shard_ranges = []
        self.feature_shape = None

        for manifest_path, manifest in zip(self.manifest_paths, self.manifests):
            manifest_dir = manifest_path.parent
            if self.feature_shape is None and manifest.get("feature_shape_per_sample") is not None:
                self.feature_shape = tuple(manifest["feature_shape_per_sample"])

            for shard in manifest.get("shards", []):
                shard_path = manifest_dir / Path(shard["path"]).name
                if not shard_path.exists():
                    raise FileNotFoundError(f"Missing cache shard referenced by {manifest_path}: {shard_path}")

                num_samples = int(shard["num_samples"])
                if num_samples <= 0:
                    continue

                start = len(self.sample_index)
                shard_id = len(self.shards)
                self.shards.append(
                    {
                        "path": shard_path,
                        "num_samples": num_samples,
                    }
                )
                self.sample_index.extend((shard_id, offset) for offset in range(num_samples))
                self.shard_ranges.append((start, len(self.sample_index)))

        if not self.sample_index:
            raise RuntimeError(f"No cached samples found under {self.cache_root}")

        self._cached_shard_id = None
        self._cached_shard = None

    @staticmethod
    def _read_manifest(path):
        with open(path, "r") as handle:
            return json.load(handle)

    @staticmethod
    def _discover_manifest_paths(cache_root):
        direct_manifest = cache_root / "manifest.json"
        if direct_manifest.exists():
            return [direct_manifest]

        ranked_manifests = sorted(cache_root.glob("rank_*/manifest.json"))
        if ranked_manifests:
            return ranked_manifests

        raise FileNotFoundError(
            f"No cache manifest found under {cache_root}. "
            "Expected either <cache_root>/manifest.json or <cache_root>/rank_*/manifest.json."
        )

    def _validate_manifests(self, require_complete_export):
        exporter_world_size = None
        exporter_ranks = set()

        for manifest_path, manifest in zip(self.manifest_paths, self.manifests):
            if manifest.get("status") != "completed":
                raise RuntimeError(f"Cache manifest is not completed: {manifest_path}")

            manifest_world_size = int(manifest.get("world_size", 1))
            if exporter_world_size is None:
                exporter_world_size = manifest_world_size
            elif exporter_world_size != manifest_world_size:
                raise RuntimeError(
                    f"Inconsistent exporter world_size under {self.cache_root}: "
                    f"expected {exporter_world_size}, found {manifest_world_size} in {manifest_path}"
                )

            if "rank" in manifest:
                exporter_ranks.add(int(manifest["rank"]))

        if require_complete_export and exporter_world_size and exporter_world_size > 1:
            if len(self.manifest_paths) < exporter_world_size or len(exporter_ranks) < exporter_world_size:
                missing = sorted(set(range(exporter_world_size)) - exporter_ranks)
                raise RuntimeError(
                    f"Incomplete distributed cache under {self.cache_root}. "
                    f"Found {len(self.manifest_paths)} manifest(s) for exporter world_size={exporter_world_size}. "
                    f"Missing rank directories: {missing}. Copy all rank_* cache folders first."
                )

    def __len__(self):
        return len(self.sample_index)

    def _load_shard(self, shard_id):
        if self._cached_shard_id == shard_id and self._cached_shard is not None:
            return self._cached_shard

        shard_path = self.shards[shard_id]["path"]
        shard = torch.load(shard_path, map_location="cpu")

        if self.feature_shape is None:
            self.feature_shape = tuple(shard["features"].shape[1:])

        self._cached_shard_id = shard_id
        self._cached_shard = shard
        return shard

    def __getitem__(self, index):
        shard_id, offset = self.sample_index[index]
        shard = self._load_shard(shard_id)

        features = shard["features"][offset]
        label = int(shard["labels"][offset])
        row_index = int(shard["row_indices"][offset])
        sample_path = shard["sample_paths"][offset]
        return features, label, row_index, sample_path


class ShardOrderDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=False):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        dataset_size = len(self.dataset)
        if self.drop_last:
            self.num_samples = dataset_size // self.num_replicas
        else:
            self.num_samples = int(math.ceil(dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _ordered_indices(self):
        shard_ids = list(range(len(self.dataset.shard_ranges)))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(shard_ids), generator=generator).tolist()
            shard_ids = [shard_ids[i] for i in perm]

        ordered = []
        for shard_id in shard_ids:
            start, end = self.dataset.shard_ranges[shard_id]
            ordered.extend(range(start, end))
        return ordered

    def __iter__(self):
        indices = self._ordered_indices()
        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            if len(indices) < self.total_size:
                repeats = self.total_size - len(indices)
                multiplier = int(math.ceil(repeats / max(1, len(indices))))
                indices = indices + (indices * multiplier)[:repeats]

        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end])


def make_backbone_feature_cache(
    cache_root,
    batch_size,
    training,
    rank=0,
    world_size=1,
    num_workers=1,
    pin_mem=True,
    persistent_workers=True,
    require_complete_export=True,
):
    dataset = BackboneFeatureCacheDataset(
        cache_root=cache_root,
        require_complete_export=require_complete_export,
    )
    sampler = ShardOrderDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
        seed=0,
        drop_last=False,
    )
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info(
        "BackboneFeatureCache dataset created "
        f"(cache_root={cache_root}, samples={len(dataset)}, shards={len(dataset.shards)})"
    )
    return dataset, data_loader, sampler
