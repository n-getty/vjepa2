import json
import math
from logging import getLogger
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = getLogger()


def read_cache_pooled(cache_root):
    """Return the 'pooled' mode of a feature cache ("none" | "mean") without
    loading it, by peeking at one manifest. Used to tell the probe head whether
    the spatial axis was already pooled at export (so it must skip its own
    spatial pool). Returns "none" if no cache / no marker (back-compat with
    caches written before the pooled flag existed)."""
    if cache_root is None:
        return "none"
    root = Path(cache_root)
    candidates = [root / "manifest.json"] + sorted(root.glob("rank_*/manifest.json"))
    for m in candidates:
        if m.exists():
            try:
                with open(m) as f:
                    return json.load(f).get("pooled", "none") or "none"
            except (OSError, json.JSONDecodeError):
                continue
    return "none"


def _peek_manifest(cache_root):
    if cache_root is None:
        return None
    root = Path(cache_root)
    candidates = [root / "manifest.json"] + sorted(root.glob("rank_*/manifest.json"))
    for m in candidates:
        if m.exists():
            try:
                with open(m) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
    return None


def assert_cache_matches_encoder(cache_root, checkpoint_path, readout_layer):
    """Hard-fail if a feature cache was exported for a DIFFERENT checkpoint or
    readout_layer than the one this probe run is about to score against.

    A mid-layer readout cache is byte-shape-identical to a final-layer cache
    (same manifest keys otherwise) -- scripts/eval_grasp_map_cached.py even
    infers embed_dim from the cache itself, so nothing downstream would ever
    catch a silent mismatch (see frozen-probe diagnosis plan, Exp-2). Caches
    written before these fields existed have them absent (None), which reads
    as "old cache, unknown provenance" and is allowed through -- only an
    explicit, non-matching value is a hard fail.
    """
    manifest = _peek_manifest(cache_root)
    if manifest is None:
        return
    # KEY ABSENT (old cache, written before these fields existed) = unknown
    # provenance, allowed through. KEY PRESENT but None = new-format export
    # that explicitly recorded "no override" (baseline final-layer / no
    # checkpoint captured) -- that IS a known value and must still be compared,
    # or a mid-layer request against a baseline cache would silently pass.
    if "checkpoint_path" in manifest:
        cached_ckpt = manifest["checkpoint_path"]
        if cached_ckpt is not None and checkpoint_path is not None and cached_ckpt != checkpoint_path:
            raise RuntimeError(
                f"Cache staleness guard: {cache_root} was exported from checkpoint "
                f"{cached_ckpt!r} but this run requests {checkpoint_path!r}. Refusing "
                "to score a cache built from a different encoder."
            )
    if "readout_layer" in manifest:
        cached_layer = manifest["readout_layer"]
        if cached_layer != readout_layer:
            raise RuntimeError(
                f"Cache staleness guard: {cache_root} was exported with readout_layer="
                f"{cached_layer!r} but this run requests readout_layer={readout_layer!r}. "
                "A mid-layer cache is byte-shape-identical to a final-layer cache -- "
                "refusing to silently score the wrong layer."
            )


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
        # mmap=True: the 2.49 GB feature tensor is memory-mapped, not read up
        # front. Only the sample slices __getitem__ actually touches fault in,
        # and because every DataLoader worker mmaps the SAME file the kernel
        # page cache is shared across workers/processes -- so a shard's pages
        # load once, not once-per-worker. This is what removes the ~8x read
        # amplification (all workers used to torch.load the whole shard eagerly).
        # Requires the zip-format checkpoint (torch>=1.13); our shards are PK.
        shard = torch.load(shard_path, map_location="cpu", mmap=True)

        if self.feature_shape is None:
            self.feature_shape = tuple(shard["features"].shape[1:])

        # Drop the previous shard's mmap before caching the new one so at most
        # one shard is mapped per worker at a time (bounds RSS; the sampler
        # iterates shard-contiguously so this is a clean handoff, not thrash).
        self._cached_shard_id = shard_id
        self._cached_shard = shard
        return shard

    def __getitem__(self, index):
        shard_id, offset = self.sample_index[index]
        shard = self._load_shard(shard_id)

        # Clone the per-sample slice out of the mmap: it must be a real
        # in-memory tensor (a few hundred KB pooled / ~39 MB full), not a view
        # over the memory-mapped shard, or it would re-touch the map when the
        # DataLoader serializes it across the worker->main IPC boundary.
        features = shard["features"][offset].clone()
        # Scalar labels -> python int (classification). Per-clip sequence labels
        # (shape [T]) -> return the tensor as-is so default collate yields [B, T],
        # matching the live (non-cached) path for sequence_labels probes (asformer).
        raw_label = shard["labels"][offset]
        if torch.is_tensor(raw_label) and raw_label.ndim >= 1:
            label = raw_label
        else:
            label = int(raw_label)
        row_index = int(shard["row_indices"][offset])
        sample_path = shard["sample_paths"][offset]
        return features, label, row_index, sample_path


class ShardOrderDistributedSampler(Sampler):
    def __init__(
        self, dataset, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=False, subset_indices=None
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Low-shot subsetting: restrict iteration to a fixed set of global sample
        # indices (identical across ranks). Keeps the shard-contiguous access order
        # so the mmap shard-cache handoff in the dataset stays a clean sweep.
        self.subset_indices = None if subset_indices is None else set(int(i) for i in subset_indices)

        dataset_size = len(self.dataset) if self.subset_indices is None else len(self.subset_indices)
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
            if self.subset_indices is None:
                ordered.extend(range(start, end))
            else:
                ordered.extend(i for i in range(start, end) if i in self.subset_indices)
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
    train_frac=1.0,
    subset_seed=0,
):
    dataset = BackboneFeatureCacheDataset(
        cache_root=cache_root,
        require_complete_export=require_complete_export,
    )
    # Low-shot subsetting is applied to the TRAIN loader only. Compute the fixed
    # (rank-invariant) index set from the built cache length; None => full data.
    subset_indices = None
    if training:
        from src.datasets.video_dataset import seeded_subset_indices

        subset_indices = seeded_subset_indices(len(dataset), train_frac, subset_seed)
    # seed was hardcoded 0, so the cached head-train path shuffled identically
    # for every "seed" of a sweep -- on this path there is also no augmentation,
    # so head init was the ONLY thing that varied. Derive it from the probe seed.
    from src.utils.probe_seed import probe_seed

    sampler = ShardOrderDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
        seed=probe_seed(),
        drop_last=False,
        subset_indices=subset_indices,
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
    subset_note = "" if subset_indices is None else f", subset={len(sampler.subset_indices)}"
    logger.info(
        "BackboneFeatureCache dataset created "
        f"(cache_root={cache_root}, samples={len(dataset)}{subset_note}, shards={len(dataset.shards)})"
    )
    return dataset, data_loader, sampler
