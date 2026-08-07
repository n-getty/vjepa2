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
import time
from logging import getLogger

import numpy as np
import torch
import torchvision
import webdataset as wds
from decord import VideoReader, cpu

from src.datasets.shard_window import shards_for_node

_GLOBAL_SEED = 0
logger = getLogger()

# One-shot flag so the "no label member -> default 0" path (label-free corpora like PE-Video)
# warns once per process instead of once per sample. See decode().
_WARNED_MISSING_LABEL = False

# Default per-pixel std (on the raw 0-255 uint8 buffer) below which a decoded
# clip is treated as degenerate (e.g. a pure-black or frozen frame). ~19% of
# surgvu24 clips are one byte-identical pure-black mp4 (decoded std == 0.0);
# at the realized mix that was ~4.4% of every training batch, a degenerate
# masked-prediction target that drives collapse pressure compounding with
# epochs. We reject any clip whose decoded std is below this floor. The floor
# is well below any genuine surgical clip (the most static real source,
# surgvu24's non-black clips, sits far above 1.0) so real data is never
# dropped. Override via VJEPA_MIN_CLIP_STD.
_DEFAULT_MIN_CLIP_STD = float(os.environ.get("VJEPA_MIN_CLIP_STD", "1.0"))

# --- Batch-source / clip-variance instrumentation -------------------------
# Per-process (DataLoader worker) diagnostics. Two purposes:
#   1. Mechanism-check: log the source + raw per-pixel std of the first
#      VJEPA_LOG_FIRST_N decoded clips so a corrupted source (e.g. surgvu24's
#      pure-black clips, std==0) is visible at training start.
#   2. Permanent tripwire: aggregate kept/dropped counts per source so the
#      drop rate (degenerate-clip prevalence) is observable in logs over a run.
# Workers are separate processes; each logs independently. ALL logging is
# restricted to the canonical worker (global rank 0, worker 0) — otherwise the
# 192 ranks x num_workers processes each emit, flooding the shared training log
# (observed: surgvu24's ~20-40% drop rate made every dropped clip log from all
# 384 workers). Every worker still keeps its own in-memory drop/keep tally
# (cheap, no I/O); only the canonical worker prints. Drops are logged at a
# coarse milestone (first drop + every VJEPA_DIAG_EVERY) at INFO — degenerate
# clips are EXPECTED and handled, not a warning condition.
_LOG_FIRST_N = int(os.environ.get("VJEPA_LOG_FIRST_N", "64"))
_DIAG_EVERY = int(os.environ.get("VJEPA_DIAG_EVERY", "500"))
_clip_diag = {"seen": 0, "dropped": {}, "kept": {}}

# --- Per-source decode timing (VJEPA_DECODE_PROFILE=1, default OFF) --------
# The 1n->8n scaling loss is an order statistic over a heavy per-rank dataload
# tail: the per-rank distribution is IDENTICAL at every rung (p50 1.2 s, p99
# ~22 s, p(>10 s) 0.098) and only max-over-ranks grows. Reducing that tail is
# the only lever that changes the asymptote, so the tail's OWNER has to be
# identified rather than guessed.
#
# Two candidates have now been tested offline and BOTH fail to explain the tail:
#
#   payload size   REFUTED outright. grasp_noleak is the largest source on disk
#                  (96.6 MB/clip) and among the fastest to decode (0.56 s);
#                  surgvu24_clean is 10x smaller and 6x slower. Pearson r vs
#                  decode time is +0.20. The near-match that made it persuasive
#                  -- p(clip from a >30 MB/clip source) = 0.104 vs measured
#                  p(dataload > 10 s) = 0.098 -- was a coincidence.
#   keyframe spacing (GOP)  REAL, but only for the BODY of the distribution.
#                  Re-encoding the same clip at GOP-16 buys 4.4-8.8x at
#                  unchanged pixels, and a mixture model over the measured
#                  per-source decode costs reproduces the median (predicted
#                  1.38 s vs observed 1.20) and the mean (2.66 vs 3.10). It
#                  does NOT reproduce the tail: predicted p99 8.6 s against an
#                  observed 23.0 s, predicted p(>10 s) 0.002 against 0.095.
#
# The decisive number is a ceiling. At bs=2, a pure-decode story cannot exceed
# twice the slowest per-clip decode ever measured (lapgyn6_events, 5.37 s), i.e.
# 10.74 s -- yet 8.7% of 18,252 observed samples are above it, median 15.1 s and
# max 67.8 s, 6.3x the ceiling. No mixture of measured decode costs can produce
# those draws. The excess appears at ONE node, so it is not fabric, and offline
# benchmarks structurally cannot see it because they do not read through DAOS.
#
# This records, per source, the wall time of the decode call itself and the
# gap since the previous sample left this worker. The split is the point:
#   * decode_ms  -- demux + frame extraction, i.e. CPU-side payload cost;
#   * gap_ms     -- time upstream (tar read / DAOS / shuffle buffer) to produce
#                   the next sample, i.e. storage-side cost.
# A payload story predicts the tail lives in decode_ms and tracks MB/clip. A
# storage story predicts it lives in gap_ms and does NOT sort by source. They
# are distinguishable, which is the whole reason to split them.
#
# OFF by default. When on, profile the first VJEPA_DECODE_PROFILE_RANKS ranks
# rather than rank 0 alone. Rank-0-only would very likely have measured nothing:
# a >ceiling event hits a median of 2 of 12 ranks in an iteration, so rank 0 is
# in the stalling set roughly one time in six and a whole rung could come back
# clean while the tail was happening two ranks over. The cap is what keeps this
# from becoming the log funnel that cost job 8730678 its MASTER_ADDR host -- 12
# ranks emitting one table per 200 samples is bounded and small; 3072 ranks
# emitting per-sample lines is not. Default 12 = one node's worth.
_DECODE_PROFILE = os.environ.get("VJEPA_DECODE_PROFILE", "0") == "1"
_DECODE_EVERY = int(os.environ.get("VJEPA_DECODE_PROFILE_EVERY", "200"))
_DECODE_RANKS = int(os.environ.get("VJEPA_DECODE_PROFILE_RANKS", "12"))
_decode_prof = {"n": 0, "last_exit": None, "by_src": {}}

# VJEPA_SHARD_CAP: restrict each NODE to a capped window of every source's
# shard list, computed by the same function the stager uses
# (src/datasets/shard_window.py). 0 = off, and off is the only correct setting
# for a training run -- a cap means the run sees a fixed subset of each source.
#
# It exists for exactly one experiment: the staged-vs-DAOS arm of the dataload
# tail study. Staging the S/N window at small node counts is ~90 min of copying
# (half the corpus per node at N=2), so the staged arm has to be capped; and a
# capped staged arm vs an uncapped DAOS arm moves the storage path AND the
# working-set size together. This knob supplies the missing control -- a DAOS
# arm reading the IDENTICAL capped window -- so the three-arm comparison
# daos-full / staged-capped / daos-capped separates the storage path from
# page-cache reuse.
#
# Node identity comes from PALS_*; with WDS_LOCAL_SLICING the node's local
# ranks then slice this window among themselves, exactly as they slice a staged
# dir.
_SHARD_CAP = int(os.environ.get("VJEPA_SHARD_CAP", "0"))
_SHARD_CAP_MIN = int(os.environ.get("VJEPA_SHARD_CAP_MIN", "24"))


def _is_profiling_worker():
    """True on worker 0 of the first _DECODE_RANKS ranks.

    Worker 0 only: `gap` is measured against module state that is per-process,
    so two workers in one process would interleave their exits and corrupt each
    other's gap. One worker per rank keeps the gap meaningful.
    """
    rank = os.environ.get("RANK", os.environ.get("PMI_RANK",
           os.environ.get("PALS_RANKID", "0")))
    try:
        if int(rank) >= _DECODE_RANKS:
            return False
    except (TypeError, ValueError):
        return False
    info = torch.utils.data.get_worker_info()
    return info is None or info.id == 0


def _record_decode(source_name, decode_s, gap_s):
    """Accumulate per-source decode/gap timings; emit a table periodically."""
    src = source_name or "?"
    d = _decode_prof["by_src"].setdefault(src, {"n": 0, "dec": [], "gap": []})
    d["n"] += 1
    # Bounded memory: keep a reservoir of the most recent samples per source.
    # Percentiles of the tail are the point, so a plain cap (not a mean) is
    # required -- a running mean would average the tail away.
    for k, v in (("dec", decode_s), ("gap", gap_s)):
        d[k].append(v)
        if len(d[k]) > 512:
            del d[k][0]
    _decode_prof["n"] += 1
    if _decode_prof["n"] == 1:
        # Announce on the first sample. Without this the instrument can be ON and
        # emit nothing, with no way to tell that from "the tail did not happen":
        # a ladder rung is 80 iters x bs=2 = 160 samples per rank, below the
        # default period of 200, so the first attempt at this measurement would
        # have produced an empty log and looked like a clean result.
        logger.info(
            "[decode-prof] ENABLED: table every %d samples, ranks < %d",
            _DECODE_EVERY, _DECODE_RANKS,
        )
    if _decode_prof["n"] % _DECODE_EVERY:
        return
    rows = []
    for s, v in _decode_prof["by_src"].items():
        dec, gap = sorted(v["dec"]), sorted(v["gap"])
        if not dec:
            continue
        q = lambda a, p: a[min(len(a) - 1, int(p * (len(a) - 1)))]  # noqa: E731
        rows.append((s, v["n"], q(dec, 0.5), q(dec, 0.99), max(dec),
                     q(gap, 0.5), q(gap, 0.99), max(gap)))
    rows.sort(key=lambda r: -r[4])
    logger.info(
        "[decode-prof] n=%d  src / n / decode p50,p99,max s / gap p50,p99,max s\n%s",
        _decode_prof["n"],
        "\n".join(
            "  %-24s %6d  %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f" % r for r in rows
        ),
    )


def _is_canonical_worker():
    """True only on global rank 0's first DataLoader worker (or main process)."""
    rank = os.environ.get("RANK", os.environ.get("PMI_RANK",
           os.environ.get("PALS_RANKID", "0")))
    if str(rank) != "0":
        return False
    info = torch.utils.data.get_worker_info()
    return info is None or info.id == 0


def _is_rank0():
    """True on global rank 0 (any worker). For setup-time logging only.

    Loader-construction logging was emitted by EVERY rank. At 16 nodes that is
    228 lines and nobody notices; at 3072 ranks it is ~58,000 lines funnelled
    into one PBS log on the head node -- which is also serving the rendezvous
    store and answering DAOS agent keepalives. Job 8730678 lost that node to a
    120 s DAOS ping timeout mid-run, and MASTER_ADDR being the affected host is
    unlikely to be coincidence.

    The diagnostics themselves are worth keeping (the mixing table is how the
    oversampling regression stays visible), so gate them to rank 0 rather than
    delete them. Distinct from _is_canonical_worker, which also requires worker
    0 -- that matters for per-sample paths, not for one-shot setup logging.
    """
    rank = os.environ.get("RANK", os.environ.get("PMI_RANK",
           os.environ.get("PALS_RANKID", "0")))
    return str(rank) == "0"


def _node_identity(rank, world_size):
    """(node_rank, num_nodes) for the VJEPA_SHARD_CAP window.

    Derived from the TORCH world (rank, world_size) and the local world size,
    NOT from PALS_RANKID/PALS_LOCAL_SIZE directly. The distinction matters
    inside scripts/scaling_ladder.sh: a rung runs a sub-world of the
    allocation, so PALS reports the rung's own MPI world -- correct -- but the
    trainer's rank is the authoritative one either way, and deriving from it
    keeps this consistent with app/vjepa_2_1/hsdp.py, which computes
    num_nodes = world_size // local_world_size for the mesh.

    Falls back to (0, 1) -- one node holding the whole window -- whenever the
    local world size is unknown or nonsensical. The cap then still applies, so
    a misdetected topology gives a smaller-than-intended read set rather than
    an exception mid-loader-construction.
    """
    try:
        lws = int(os.environ.get(
            "LOCAL_WORLD_SIZE", os.environ.get("PALS_LOCAL_SIZE",
            os.environ.get("PMI_LOCAL_SIZE", "0"))))
    except (TypeError, ValueError):
        lws = 0
    if rank is None or world_size is None or lws <= 0 or world_size < lws:
        return 0, 1
    return int(rank) // lws, max(1, int(world_size) // lws)


def _record_kept_clip(source_name, clip_std):
    src = source_name or "?"
    _clip_diag["kept"][src] = _clip_diag["kept"].get(src, 0) + 1
    if _clip_diag["seen"] < _LOG_FIRST_N and _is_canonical_worker():
        _clip_diag["seen"] += 1
        logger.info(
            "[clip-diag] #%d source=%-24s raw_std=%.3f keep "
            "(per-source kept=%d dropped=%d)",
            _clip_diag["seen"], src, clip_std,
            _clip_diag["kept"].get(src, 0), _clip_diag["dropped"].get(src, 0),
        )


def _record_dropped_clip(source_name, clip_std):
    src = source_name or "?"
    _clip_diag["dropped"][src] = _clip_diag["dropped"].get(src, 0) + 1
    # Canonical-worker only, coarse milestones (first drop + every _DIAG_EVERY)
    # so a high-prevalence corrupt source stays visible without flooding the log.
    if not _is_canonical_worker():
        return
    d = _clip_diag["dropped"][src]
    if d == 1 or d % _DIAG_EVERY == 0:
        k = _clip_diag["kept"].get(src, 0)
        total = d + k
        logger.info(
            "[clip-diag] source=%s cumulative dropped=%d kept=%d "
            "(%.1f%% degenerate)",
            src, d, k, 100.0 * d / total if total else 0.0,
        )


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
        min_clip_std=_DEFAULT_MIN_CLIP_STD,
    ):
        self.min_clip_std = float(min_clip_std)
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

    def decode(self, sample, frames_per_clip, source_name=None):
        """Decode a webdataset sample dict into (clips, label, clip_indices).

        ``source_name`` (the originating dataset, e.g. ``surgvu24``) is used
        only for diagnostics — black-clip rejection logging and the optional
        rank-0 batch-source instrumentation. It does not affect decoding.
        """
        try:
            # 1. label
            label_bytes = None
            for key in ("label.txt", "label", "txt", "cls", "class.txt", "class"):
                if key in sample:
                    label_bytes = sample[key]
                    break
            if label_bytes is None:
                # Label-free corpora (e.g. PE-Video, which ships only .mp4/.json with no .cls) are valid
                # for V-JEPA: SSL pretraining never reads the label (train.py's load_clips consumes only
                # udata[0][0], the clip tensor). So default to 0 rather than DROP the sample — dropping
                # would silently discard every PE-Video clip. Warn ONCE per process, not per sample, to
                # flag the case without flooding logs. Datasets with real labels still carry .cls and are
                # unaffected; downstream probes read labels via separate CSV loaders, not this path.
                global _WARNED_MISSING_LABEL
                if not _WARNED_MISSING_LABEL:
                    available_keys = [k for k in sample.keys() if not k.startswith("__")]
                    logger.warning(
                        f"No label member for key {sample.get('__key__', 'N/A')} (available: "
                        f"{available_keys}); defaulting label=0 (label-free corpus, expected for "
                        f"PE-Video). Warning suppressed for subsequent samples."
                    )
                    _WARNED_MISSING_LABEL = True
                label = 0
            else:
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

            # 3b. Degenerate-clip reject (run on the RAW 0-255 buffer, BEFORE
            # transforms — normalization would erase the black signal). A clip
            # whose per-pixel std is below the floor is pure-black / frozen and
            # is a degenerate masked-prediction target. We drop it (return None
            # -> .select(is_not_none) skips it -> resampled stream draws the
            # next sample). Skip for images (single repeated frame -> 0 temporal
            # variance is expected and benign for the image branch).
            if not is_image and self.min_clip_std > 0.0:
                arr = np.asarray(buffer, dtype=np.float32)
                clip_std = float(arr.std())
                # NON-FINITE GATE (2026-07-05): a corrupt clip with NaN/Inf pixels gives
                # std()=nan, and `nan < min_clip_std` is FALSE — so it would be KEPT and feed
                # NaN straight into the model (this crashed the 2B at epoch 16, rank 180 only:
                # loss->nan->allreduce->assert). np.isfinite catches BOTH the nan/inf-std case
                # and, defensively, any non-finite pixel. Checked before the low-variance test.
                if (not np.isfinite(clip_std)) or (not np.isfinite(arr).all()):
                    key = sample.get("__key__", "N/A")
                    logger.warning(
                        "Dropping NON-FINITE clip (std=%s) source=%s key=%s "
                        "— corrupt pixels (NaN/Inf), would poison the loss.",
                        clip_std, source_name or "?", key,
                    )
                    _record_dropped_clip(source_name, -1.0)
                    return None
                if clip_std < self.min_clip_std:
                    key = sample.get("__key__", "N/A")
                    logger.warning(
                        "Dropping degenerate clip (std=%.4f < %.4f) "
                        "source=%s key=%s — likely black/frozen frame.",
                        clip_std, self.min_clip_std, source_name or "?", key,
                    )
                    _record_dropped_clip(source_name, clip_std)
                    return None
                _record_kept_clip(source_name, clip_std)

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

    def __init__(self, decoder, fpc, source_name=None):
        self.decoder = decoder
        self.fpc = fpc
        self.source_name = source_name

    def __call__(self, sample):
        if not (_DECODE_PROFILE and _is_profiling_worker()):
            return self.decoder.decode(sample, self.fpc, source_name=self.source_name)
        # gap = time since the PREVIOUS sample left this worker, i.e. everything
        # upstream of us (tar read / DAOS / shuffle buffer refill). decode = our
        # own demux+extract. Splitting them is what separates a payload-cost tail
        # from a storage-stall tail; see _record_decode.
        #
        # READ THE TWO COLUMNS DIFFERENTLY DEPENDING ON num_workers.
        #   decode  always clean. It is one call, in this process, either way.
        #   gap     clean at nw>0 only. With workers, this process does nothing
        #           but decode in a loop, so a gap IS upstream wait. At nw=0
        #           decode runs inline in the training process, so the gap across
        #           a batch boundary also contains the entire training step
        #           (~3.4 s of compute here) and is NOT a storage measurement.
        #           At bs=2 that makes the nw=0 gap bimodal by construction:
        #           within-batch gaps are real, across-batch gaps are compute.
        # So the discriminator is run as a PAIR of rungs. If the tail lands in
        # `decode` at nw=0, it is codec cost and we are done -- gap is irrelevant.
        # If it does not, the tail is upstream, and only the nw=2 rung can say
        # whether upstream means storage.
        _t0 = time.time()
        _gap = 0.0 if _decode_prof["last_exit"] is None else _t0 - _decode_prof["last_exit"]
        out = self.decoder.decode(sample, self.fpc, source_name=self.source_name)
        _t1 = time.time()
        _decode_prof["last_exit"] = _t1
        _record_decode(self.source_name, _t1 - _t0, _gap)
        return out


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

    shard_names = meta["shard_urls"]
    _n_source = len(shard_names)
    _capped_to = 0
    if _SHARD_CAP > 0:
        node_rank, num_nodes = _node_identity(rank, world_size)
        idx = shards_for_node(_n_source, node_rank, num_nodes,
                              min_shards=_SHARD_CAP_MIN, max_shards=_SHARD_CAP)
        if idx:
            shard_names = [shard_names[i] for i in idx]
            _capped_to = len(shard_names)
    urls = [os.path.join(dataset_dir, u) for u in shard_names]
    _n_total = len(urls)
    _sliced = False
    if slice_rank is not None and slice_ws is not None and slice_ws > 1:
        if len(urls) >= slice_ws:
            urls = urls[slice_rank::slice_ws]
            _sliced = True
        # else: keep full list — tiny dataset, every rank uses all shards.
        nodesplitter = None
    else:
        nodesplitter = wds.split_by_node
    # Make the unsliced branch observable. It is silent by construction: a source
    # with fewer shards than slice_ws is handed to every rank IN FULL, so ranks
    # overlap on it with no error and no signal in the loss. At 3072 global ranks
    # all 16 sources fall below the threshold. Rank-0-gated (see _is_rank0) --
    # one line per source, not one per rank per source.
    if _is_rank0():
        cap_note = (
            f" cap={_SHARD_CAP}({_n_source}->{_capped_to} on this node)"
            if _capped_to else ""
        )
        logger.info(
            f"[wds-slice] source={meta.get('name')} shards={_n_total}{cap_note} "
            f"slice_rank={slice_rank} slice_ws={slice_ws} "
            f"sliced={_sliced} per_rank_urls={len(urls)}"
            + ("" if _sliced else "  <-- UNSLICED: every rank sees all shards")
        )
    stream = wds.WebDataset(
        urls,
        resampled=True,
        shardshuffle=True,
        nodesplitter=nodesplitter,
        workersplitter=wds.split_by_worker,
        handler=wds.warn_and_continue,
    ).shuffle(shuffle_buffer).map(
        _PerSampleDecode(decoder, fpc, source_name=meta.get("name"))
    ).select(is_not_none)
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
    min_clip_std=_DEFAULT_MIN_CLIP_STD,
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
        min_clip_std=min_clip_std,
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

    if _is_rank0():
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
        _log0 = _is_rank0()
        if _log0:
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
            # The per-source table is the bulk of the volume: one line per
            # source per rank. reps_per_clip is still built on EVERY rank so the
            # tripwire below stays a collective check.
            if _log0:
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
        # NOTE the RuntimeError above is deliberately NOT rank-gated: every rank
        # must raise, or the ranks that stay silent hang the collective.
        for name, cnt, p, reps in reps_per_clip:
            if reps > warn_reps and _log0:
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
    # wds.WebLoader wraps a torch DataLoader (super().__init__(DataLoader(**kw))), which
    # REJECTS prefetch_factor when num_workers==0 (ValueError). Make it conditional so a
    # workerless run (VJEPA_NUM_WORKERS=0 — the Mode-A shm-crash isolator) is valid.
    loader_kwargs = dict(
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(os.environ.get("VJEPA_PREFETCH_FACTOR", "2"))
    data_loader = wds.WebLoader(mixed, **loader_kwargs)

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
    if _is_rank0():
        logger.info(f"WebDataset loader ready: batches_per_rank={ipe}, batch_size={batch_size}")

    sampler = _NoOpSampler()
    return mixed, data_loader, sampler
