"""Mamba-based head for frame-wise action segmentation on top of V-JEPA2.

ActionMamba / DiM-style multi-stage refinement decoder built on top of the
frozen V-JEPA backbone, designed as a drop-in alternative to
`src/models/asformer_head.py::ASFormerHead`.

Input shape (from `ClipAggregation(preserve_clip_dim=True)`):
    x : [B, num_clips, T_clip * S, D]
        num_clips    = number of V-JEPA clips per sample (3 for ctx3)
        T_clip       = temporal tokens per clip = frames_per_clip // tubelet_size
        S            = spatial tokens per clip
        D            = embed_dim

Pipeline:
    1. Reshape to [B, num_clips, T_clip, S, D].
    2. Attention-pool over S with a single learned query (shared across
       (clip, time) positions) -> [B, num_clips, T_clip, D].
    3. Flatten (clip, time) -> [B, T_total, D] where
       T_total = num_clips * T_clip (24 with 16f / ts=2 / ctx3).
    4. Stage 0 (encoder): linear-projected features + sinusoidal positional
       embedding pass through a stack of N Mamba blocks (Pre-LN + MLP).
       Selective SSM provides global receptive field via the recurrence —
       no dilated convs needed.
    5. Stages 1..(num_stages-1) (refinement): each stage takes the previous
       stage's *softmax probabilities* concatenated (projected back to
       embed_dim via a small linear) onto the encoder features, passes them
       through another N Mamba blocks, and produces refined logits.
    6. Linear classifier per stage -> [B, T_total, num_classes].

Return contract
---------------
Mirrors ``ASFormerHead`` exactly: returns the **final-stage** logits as a
single tensor of shape ``[B, T_total, num_classes]`` (or ``[B, num_classes]``
if ``return_sequence=False``). The multi-stage refinement happens *inside*
the head; only the last stage's logits are exposed.

This keeps the existing per-view CE loss path in ``eval.py`` unchanged
(``criterion(o.reshape(-1, o.shape[-1]).float(), labels.reshape(-1))``) and
keeps the smoothing-loss path compatible. Per-stage CE supervision can be
added later by changing the return type to a list; that requires
corresponding edits in ``eval.py`` to flatten/handle the list, which is
explicitly out-of-scope for the current ASFormer-comparable A/B.

Mamba / AMP note
----------------
``mamba_ssm.Mamba`` uses a custom CUDA kernel (selective_scan_cuda) that
historically can be picky about fp16 autocast. We therefore wrap the Mamba
call in ``torch.cuda.amp.autocast(enabled=False)`` and explicitly cast the
input to fp32 inside the block. The surrounding Linear/LayerNorm/MLP stays
under autocast, so the perf impact is small.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as e:  # pragma: no cover - env-check at import time
    raise ImportError(
        "MambaHead requires `mamba-ssm` and `causal-conv1d`. Install with:\n"
        "    pip install mamba-ssm causal-conv1d"
    ) from e


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SpatialAttentionPool(nn.Module):
    """Pool over the spatial-token axis with a single learned query.

    Input :  [B, T, S, D]
    Output:  [B, T, D]
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape
        x = x.reshape(B * T, S, D)
        x = self.norm(x)
        q = self.query.expand(B * T, -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.reshape(B, T, D)


class MambaBlock(nn.Module):
    """Pre-LN Mamba block with a residual MLP, applied over the temporal axis.

    x : [B, T, D]
    """

    def __init__(
        self,
        embed_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mamba = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        y = self.norm1(x)
        # selective_scan_cuda dislikes autocast(fp16); force fp32 inside the
        # SSM call but stay under autocast for everything else.
        with torch.cuda.amp.autocast(enabled=False):
            y = self.mamba(y.float())
        x = residual + self.drop1(y.to(residual.dtype))
        x = x + self.mlp(self.norm2(x))
        return x


class MambaStage(nn.Module):
    """A stack of MambaBlocks + final LayerNorm + per-token classifier.

    x : [B, T, D] -> logits [B, T, C]
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    embed_dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        logits = self.classifier(x)
        return x, logits


# ---------------------------------------------------------------------------
# Mamba head
# ---------------------------------------------------------------------------


class MambaHead(nn.Module):
    """Frame-wise action head — Mamba multi-stage refinement variant.

    Args (same names as ASFormerHead unless noted):
        embed_dim:        V-JEPA backbone embed_dim.
        num_classes:      Number of action classes.
        num_clips:        Number of V-JEPA clips per sample.
        tokens_per_clip:  Temporal tokens per clip
                          = frames_per_clip // tubelet_size.
        num_layers:       Number of Mamba blocks PER STAGE.
        d_state, d_conv, expand: Mamba-specific hyperparameters. Defaults
                          match the mamba-ssm reference (d_state=16, d_conv=4,
                          expand=2).
        num_stages:       Total stages (1 encoder + (num_stages-1)
                          refinement). Mirror ASFormer's 4-stage decoder.
        mlp_ratio:        Hidden expansion ratio in the Mamba block MLP.
        dropout:          Dropout in pooling, blocks, MLPs.
        return_sequence:  If True, return per-token logits [B, T_total, C].
                          If False, mean-pool over T_total -> [B, C].
        temporal_tokens:  Optional sanity-assert: must equal
                          num_clips * tokens_per_clip.

    Forward returns the FINAL-stage logits only (single tensor), matching
    ASFormerHead's return contract so the existing eval.py loss path is
    unchanged.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_clips: int = 3,
        tokens_per_clip: int = 8,
        num_layers: int = 10,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_stages: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        return_sequence: bool = True,
        temporal_tokens: int | None = None,
    ):
        super().__init__()
        if temporal_tokens is not None and temporal_tokens != num_clips * tokens_per_clip:
            raise ValueError(
                f"temporal_tokens={temporal_tokens} != num_clips*tokens_per_clip="
                f"{num_clips * tokens_per_clip}"
            )
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.tokens_per_clip = tokens_per_clip
        self.num_stages = num_stages
        self.return_sequence = return_sequence
        self.temporal_tokens = num_clips * tokens_per_clip

        # Reuse the same spatial-attention pooler as ASFormerHead so any
        # difference at training time is attributable to the temporal
        # backbone, not the spatial-pool head. Keep num_heads=8 (matches
        # ASFormer default).
        self.spatial_pool = SpatialAttentionPool(
            embed_dim=embed_dim, num_heads=8, dropout=dropout,
        )

        # Sinusoidal positional embedding over the full temporal axis.
        pe = torch.zeros(1, self.temporal_tokens, embed_dim)
        position = torch.arange(self.temporal_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_embed", pe, persistent=False)

        # Stage 0 ("encoder"): map pooled features -> initial logits.
        self.stage0 = MambaStage(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Refinement stages: each takes [prev_softmax_probs, encoder_feats]
        # and projects back to embed_dim. Following MS-TCN / ASFormer, we
        # concat the previous stage's softmax (not its raw logits) with the
        # original pooled features, then linearly project to embed_dim.
        self.refine_projs = nn.ModuleList()
        self.refine_stages = nn.ModuleList()
        for _ in range(num_stages - 1):
            self.refine_projs.append(
                nn.Linear(num_classes + embed_dim, embed_dim)
            )
            self.refine_stages.append(
                MambaStage(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    num_layers=num_layers,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Shape coercion (verbatim from ASFormerHead — keep the two heads
    # interchangeable so configs pointing at "preserve_clip_dim: true" work
    # for both).
    # ------------------------------------------------------------------

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, NC, TS, D = x.shape
            if NC != self.num_clips:
                raise ValueError(
                    f"MambaHead expected num_clips={self.num_clips}, got {NC}"
                )
            if TS % self.tokens_per_clip != 0:
                raise ValueError(
                    f"Token count {TS} not divisible by tokens_per_clip="
                    f"{self.tokens_per_clip}"
                )
            S = TS // self.tokens_per_clip
            return x.reshape(B, NC, self.tokens_per_clip, S, D)
        if x.dim() == 3:
            B, N, D = x.shape
            T_total = self.temporal_tokens
            if N % T_total != 0:
                raise ValueError(
                    "Got flat token sequence but token count "
                    f"({N}) is not divisible by temporal_tokens ({T_total}). "
                    "Set wrapper_kwargs.preserve_clip_dim: true in the YAML."
                )
            S = N // T_total
            return x.reshape(B, self.num_clips, self.tokens_per_clip, S, D)
        raise ValueError(f"Unsupported MambaHead input rank: {x.dim()}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)                       # [B, NC, T_clip, S, D]
        B, NC, T_clip, S, D = x.shape

        # Pool spatial tokens per (clip, time) step.
        x = x.reshape(B * NC, T_clip, S, D)
        feats = self.spatial_pool(x)                     # [B*NC, T_clip, D]
        feats = feats.reshape(B, NC * T_clip, D)         # [B, T_total, D]
        feats = feats + self.pos_embed[:, : feats.shape[1]]

        # Stage 0.
        _, logits = self.stage0(feats)                   # [B, T_total, C]

        # Refinement stages: project softmax(prev_logits) || feats -> D,
        # run a fresh MambaStage on top.
        for proj, stage in zip(self.refine_projs, self.refine_stages):
            prob = F.softmax(logits, dim=-1)             # [B, T_total, C]
            x_in = torch.cat([prob, feats], dim=-1)      # [B, T_total, C + D]
            x_in = proj(x_in)                            # [B, T_total, D]
            _, logits = stage(x_in)                      # [B, T_total, C]

        if not self.return_sequence:
            logits = logits.mean(dim=1)                  # [B, num_classes]
        return logits


__all__ = ["MambaHead"]
