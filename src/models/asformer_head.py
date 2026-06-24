"""ASFormer-style head for frame-wise action segmentation on top of V-JEPA2.

Designed for the SAR_RARP50 action-recognition probe (see
`phase_triplet_heads_bundle/vjepa2_1/head_phases/AGENTS_SARRARP50_ACTION_BACKBONES.md`).

Input shape (from `ClipAggregation(preserve_clip_dim=True)`):
    x : [B, num_clips, T_clip * S, D]
        num_clips    = number of V-JEPA clips per sample (e.g. 3 for ctx3)
        T_clip       = temporal tokens per clip = frames_per_clip // tubelet_size
        S            = spatial tokens per clip
        D            = embed_dim

Pipeline:
    1. Reshape to [B, num_clips, T_clip, S, D].
    2. Attention-pool over S with one learned query per (clip, time) step
       (shared across positions) -> [B, num_clips, T_clip, D].
    3. Flatten (clip, time) -> [B, T_total, D] where T_total = num_clips * T_clip
       (24 with frames_per_clip=16, tubelet_size=2, num_clips=3).
    4. Encoder stack: one temporal self-attention block + N dilated 1-D
       temporal-convolution blocks with exponentially increasing dilations.
    5. Linear classifier -> [B, T_total, num_classes].

If `return_sequence=False`, the last step mean-pools over T_total to return
`[B, num_classes]` (pooled-clip mode; not the recommended training path).

This module is independent of `src/models/utils/modules.py` so it stays usable
even if upstream V-JEPA helpers change shape.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TemporalSelfAttentionBlock(nn.Module):
    """Standard transformer block applied over the temporal token axis."""

    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
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
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class DilatedTemporalConvBlock(nn.Module):
    """ASFormer-style dilated residual conv block over the temporal axis.

    x : [B, T, D]
    """

    def __init__(self, embed_dim: int, dilation: int, dropout: float = 0.0,
                 kernel_size: int = 3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(embed_dim)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.act = nn.GELU()
        self.proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        y = self.norm(x)
        y = y.transpose(1, 2)            # [B, D, T]
        y = self.act(self.conv(y))
        y = self.proj(y)
        y = y.transpose(1, 2)            # [B, T, D]
        return residual + self.dropout(y)


# ---------------------------------------------------------------------------
# ASFormer head
# ---------------------------------------------------------------------------


class ASFormerHead(nn.Module):
    """Frame-wise action head used on the SAR_RARP50 transfer benchmark.

    Args (commonly set from config):
        embed_dim:        V-JEPA backbone embed_dim (1024 for vit_large, 1408
                          for vit_giant_xformers).
        num_classes:      Number of action classes (8 for SAR_RARP50).
        num_clips:        Number of V-JEPA clips per sample (3 for ctx3).
        tokens_per_clip:  Temporal tokens per clip
                          = frames_per_clip // tubelet_size (8 for 16f / ts=2).
        num_layers:       Number of dilated temporal conv blocks.
        num_heads:        Attention-head count (for both pool + self-attn block).
        mlp_ratio:        Hidden expansion ratio in the self-attn block MLP.
        dropout:          Dropout for attention + MLP + conv blocks.
        return_sequence:  If True (default), return per-token logits
                          [B, T_total, num_classes]. If False, mean-pool over
                          T_total and return [B, num_classes].
        temporal_tokens:  Optional. If set, used as a sanity assert
                          (num_clips * tokens_per_clip == temporal_tokens).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_clips: int = 3,
        tokens_per_clip: int = 8,
        num_layers: int = 10,
        num_heads: int = 8,
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

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.tokens_per_clip = tokens_per_clip
        self.return_sequence = return_sequence
        self.temporal_tokens = num_clips * tokens_per_clip

        self.spatial_pool = SpatialAttentionPool(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
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

        self.temporal_self_attn = TemporalSelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Cap the dilation cycle to the longest dilation that still produces a
        # useful receptive field over `temporal_tokens`. With T_total=24, the
        # full receptive-field doubling saturates at dilation=16 (i=4); going
        # beyond that just produces zero-padded reads. MS-TCN-style: wrap the
        # cycle so deeper layers refine instead of pass-through.
        max_dilation_exp = max(1, math.ceil(math.log2(self.temporal_tokens)))
        self.temporal_blocks = nn.ModuleList(
            [
                DilatedTemporalConvBlock(
                    embed_dim=embed_dim,
                    dilation=2 ** (i % max_dilation_exp),
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Coerce the encoder output to [B, num_clips, T_clip, S, D]."""
        if x.dim() == 4:
            # [B, num_clips, T_clip*S, D] — preserve_clip_dim path.
            B, NC, TS, D = x.shape
            if NC != self.num_clips:
                raise ValueError(
                    f"ASFormerHead expected num_clips={self.num_clips}, got {NC}"
                )
            if TS % self.tokens_per_clip != 0:
                raise ValueError(
                    f"Token count {TS} not divisible by tokens_per_clip="
                    f"{self.tokens_per_clip}"
                )
            S = TS // self.tokens_per_clip
            return x.reshape(B, NC, self.tokens_per_clip, S, D)
        if x.dim() == 3:
            # [B, num_clips*T_clip*S, D] — legacy path, only works if S can be
            # inferred uniquely.
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
        raise ValueError(f"Unsupported ASFormer input rank: {x.dim()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_clips, T_clip*S, D] or [B, num_clips*T_clip*S, D]
        x = self._reshape_input(x)               # [B, NC, T_clip, S, D]
        B, NC, T_clip, S, D = x.shape

        # Pool spatial tokens per (clip, time) step.
        x = x.reshape(B * NC, T_clip, S, D)
        x = self.spatial_pool(x)                 # [B*NC, T_clip, D]
        x = x.reshape(B, NC * T_clip, D)         # [B, T_total, D]

        # Add positional info, run temporal self-attention + dilated convs.
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.temporal_self_attn(x)
        for block in self.temporal_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.classifier(x)              # [B, T_total, num_classes]

        if not self.return_sequence:
            logits = logits.mean(dim=1)          # [B, num_classes]
        return logits


__all__ = ["ASFormerHead"]
