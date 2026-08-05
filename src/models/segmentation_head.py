"""Dense segmentation head for frozen V-JEPA2 features (SAR_RARP50 tools).

This is the spatial-probe counterpart to `asformer_head.ASFormerHead`. Where
the ASFormer head *pools away* the spatial token axis to produce per-frame
temporal action logits, this head *keeps* the H_p x W_p patch grid and predicts
a dense per-pixel class map — the localization-quality probe the existing probe
suite lacked.

Input shape (from `ClipAggregation(preserve_clip_dim=True)`, one spatial view):
    x : [B, num_clips, T_clip * S, D]
        num_clips   = V-JEPA clips per sample
        T_clip      = temporal tokens per clip = frames_per_clip // tubelet_size
        S           = spatial tokens per clip = H_p * W_p (24*24 = 576 @ 384/patch16)
        D           = embed_dim (1024 for vit_large)

Pipeline:
    1. Reshape to [B, num_clips, T_clip, H_p, W_p, D] (grid recovered from S).
    2. Fold (B, num_clips, T_clip) into the batch, move D to channels ->
       [B*num_clips*T_clip, D, H_p, W_p].
    3. Light conv decoder (Conv3x3-GN-ReLU stack) refines the coarse grid, then
       a 1x1 conv projects to `num_classes` and bilinear-upsamples to `out_hw`.
    4. Reshape back to a per-token sequence [B, T_total * H_out * W_out,
       num_classes] so it plugs straight into the existing `sequence_labels`
       cross-entropy path (which flattens logits to [-1, C] and labels to [-1]).

The decoder blocks are adapted from the vendored SurgeNet FPN
(`evals/.../modelcustom/_metaformer_surgenet.py`: Conv3x3GNReLU +
SegmentationHead), but fed a *single* scale (the last ViT block's token grid)
rather than a 4-level CNN pyramid, so no FPN is needed.

The label grid the dataset emits MUST match (T_total, H_out, W_out); see
`src/datasets/video_seg_dataset.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3GNReLU(nn.Module):
    """Conv3x3 - GroupNorm - ReLU, optional 2x bilinear upsample.

    Mirrors the SurgeNet decoder block so the probe's decoder capacity matches
    a known-good surgical segmentation recipe.
    """

    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False,
                 num_groups: int = 32):
        super().__init__()
        self.upsample = upsample
        # GroupNorm needs num_channels divisible by num_groups; fall back to a
        # divisor when decoder_channels isn't a multiple of 32.
        groups = num_groups
        if out_channels % groups != 0:
            groups = _largest_divisor(out_channels, num_groups)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


def _largest_divisor(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap (>=1). Keeps GroupNorm valid."""
    for g in range(min(cap, n), 0, -1):
        if n % g == 0:
            return g
    return 1


class SegmentationDecoderHead(nn.Module):
    """Dense per-pixel segmentation head on frozen ViT token grids.

    Args:
        embed_dim:        Backbone embed_dim (1024 vit_large, 1408 vit_giant).
        num_classes:      Segmentation classes incl. background (10 SAR_RARP50).
        num_clips:        V-JEPA clips per sample.
        tokens_per_clip:  Temporal tokens per clip = frames_per_clip//tubelet.
        grid_hw:          (H_p, W_p) patch grid = (resolution/patch_size)^2.
                          For 384/patch16 this is (24, 24) and S = 576.
        out_hw:           (H_out, W_out) output mask resolution. Logits are
                          bilinearly upsampled to this; the dataset downsamples
                          masks to the same size with nearest interpolation.
        decoder_channels: Width of the conv decoder.
        num_conv_blocks:  Number of Conv3x3-GN-ReLU refinement blocks.
        dropout:          Dropout2d before the final projection.
        supervise_all_frames: If False (default), only the *center* temporal
                          token of each clip is returned/supervised — SAR_RARP50
                          masks are 1 Hz so only one frame per clip is annotated.
                          If True, every temporal token is returned (dataset must
                          then provide a mask per temporal token).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_clips: int = 1,
        tokens_per_clip: int = 8,
        grid_hw: tuple[int, int] = (24, 24),
        out_hw: tuple[int, int] = (384, 384),
        decoder_channels: int = 256,
        num_conv_blocks: int = 2,
        dropout: float = 0.1,
        supervise_all_frames: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.tokens_per_clip = tokens_per_clip
        self.grid_hw = tuple(grid_hw)
        self.out_hw = tuple(out_hw)
        self.supervise_all_frames = supervise_all_frames
        self.grid_tokens = self.grid_hw[0] * self.grid_hw[1]

        blocks = [Conv3x3GNReLU(embed_dim, decoder_channels)]
        for _ in range(max(0, num_conv_blocks - 1)):
            blocks.append(Conv3x3GNReLU(decoder_channels, decoder_channels))
        self.decoder = nn.Sequential(*blocks)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=3,
                                    padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Coerce encoder output to [B, num_clips, T_clip, H_p, W_p, D]."""
        H_p, W_p = self.grid_hw
        if x.dim() == 4:
            # [B, num_clips, T_clip*S, D] — preserve_clip_dim path.
            B, NC, TS, D = x.shape
            if NC != self.num_clips:
                raise ValueError(
                    f"SegmentationDecoderHead expected num_clips={self.num_clips}, got {NC}"
                )
            if TS != self.tokens_per_clip * self.grid_tokens:
                raise ValueError(
                    f"Token count {TS} != tokens_per_clip*grid ({self.tokens_per_clip}*"
                    f"{self.grid_tokens}={self.tokens_per_clip * self.grid_tokens}). "
                    f"Check resolution/patch_size vs grid_hw={self.grid_hw}."
                )
            return x.reshape(B, NC, self.tokens_per_clip, H_p, W_p, D)
        if x.dim() == 3:
            # [B, num_clips*T_clip*S, D] — flattened path.
            B, N, D = x.shape
            per = self.tokens_per_clip * self.grid_tokens
            if N != self.num_clips * per:
                raise ValueError(
                    f"Flat token count {N} != num_clips*tokens_per_clip*grid "
                    f"({self.num_clips}*{per}). Set wrapper_kwargs.preserve_clip_dim: true."
                )
            return x.reshape(B, self.num_clips, self.tokens_per_clip, H_p, W_p, D)
        raise ValueError(f"Unsupported segmentation input rank: {x.dim()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_clips, T_clip*S, D]  ->  [B, NC, T_clip, H_p, W_p, D]
        x = self._reshape_input(x)
        B, NC, T_clip, H_p, W_p, D = x.shape

        if not self.supervise_all_frames:
            # Keep only the center temporal token per clip (the annotated frame).
            t_center = T_clip // 2
            x = x[:, :, t_center : t_center + 1]        # [B, NC, 1, H_p, W_p, D]
            T_keep = 1
        else:
            T_keep = T_clip

        T_total = NC * T_keep
        # Fold clip+time into batch, move D to channel dim for conv decoding.
        x = x.reshape(B * T_total, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()
        # -> [B*T_total, D, H_p, W_p]

        x = self.decoder(x)                            # [B*T_total, dec_ch, H_p, W_p]
        x = self.dropout(x)
        logits = self.classifier(x)                    # [B*T_total, C, H_p, W_p]
        logits = F.interpolate(logits, size=self.out_hw, mode="bilinear",
                               align_corners=False)    # [B*T_total, C, H_out, W_out]

        H_out, W_out = self.out_hw
        C = self.num_classes
        # -> [B, T_total*H_out*W_out, C] to match the sequence_labels CE path,
        # which flattens logits to [-1, C] and labels to [-1]. The dataset must
        # emit labels in the SAME (T_total, H_out, W_out) raster order.
        logits = logits.reshape(B, T_total, C, H_out, W_out)
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()   # [B, T_total, H, W, C]
        logits = logits.reshape(B, T_total * H_out * W_out, C)
        return logits


__all__ = ["SegmentationDecoderHead"]
