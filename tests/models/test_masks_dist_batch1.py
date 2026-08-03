"""Per-rank batch_size=1 on the weight_distance_loss path.

This was a documented landmine: `compute_mask_distance` used a bare
`.squeeze()`, which at BS==1 collapsed the batch dim as well as the intended
size-1 middle dim, yielding (N_enc,) instead of (1, N_enc). The downstream
`d_ij.unsqueeze(2)` in app/vjepa_2_1/train.py then indexed the wrong axis and
crashed -- hence the "per-rank batch_size >= 2" rule in CLAUDE.md.

`masks_dist.py` now uses `.squeeze(1)`. These tests pin that fix, because it is
load-bearing for scaling out: at 3072 ranks, per-rank bs=1 is what keeps the
global batch at 3072 instead of 6144.
"""

import pytest
import torch

from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance

GRID = 8
N_PRED = 12
N_ENC = 10
DIM = 16


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_distance_shape_keeps_batch_dim(bs):
    """d_ij must be (BS, N_enc) for every batch size, including 1."""
    torch.manual_seed(0)
    masks_pred = [[torch.randint(0, GRID**3, (bs, N_PRED))]]
    masks_enc = [[torch.randint(0, GRID**3, (bs, N_ENC))]]
    d_ij = compute_mask_distance(masks_pred, masks_enc, GRID, False)[0][0]
    assert d_ij.shape == (bs, N_ENC)


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_weighted_loss_runs_at_batch_one(bs):
    """The exact op from train.py's d_weights branch must not raise at bs=1."""
    torch.manual_seed(0)
    masks_pred = [[torch.randint(0, GRID**3, (bs, N_PRED))]]
    masks_enc = [[torch.randint(0, GRID**3, (bs, N_ENC))]]
    d_ij = compute_mask_distance(masks_pred, masks_enc, GRID, False)[0][0]

    zij = torch.randn(bs, N_ENC, DIM)
    hij = torch.randn(bs, N_ENC, DIM)
    loss = (torch.abs(zij - hij) ** 1.0 * (1 / d_ij.unsqueeze(2).clamp_min(1.0))).mean()

    assert torch.isfinite(loss), "non-finite loss on the distance-weighted path"


def test_clamp_guards_coincident_tokens():
    """d_ij can legitimately be 0 when enc/pred share a grid cell -> 1/0 = inf."""
    d_ij = torch.zeros(1, N_ENC)
    w = 1 / d_ij.unsqueeze(2).clamp_min(1.0)
    assert torch.isfinite(w).all()
    assert torch.equal(w, torch.ones(1, N_ENC, 1))


def test_squeeze1_differs_from_bare_squeeze_at_batch_one():
    """Pins WHY the fix matters -- the regression is invisible at bs>=2."""
    x = torch.randn(1, 1, N_ENC)
    assert x.squeeze().shape == (N_ENC,)      # old: batch dim gone
    assert x.squeeze(1).shape == (1, N_ENC)   # new: batch dim kept
    y = torch.randn(2, 1, N_ENC)
    assert y.squeeze().shape == y.squeeze(1).shape == (2, N_ENC)
