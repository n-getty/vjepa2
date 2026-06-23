"""Regression guard for the XPU `_sdpa` layout bug.

History: the Aurora port wrapped F.scaled_dot_product_attention with a
transpose to BSHD ([B, N, H, D]) on XPU, believing FlashAttentionXPU "requires"
that layout. But SDPA attends over dim -2 by contract, so feeding it BSHD made
it attend over the heads dim and treat the N sequence tokens as heads --
silently scrambling every attention output (cos ~0.02 vs reference) with no
error and a plausible-looking training loss. All Aurora V-JEPA 2.1 pretraining
ran through it. See app/vjepa_2_1/models/utils/modules.py::_sdpa.

These tests run on CPU (no XPU needed) and assert the invariant that broke:
attention is computed over the sequence axis, in BHND layout.
"""
import torch
import torch.nn.functional as F

from app.vjepa_2_1.models.utils.modules import _sdpa


def _reference_attention(q, k, v):
    # Explicit attention over the sequence axis (dim -2) of BHND.
    scale = q.shape[-1] ** -0.5
    attn = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
    return attn @ v


def test_sdpa_matches_reference_bhnd():
    torch.manual_seed(0)
    B, H, N, D = 2, 16, 128, 64
    q, k, v = (torch.randn(B, H, N, D) for _ in range(3))
    out = _sdpa(q, k, v)
    ref = _reference_attention(q, k, v)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.999, f"_sdpa diverged from reference attention (cos={cos:.4f})"


def test_old_transposed_path_would_be_wrong():
    """The exact transpose the buggy port used must NOT reproduce reference.

    This pins *why* the fix matters: if someone reintroduces the BSHD transpose,
    this test documents that it scrambles attention rather than just reshaping.
    """
    torch.manual_seed(0)
    B, H, N, D = 2, 16, 128, 64
    q, k, v = (torch.randn(B, H, N, D) for _ in range(3))
    ref = _reference_attention(q, k, v)
    bshd = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    cos = F.cosine_similarity(bshd.flatten(), ref.flatten(), dim=0).item()
    # N != H, so attending over the wrong axis must be visibly wrong.
    assert cos < 0.5, (
        f"transposed BSHD path unexpectedly matched reference (cos={cos:.4f}); "
        "the head/seq dims may be equal in this shape -- pick N != H"
    )
