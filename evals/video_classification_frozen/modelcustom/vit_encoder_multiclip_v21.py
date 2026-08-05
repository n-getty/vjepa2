"""V-JEPA 2.1 dual-modal encoder for vjepa2_polaris video_classification_frozen.

Same protocol as :mod:`vit_encoder_multiclip` (single encoder, ClipAggregation
wrapper with the SAR-RARP50 `preserve_clip_dim` flag), but builds the encoder
from ``app.vjepa_2_1.models.vision_transformer`` so V-JEPA 2.1 surgical
checkpoints (with patch_embed_img / norms_block / *_mod_embed) load with zero
missing or unexpected keys.

Used by the ASFormer probe on surg_2_1_v1 backbones. Drop-in: set
``model_kwargs.module_name`` in the eval yaml to
``evals.video_classification_frozen.modelcustom.vit_encoder_multiclip_v21``.
"""

import logging

import torch

import app.vjepa_2_1.models.vision_transformer as vit

from evals.video_classification_frozen.modelcustom.vit_encoder_multiclip import (
    ClipAggregation,
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info(f"Loading pretrained V-JEPA 2.1 model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)

    enc_kwargs = dict(model_kwargs["encoder"])
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    # V-JEPA 2.1 surgical ckpts need the dual-modal patch/embeddings.
    enc_kwargs.setdefault("img_temporal_dim_size", 1)
    enc_kwargs.setdefault("uniform_power", True)

    model = vit.__dict__[enc_model_name](
        img_size=resolution, num_frames=frames_per_clip, **enc_kwargs
    )

    pretrained_dict = checkpoint[enc_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")

    # HIERARCHICAL multi-level features (flag-gated, default OFF -> bit-identical).
    # When wrapper_kwargs.return_hierarchical is set, the V-JEPA-2.1 ViT returns the
    # concat of its 4 distillation layers ([11,23,37,47] for gigantic) along the
    # FEATURE axis, each passed through its own ALREADY-TRAINED norms_block ->
    # [B, N, 4*embed_dim]. This is the paper's native hierarchical head input (used
    # during distillation), reused here as a frozen-probe lever: richer multi-scale
    # tokens for the same frozen encoder. We pop the key so it isn't forwarded to
    # ClipAggregation, set the flag on the inner ViT, and widen the wrapper's
    # advertised embed_dim to 4x so the downstream head builds at the right width.
    return_hier = bool(wrapper_kwargs.pop("return_hierarchical", False))
    if return_hier:
        model.return_hierarchical = True
        n_levels = len(model.hierarchical_layers)
        logger.info(
            "HIERARCHICAL features ON: concat of %d distillation layers %s -> "
            "embed_dim %d x%d = %d",
            n_levels, model.hierarchical_layers, model.embed_dim, n_levels,
            model.embed_dim * n_levels,
        )

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    if return_hier:
        # forward now emits 4*D features; advertise that to the head builder.
        model.embed_dim = model.model.embed_dim * len(model.model.hierarchical_layers)
    del checkpoint
    return model
