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

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint
    return model
