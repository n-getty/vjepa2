# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    module_name,
    device,
    frames_per_clip,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
    unfreeze_last_n=0,
):
    """
    Build model and initialize from pretrained checkpoint.

    By default the encoder is fully FROZEN (eval mode, no grad) -- the standard
    linear/attentive/asformer probe setup. When ``unfreeze_last_n > 0`` the last
    N transformer blocks are made trainable (requires_grad=True + train mode) for
    end-to-end partial fine-tuning; everything else (patch_embed, earlier blocks,
    norms) stays frozen. This is an opt-in lever for end-to-end partial fine-tuning
    of the frozen-probe encoder; it is additive and OFF by default, so the frozen
    probe path is bit-identical when unfreeze_last_n == 0.

    API requirements for Encoder module:
      1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    """
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if unfreeze_last_n and unfreeze_last_n > 0:
        # The eval encoder is a ClipAggregation wrapper whose `.model` is the
        # VisionTransformer holding `.blocks` (nn.ModuleList). Reach the block
        # list, then re-enable grad + train mode on the last N blocks only.
        vit = getattr(model, "model", model)
        blocks = getattr(vit, "blocks", None)
        if blocks is None:
            raise ValueError(
                "unfreeze_last_n set but could not find `.blocks` on the encoder "
                f"({type(vit).__name__}); partial unfreeze not supported for this module."
            )
        n = min(int(unfreeze_last_n), len(blocks))
        trainable = 0
        for blk in blocks[-n:]:
            blk.train()
            for p in blk.parameters():
                p.requires_grad = True
                trainable += p.numel()
        logger.info(
            "PARTIAL UNFREEZE: last %d/%d encoder blocks trainable "
            "(%d params); patch_embed/early-blocks/norms stay frozen.",
            n, len(blocks), trainable,
        )
    print(model)
    return model
