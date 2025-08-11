# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import numpy as np
import pytest
import torch

from src.models.vision_transformer import vit_giant_xformers_rope


# Usage: pytest tests/models/test_vision_transformer.py
def is_cuda_or_xpu_available():
    if torch.cuda.is_available():
        return True
    try:
        import intel_extension_for_pytorch as ipex
        if ipex.xpu.is_available():
            return True
    except ImportError:
        pass
    return False

@pytest.mark.skipif(not is_cuda_or_xpu_available(), reason="This test requires CUDA or XPU")
class TestViTGiant(unittest.TestCase):
    def setUp(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
        else:
            self.device = torch.device("xpu")
            self.device_type = "xpu"

        self.model_shape_invariant = vit_giant_xformers_rope(
            img_size=256, patch_size=16, num_frames=16, handle_nonsquare_inputs=True
        ).to(self.device)
        self.model_square = deepcopy(self.model_shape_invariant)
        self.model_square.handle_nonsquare_inputs = False
        torch.manual_seed(42)
        self.total_iters = 10

    def test_square_inputs(self):
        for i in range(self.total_iters):
            input = torch.rand(1, 3, 16, 256, 256).to(self.device)
            with torch.autocast(device_type=self.device_type, enabled=True):
                with torch.no_grad():
                    out1 = self.model_shape_invariant(input)
                    out2 = self.model_square(input)
                    torch.testing.assert_close(out1, out2)

    def test_square_inputs_with_mask(self):
        for i in range(self.total_iters):
            input = torch.rand(1, 3, 16, 256, 256).to(self.device)
            mask = torch.randint(0, 2, (1, 2048)).to(self.device)
            with torch.autocast(device_type=self.device_type, enabled=True):
                with torch.no_grad():
                    out1 = self.model_shape_invariant(input, masks=mask)
                    out2 = self.model_square(input, masks=mask)
                    torch.testing.assert_close(out1, out2)

    def test_nonsquare_inputs(self):
        for i in range(self.total_iters):
            rand_width = np.random.randint(256, 512)
            rand_height = np.random.randint(256, 512)
            input = torch.rand(1, 3, 16, rand_height, rand_width).to(self.device)
            # Since input is interpolated, output won't be exactly the same
            input_resized_to_square = [
                torch.nn.functional.interpolate(input[:, :, frame_idx], size=256, mode="bicubic")
                for frame_idx in range(input.shape[2])
            ]
            input_resized_to_square = torch.stack(input_resized_to_square, dim=2)

            with torch.autocast(device_type=self.device_type, enabled=True):
                with torch.no_grad():
                    out1 = self.model_shape_invariant(input).mean(dim=1)
                    out2 = self.model_square(input_resized_to_square).mean(dim=1)
                    self.assertAlmostEqual(torch.nn.functional.cosine_similarity(out2, out1).item(), 1.0, places=3)
