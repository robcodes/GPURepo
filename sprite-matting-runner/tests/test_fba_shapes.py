import sys
import types

import numpy as np
import torch

from sprite_matting.config import Config
from sprite_matting.runners.fba_inprocess import run_fba_inprocess


class DummyModel(torch.nn.Module):
    def forward(self, image, two_chan_trimap, image_n, trimap_transformed):
        b, _, h, w = image.shape
        return {"alpha": torch.ones((b, 1, h, w), device=image.device) * 0.5}


def test_fba_runner_shapes(tmp_path):
    weights = tmp_path / "dummy.pth"
    torch.save(DummyModel().state_dict(), weights)

    networks = types.ModuleType("networks")
    models = types.ModuleType("networks.models")

    def build_model(*args, **kwargs):
        return DummyModel()

    models.build_model = build_model
    networks.models = models
    sys.modules["networks"] = networks
    sys.modules["networks.models"] = models

    try:
        cfg = Config(
            input_image="image.png",
            input_mask="mask.png",
            output_dir=str(tmp_path),
        )
        cfg.model.weights = str(weights)
        cfg.model.device = "cpu"
        cfg.model.amp = False

        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[16:48, 16:48] = 1.0
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1

        result = run_fba_inprocess(cfg, image, mask)
        alpha = result["alpha_full"]
        assert alpha.shape == mask.shape
        stats = result["stats"]
        assert stats["num_crops"] >= 1
        assert stats["num_batches"] >= 1
    finally:
        sys.modules.pop("networks", None)
        sys.modules.pop("networks.models", None)
