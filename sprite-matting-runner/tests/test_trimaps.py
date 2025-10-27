import numpy as np

from sprite_matting.mask_utils import build_trimap


def test_build_trimap_basic():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    tri = build_trimap(mask, fg_px=1, bg_px=1)

    assert tri[2, 2] == 1.0
    assert tri[0, 0] == 0.0
    assert 0.0 < tri[1, 1] < 1.0
