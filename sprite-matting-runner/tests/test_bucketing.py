import numpy as np

from sprite_matting.batching import bucketize


def _make_crop(h, w):
    return {"rgb": np.zeros((h, w, 3), dtype=np.float32), "tri": np.zeros((h, w), dtype=np.float32)}


def test_bucketize_respects_budget():
    crops = [_make_crop(20, 10), _make_crop(15, 10), _make_crop(12, 10)]
    buckets = bucketize(crops, pixel_budget=400)
    assert len(buckets) == 2
    areas = [sum(c["rgb"].shape[0] * c["rgb"].shape[1] for c in bucket) for bucket in buckets]
    assert max(areas) <= 400
