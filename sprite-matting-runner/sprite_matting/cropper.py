"""Crop extraction utilities."""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from .mask_utils import build_trimap, connected_components


Crop = Dict[str, object]


def extract_crops(
    image_rgb: np.ndarray,
    mask_bin: np.ndarray,
    min_area: int,
    pad: int,
    tri_fg_px: int,
    tri_bg_px: int,
    max_crop_pixels: int,
) -> List[Crop]:
    """Slice sprite crops from the sheet."""

    h, w = mask_bin.shape
    boxes = connected_components(mask_bin)
    crops: List[Crop] = []

    if not boxes and mask_bin.any():
        boxes = [(0, 0, w, h)]

    for box in boxes:
        x0, y0, x1, y1 = box
        if (x1 - x0) * (y1 - y0) < max(min_area, 1):
            continue
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)

        rgb_crop = image_rgb[y0:y1, x0:x1]
        mask_crop = mask_bin[y0:y1, x0:x1]
        tri = build_trimap(mask_crop, tri_fg_px, tri_bg_px)
        crop_h, crop_w = rgb_crop.shape[:2]
        pixels = crop_h * crop_w
        ds = 1.0
        orig_hw = (crop_h, crop_w)
        if max_crop_pixels > 0 and pixels > max_crop_pixels:
            scale = np.sqrt(max_crop_pixels / float(pixels))
            if scale < 1.0:
                ds = float(scale)
                new_size = (max(1, int(round(crop_w * scale))), max(1, int(round(crop_h * scale))))
                rgb_crop = cv2.resize(rgb_crop, new_size, interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(mask_crop, new_size, interpolation=cv2.INTER_NEAREST)
                tri = build_trimap(mask_resized, tri_fg_px, tri_bg_px)
                crop_h, crop_w = rgb_crop.shape[:2]
        crops.append(
            {
                "box": (x0, y0, x1, y1),
                "ds": ds,
                "orig_hw": orig_hw,
                "rgb": rgb_crop.astype(np.float32),
                "tri": tri.astype(np.float32),
            }
        )
    return crops
