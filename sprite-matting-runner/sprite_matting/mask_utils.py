"""Mask utilities including connected components and trimap creation."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


def connected_components(mask_bin: np.ndarray) -> List[Box]:
    """Return bounding boxes for connected components of a binary mask."""

    if mask_bin.ndim != 2:
        raise ValueError("Mask must be 2D")
    mask_u8 = (mask_bin > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes: List[Box] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area == 0:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def build_trimap(mask_bin: np.ndarray, fg_px: int, bg_px: int) -> np.ndarray:
    """Create a float32 trimap from a binary mask."""

    if mask_bin.ndim != 2:
        raise ValueError("Mask must be 2D")
    mask_u8 = (mask_bin > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    fg = mask_u8.copy()
    if fg_px > 0:
        fg = cv2.erode(fg, kernel, iterations=fg_px)

    bg = mask_u8.copy()
    if bg_px > 0:
        bg = cv2.dilate(bg, kernel, iterations=bg_px)
    bg = cv2.threshold(bg, 127, 255, cv2.THRESH_BINARY)[1]

    trimap = np.zeros_like(mask_u8, dtype=np.float32)
    trimap[bg == 0] = 0.0
    trimap[fg == 255] = 1.0
    unknown = (bg > 0) & (fg < 255)
    trimap[unknown] = 0.5
    return trimap
