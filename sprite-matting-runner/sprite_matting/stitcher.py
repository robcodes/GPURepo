"""Stitch per-crop alpha mattes into a full canvas."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def init_canvas(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.float32)


def stitch(alpha_canvas: np.ndarray, crop_alpha: np.ndarray, box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    h, w = crop_alpha.shape[:2]
    alpha_canvas[y0 : y0 + h, x0 : x0 + w] = np.maximum(
        alpha_canvas[y0 : y0 + h, x0 : x0 + w], crop_alpha
    )


def finalize_canvas(alpha_canvas: np.ndarray) -> np.ndarray:
    alpha_canvas = np.clip(alpha_canvas * 255.0, 0, 255).astype(np.uint8)
    return alpha_canvas
