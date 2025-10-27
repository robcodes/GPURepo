"""Utility helpers for producing quick visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


def save_strip(images: Iterable[np.ndarray], path: str | Path) -> Path:
    """Save a horizontal strip of images for quick inspection."""

    path = Path(path)
    imgs = [_ensure_three_channel(img) for img in images]
    if not imgs:
        raise ValueError("No images provided")
    max_height = max(img.shape[0] for img in imgs)
    padded = [
        _pad_to_height(img, max_height) for img in imgs
    ]
    strip = np.concatenate(padded, axis=1)
    if strip.dtype != np.uint8:
        strip = np.clip(strip * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return path


def _ensure_three_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def _pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    pad = height - img.shape[0]
    top = pad // 2
    bottom = pad - top
    return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
