"""I/O helpers for the matting pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1]."""

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_binary_mask(path: str | Path) -> np.ndarray:
    """Load a binary mask as uint8 {0,1}."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    mask = (mask > 127).astype(np.uint8)
    return mask


def save_alpha_png(path: str | Path, alpha: np.ndarray) -> None:
    """Save a single channel uint8 alpha PNG."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if alpha.dtype != np.uint8:
        raise ValueError("Alpha matte must be uint8")
    cv2.imwrite(str(path), alpha)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
