"""Batch preparation utilities."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

Crop = Dict[str, object]


def bucketize(crops: Sequence[Crop], pixel_budget: int) -> List[List[Crop]]:
    """Greedy bin packing by pixel budget."""

    if pixel_budget <= 0:
        return [list(crops)] if crops else []

    sorted_crops = sorted(
        crops,
        key=lambda c: int(c["rgb"].shape[0]) * int(c["rgb"].shape[1]),
        reverse=True,
    )
    buckets: List[List[Crop]] = []
    bucket_pixels: List[int] = []
    for crop in sorted_crops:
        crop_h, crop_w = crop["rgb"].shape[:2]
        crop_pixels = crop_h * crop_w
        placed = False
        for idx, pixels in enumerate(bucket_pixels):
            if pixels + crop_pixels <= pixel_budget or not buckets[idx]:
                buckets[idx].append(crop)
                bucket_pixels[idx] += crop_pixels
                placed = True
                break
        if not placed:
            buckets.append([crop])
            bucket_pixels.append(crop_pixels)
    return buckets


def pad_stack_11(crops: Sequence[Crop], pad_multiple: int) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Stack crops into a padded tensor of handcrafted features."""

    if not crops:
        raise ValueError("No crops provided")

    h_max = max(int(crop["rgb"].shape[0]) for crop in crops)
    w_max = max(int(crop["rgb"].shape[1]) for crop in crops)
    h_pad = _pad_to_multiple(h_max, pad_multiple)
    w_pad = _pad_to_multiple(w_max, pad_multiple)

    batch = torch.zeros((len(crops), 11, h_pad, w_pad), dtype=torch.float32)
    sizes: List[Tuple[int, int]] = []
    for i, crop in enumerate(crops):
        rgb = crop["rgb"]
        tri = crop["tri"]
        h, w = rgb.shape[:2]
        sizes.append((h, w))
        batch[i, 0:3, :h, :w] = torch.from_numpy(rgb.transpose(2, 0, 1))
        batch[i, 3, :h, :w] = torch.from_numpy(tri)
        fg = (tri == 1.0).astype(np.float32)
        bg = (tri == 0.0).astype(np.float32)
        unknown = (tri == 0.5).astype(np.float32)
        batch[i, 4, :h, :w] = torch.from_numpy(fg)
        batch[i, 5, :h, :w] = torch.from_numpy(bg)
        batch[i, 6, :h, :w] = torch.from_numpy(unknown)
        unknown_dist = cv2.distanceTransform((unknown > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        if unknown_dist.max() > 0:
            unknown_dist = unknown_dist / unknown_dist.max()
        batch[i, 7, :h, :w] = torch.from_numpy(unknown_dist.astype(np.float32))
        sobel_x = cv2.Sobel(tri, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(tri, cv2.CV_32F, 0, 1, ksize=3)
        lap = cv2.Laplacian(tri, cv2.CV_32F, ksize=3)
        batch[i, 8, :h, :w] = torch.from_numpy(sobel_x)
        batch[i, 9, :h, :w] = torch.from_numpy(sobel_y)
        batch[i, 10, :h, :w] = torch.from_numpy(lap)
    return batch, sizes


def prepare_fba_tensors(
    crops: Sequence[Crop], pad_multiple: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """Return the four tensors required by the FBA model."""

    if not crops:
        raise ValueError("No crops provided")

    h_max = max(int(crop["rgb"].shape[0]) for crop in crops)
    w_max = max(int(crop["rgb"].shape[1]) for crop in crops)
    h_pad = _pad_to_multiple(h_max, pad_multiple)
    w_pad = _pad_to_multiple(w_max, pad_multiple)

    b = len(crops)
    images = torch.zeros((b, 3, h_pad, w_pad), dtype=torch.float32, device=device)
    tri_two = torch.zeros((b, 2, h_pad, w_pad), dtype=torch.float32, device=device)
    image_n = torch.zeros_like(images)
    tri_feat = torch.zeros((b, 6, h_pad, w_pad), dtype=torch.float32, device=device)
    sizes: List[Tuple[int, int]] = []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    for i, crop in enumerate(crops):
        rgb = crop["rgb"].astype(np.float32)
        tri = crop["tri"].astype(np.float32)
        h, w = rgb.shape[:2]
        sizes.append((h, w))
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)
        tri_t = torch.from_numpy(tri).to(device)
        images[i, :3, :h, :w] = rgb_t
        image_n[i, :3, :h, :w] = (rgb_t - mean) / std
        fg_np = (tri == 1.0).astype(np.float32)
        bg_np = (tri == 0.0).astype(np.float32)
        tri_two[i, 0, :h, :w] = torch.from_numpy(bg_np).to(device)
        tri_two[i, 1, :h, :w] = torch.from_numpy(fg_np).to(device)

        two_chan = np.stack((bg_np, fg_np), axis=2)
        tri_features = _trimap_transform(two_chan, max(h, w))
        tri_feat[i, :6, :h, :w] = torch.from_numpy(tri_features).to(device)
    return images, tri_two, image_n, tri_feat, sizes


def _pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return int(math.ceil(value / multiple) * multiple)


_SIGMA_SCALES = (0.02, 0.08, 0.16)


def _trimap_transform(two_chan: np.ndarray, length: int) -> np.ndarray:
    """Replicate FBA's trimap feature transform.

    Args:
        two_chan: Array of shape (H, W, 2) with background/foreground masks in {0, 1}.
        length: Spatial scale used in the Gaussian falloff.

    Returns:
        np.ndarray of shape (6, H, W) containing distance-based features.
    """

    h, w, _ = two_chan.shape
    L = max(int(length), 1)
    features = []
    for k in range(2):
        channel = two_chan[:, :, k]
        # Invert mask: distances are computed from the boundary of the region.
        inverted = 1.0 - channel
        dt_mask = -_distance_transform(inverted) ** 2
        for sigma in _SIGMA_SCALES:
            denom = 2.0 * ((sigma * L) ** 2)
            if denom <= 0:
                denom = 1.0
            features.append(np.exp(dt_mask / denom))
    return np.stack(features, axis=0).astype(np.float32)


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    return cv2.distanceTransform(mask_u8, cv2.DIST_L2, 0).astype(np.float32)
