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
    tri_feat = torch.zeros((b, 5, h_pad, w_pad), dtype=torch.float32, device=device)
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
        fg = (tri_t == 1.0).float()
        bg = (tri_t == 0.0).float()
        tri_two[i, 0, :h, :w] = bg
        tri_two[i, 1, :h, :w] = fg
        unknown_mask = (tri_t == 0.5).float()
        unknown_dist = cv2.distanceTransform((tri == 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        if unknown_dist.max() > 0:
            unknown_dist = unknown_dist / unknown_dist.max()
        sobel_x = cv2.Sobel(tri, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(tri, cv2.CV_32F, 0, 1, ksize=3)
        lap = cv2.Laplacian(tri, cv2.CV_32F, ksize=3)
        tri_feat[i, 0, :h, :w] = unknown_mask
        tri_feat[i, 1, :h, :w] = torch.from_numpy(unknown_dist.astype(np.float32)).to(device)
        tri_feat[i, 2, :h, :w] = torch.from_numpy(sobel_x).to(device)
        tri_feat[i, 3, :h, :w] = torch.from_numpy(sobel_y).to(device)
        tri_feat[i, 4, :h, :w] = torch.from_numpy(lap).to(device)
    return images, tri_two, image_n, tri_feat, sizes


def _pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return int(math.ceil(value / multiple) * multiple)
