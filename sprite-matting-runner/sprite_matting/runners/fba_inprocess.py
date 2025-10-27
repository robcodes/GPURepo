"""In-process FBA matting runner."""

from __future__ import annotations

import importlib
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import cv2

from ..batching import bucketize, prepare_fba_tensors
from ..config import Config
from ..cropper import extract_crops
from ..stitcher import finalize_canvas, init_canvas, stitch


_ALPHA_KEYS = ("alpha", "pha", "alpha_pred", "pred_alpha")


def run_fba_inprocess(cfg: Config, image_rgb: np.ndarray, mask_bin: np.ndarray) -> Dict[str, object]:
    """Run the FBA model across all sprite crops."""

    if image_rgb.ndim != 3:
        raise ValueError("image_rgb must be HxWx3")
    if mask_bin.shape != image_rgb.shape[:2]:
        raise ValueError("Mask size mismatch")

    crops = extract_crops(
        image_rgb,
        mask_bin,
        cfg.sprite.min_area,
        cfg.sprite.pad,
        cfg.sprite.tri_fg_px,
        cfg.sprite.tri_bg_px,
        cfg.sprite.max_crop_pixels,
    )
    buckets = bucketize(crops, cfg.batching.pixel_budget)

    canvas_shape = image_rgb.shape[:2]
    if cfg.multi_gpu.enabled and len(cfg.multi_gpu.devices) >= 2:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for multi-GPU execution')
        from . import multi_gpu

        result = multi_gpu.run_dual_gpu(
            cfg,
            buckets,
            canvas_shape,
            devices=cfg.multi_gpu.devices,
        )
        result["alpha_full"] = finalize_canvas(result["alpha_full"])
        return result

    device = _resolve_device(cfg.model.device)
    result = _run_single_device(cfg, buckets, canvas_shape, device)
    result["alpha_full"] = finalize_canvas(result["alpha_full"])
    return result


def _run_single_device(
    cfg: Config,
    buckets: Sequence[Sequence[dict]],
    canvas_shape: Tuple[int, int],
    device: torch.device,
) -> Dict[str, object]:
    model = _load_fba_model(cfg.model.weights, device)
    model.eval()

    alpha_canvas = init_canvas(*canvas_shape)
    batch_times: List[float] = []
    num_crops = 0

    use_amp = cfg.model.amp and device.type == "cuda"
    start_wall = time.perf_counter()
    with torch.no_grad():
        for bucket in buckets:
            if not bucket:
                continue
            images, tri_two, image_n, tri_feat, sizes = prepare_fba_tensors(
                bucket, cfg.batching.pad_multiple, device
            )
            tick = time.perf_counter()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(images, tri_two, image_n, tri_feat)
            else:
                output = model(images, tri_two, image_n, tri_feat)
            tock = time.perf_counter()
            batch_times.append(tock - tick)
            alpha = _extract_alpha_tensor(output)
            if alpha is None:
                raise RuntimeError("FBA model did not return an alpha tensor")
            alpha = alpha.to(torch.float32).detach().cpu()
            for idx, crop in enumerate(bucket):
                h, w = sizes[idx]
                crop_alpha = _slice_alpha(alpha[idx], h, w)
                if crop.get("ds", 1.0) < 1.0:
                    orig_h, orig_w = crop["orig_hw"]
                    crop_alpha = cv2.resize(
                        crop_alpha, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
                    )
                crop_alpha = np.clip(crop_alpha, 0.0, 1.0)
                stitch(alpha_canvas, crop_alpha, crop["box"])
                num_crops += 1
    wall_time = time.perf_counter() - start_wall

    return {
        "alpha_full": alpha_canvas,
        "stats": {
            "num_crops": num_crops,
            "num_batches": len([b for b in buckets if b]),
            "model_time_sum": float(sum(batch_times)),
            "wall_time": float(wall_time),
            "batch_times": batch_times,
        },
    }


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_fba_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    module = importlib.import_module("networks.models")
    build_model = getattr(module, "build_model")
    try:
        model = build_model()
    except TypeError:
        model = build_model(weights_path)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    return model


def _extract_alpha_tensor(output: object) -> Optional[torch.Tensor]:
    if output is None:
        return None
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in _ALPHA_KEYS:
            if key in output:
                value = output[key]
                if isinstance(value, torch.Tensor):
                    return value
        for value in output.values():
            tensor = _extract_alpha_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(output, (list, tuple)):
        for value in output:
            tensor = _extract_alpha_tensor(value)
            if tensor is not None:
                return tensor
    return None


def _slice_alpha(alpha: torch.Tensor, height: int, width: int) -> np.ndarray:
    if alpha.ndim == 3:
        if alpha.shape[0] == 1:
            alpha = alpha[0]
        else:
            alpha = alpha[0]
    elif alpha.ndim == 4:
        alpha = alpha[0]
        if alpha.shape[0] == 1:
            alpha = alpha[0]
    alpha = alpha[:height, :width]
    return alpha.cpu().numpy()


def _multi_gpu_worker(cfg: Config, device: str, buckets: Sequence[Sequence[dict]], canvas_shape, queue):
    torch.cuda.set_device(device)
    sub_cfg = replace(cfg, model=replace(cfg.model, device=device))
    result = _run_single_device(sub_cfg, buckets, canvas_shape, torch.device(device))
    queue.put(result)
