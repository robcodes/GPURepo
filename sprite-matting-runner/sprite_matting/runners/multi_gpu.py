"""Dual-GPU orchestration helpers."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch.multiprocessing as mp

from ..config import Config


def run_dual_gpu(
    cfg: Config,
    buckets: Sequence[Sequence[dict]],
    canvas_shape: Tuple[int, int],
    devices: Sequence[str],
):
    """Split work across two CUDA devices."""

    if len(devices) < 2:
        raise ValueError("At least two devices required for dual-GPU mode")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    chunks = _split_buckets(buckets, 2)
    processes = []
    for device, subset in zip(devices[:2], chunks):
        p = ctx.Process(
            target=_worker_entry,
            args=(cfg, device, subset, canvas_shape, queue),
        )
        p.start()
        processes.append(p)

    results = []
    for _ in processes:
        results.append(queue.get())
    for p in processes:
        p.join()

    if not results:
        return {"alpha_full": np.zeros(canvas_shape, dtype=np.float32), "stats": {}}

    alpha_full = np.zeros(canvas_shape, dtype=np.float32)
    stats = {
        "num_crops": 0,
        "num_batches": 0,
        "model_time_sum": 0.0,
        "wall_time": 0.0,
        "batch_times": [],
    }
    for result in results:
        alpha_full = np.maximum(alpha_full, result["alpha_full"])
        stat = result.get("stats", {})
        stats["num_crops"] += stat.get("num_crops", 0)
        stats["num_batches"] += stat.get("num_batches", 0)
        stats["model_time_sum"] += stat.get("model_time_sum", 0.0)
        stats["batch_times"].extend(stat.get("batch_times", []))
        stats["wall_time"] = max(stats["wall_time"], stat.get("wall_time", 0.0))

    return {"alpha_full": alpha_full, "stats": stats}


def _split_buckets(buckets: Sequence[Sequence[dict]], parts: int):
    filtered = [bucket for bucket in buckets if bucket]
    splits = [[] for _ in range(parts)]
    for idx, bucket in enumerate(filtered):
        splits[idx % parts].append(bucket)
    return splits


def _worker_entry(cfg: Config, device: str, buckets, canvas_shape, queue):
    from .fba_inprocess import _multi_gpu_worker

    _multi_gpu_worker(cfg, device, buckets, canvas_shape, queue)
