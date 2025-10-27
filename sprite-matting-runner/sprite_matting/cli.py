"""Command line interface for sprite-matting-runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from .config import Config
from .io_utils import ensure_dir, load_binary_mask, load_image_rgb, save_alpha_png
from .runners.fba_inprocess import run_fba_inprocess
from .viewers import save_strip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sprite-matting")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the matting pipeline")
    run_p.add_argument("--config", help="YAML configuration file", default=None)
    run_p.add_argument("--image", dest="input_image", required=True)
    run_p.add_argument("--mask", dest="input_mask", required=True)
    run_p.add_argument("--out-dir", dest="output_dir", required=True)
    run_p.add_argument("--model", dest="model_type", default=None)
    run_p.add_argument("--weights", dest="model_weights", default=None)
    run_p.add_argument("--fba-repo", dest="model_fba_repo", default=None,
                       help="Path to the cloned FBA_Matting repository")
    run_p.add_argument("--amp", dest="model_amp", action="store_true")
    run_p.add_argument("--no-amp", dest="model_amp", action="store_false")
    run_p.set_defaults(model_amp=None)
    run_p.add_argument("--device", dest="model_device", default=None)
    run_p.add_argument("--tri-fg", dest="sprite_tri_fg_px", type=int, default=None)
    run_p.add_argument("--tri-bg", dest="sprite_tri_bg_px", type=int, default=None)
    run_p.add_argument("--min-area", dest="sprite_min_area", type=int, default=None)
    run_p.add_argument("--pad", dest="sprite_pad", type=int, default=None)
    run_p.add_argument("--max-crop-pixels", dest="sprite_max_crop_pixels", type=int, default=None)
    run_p.add_argument("--pixel-budget", dest="batching_pixel_budget", type=int, default=None)
    run_p.add_argument("--pad-multiple", dest="batching_pad_multiple", type=int, default=None)
    run_p.add_argument("--multi-gpu", dest="multi_gpu_devices", default=None)
    run_p.add_argument("--save-per-sprite", dest="save_save_per_sprite", action="store_true")
    run_p.set_defaults(save_save_per_sprite=None)

    view_p = sub.add_parser("view", help="Create quick viewer strips")
    view_p.add_argument("--image", required=True)
    view_p.add_argument("--trimap")
    view_p.add_argument("--alpha")
    view_p.add_argument("--out", required=True)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = _load_config(args)
        image = load_image_rgb(cfg.input_image)
        mask = load_binary_mask(cfg.input_mask)
        ensure_dir(cfg.output_dir)
        if cfg.model.type.lower() != "fba":
            raise ValueError("Only FBA model is currently implemented")
        result = run_fba_inprocess(cfg, image, mask)
        alpha = result["alpha_full"]
        output_dir = Path(cfg.output_dir)
        output_path = output_dir / f"{Path(cfg.input_image).stem}_alpha.png"
        save_alpha_png(output_path, alpha)
        stats = result.get("stats", {})
        amp_tag = "AMP" if cfg.model.amp else "FP32"
        print(
            f"Model: FBA ({amp_tag}), device: {cfg.model.device}\n"
            f"Crops: {stats.get('num_crops', 0)} | Batches: {stats.get('num_batches', 0)} | "
            f"model-only sum: {stats.get('model_time_sum', 0.0):.2f}s | wall: {stats.get('wall_time', 0.0):.2f}s"
        )
        if stats.get("batch_times"):
            ms = ", ".join(f"{t * 1000:.1f}" for t in stats["batch_times"])
            print(f"Per-batch ms: [{ms}]")
        print(f"Saved: {output_path}")

    elif args.command == "view":
        images = []
        if args.image:
            images.append(load_image_rgb(args.image))
        if args.trimap:
            tri = load_binary_mask(args.trimap).astype(np.float32)
            images.append(np.repeat(tri[:, :, None], 3, axis=2))
        if args.alpha:
            alpha = load_binary_mask(args.alpha).astype(np.float32)
            images.append(np.repeat(alpha[:, :, None], 3, axis=2))
        save_strip(images, args.out)


def _load_config(args) -> Config:
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config(
            input_image=args.input_image,
            input_mask=args.input_mask,
            output_dir=args.output_dir,
        )
    cfg.update_from_namespace(args)
    if args.multi_gpu_devices:
        cfg.multi_gpu.enabled = True
        cfg.multi_gpu.devices = [d.strip() for d in args.multi_gpu_devices.split(",") if d.strip()]
    return cfg


if __name__ == "__main__":
    main()
