"""Configuration helpers for sprite-matting-runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelConfig:
    """Model selection and runtime toggles."""

    type: str = "fba"
    weights: str = "checkpoints/fba_matting.pth"
    amp: bool = True
    device: str = "cuda:0"


@dataclass
class SpriteConfig:
    """Sprite extraction configuration."""

    min_area: int = 100
    pad: int = 20
    tri_fg_px: int = 8
    tri_bg_px: int = 8
    max_crop_pixels: int = 1_500_000


@dataclass
class BatchingConfig:
    """Batch binning configuration."""

    pixel_budget: int = 2_000_000
    pad_multiple: int = 32


@dataclass
class MultiGPUConfig:
    """Dual-GPU runner configuration."""

    enabled: bool = False
    devices: List[str] = field(default_factory=lambda: ["cuda:0", "cuda:1"])


@dataclass
class SaveConfig:
    """Output save toggles."""

    save_per_sprite: bool = False
    save_composites: bool = False


@dataclass
class Config:
    """Top level configuration."""

    input_image: str
    input_mask: str
    output_dir: str
    model: ModelConfig = field(default_factory=ModelConfig)
    sprite: SpriteConfig = field(default_factory=SpriteConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    multi_gpu: MultiGPUConfig = field(default_factory=MultiGPUConfig)
    save: SaveConfig = field(default_factory=SaveConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        kwargs = dict(data)
        kwargs["model"] = _coerce(ModelConfig, data.get("model", {}))
        kwargs["sprite"] = _coerce(SpriteConfig, data.get("sprite", {}))
        kwargs["batching"] = _coerce(BatchingConfig, data.get("batching", {}))
        kwargs["multi_gpu"] = _coerce(MultiGPUConfig, data.get("multi_gpu", {}))
        kwargs["save"] = _coerce(SaveConfig, data.get("save", {}))
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_image": self.input_image,
            "input_mask": self.input_mask,
            "output_dir": self.output_dir,
            "model": vars(self.model),
            "sprite": vars(self.sprite),
            "batching": vars(self.batching),
            "multi_gpu": {"enabled": self.multi_gpu.enabled, "devices": list(self.multi_gpu.devices)},
            "save": vars(self.save),
        }

    def update_from_namespace(self, ns: Any) -> None:
        for field_name in ("input_image", "input_mask", "output_dir"):
            value = getattr(ns, field_name, None)
            if value is not None:
                setattr(self, field_name, value)

        for name, cls_type in (
            ("model", ModelConfig),
            ("sprite", SpriteConfig),
            ("batching", BatchingConfig),
            ("multi_gpu", MultiGPUConfig),
            ("save", SaveConfig),
        ):
            section = getattr(self, name)
            for key, _ in cls_type.__annotations__.items():
                attr = f"{name}_{key}"
                if hasattr(ns, attr):
                    val = getattr(ns, attr)
                    if val is not None:
                        setattr(section, key, val)

    def ensure_output_dir(self) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out


def _coerce(cls: Any, value: Any) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"Cannot coerce {value!r} to {cls}")
