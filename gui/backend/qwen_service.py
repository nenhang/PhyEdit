from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


class QwenConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenGenerationConfig:
    config_path: Path
    checkpoint_path: Path | None
    pretrained_model_path: str
    device: str
    base_area: int


class QwenGenerationService:
    def __init__(self, config: QwenGenerationConfig):
        self.config = config
        self._pipeline = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def is_configured(self) -> bool:
        return self.config.checkpoint_path is not None

    def status(self) -> dict[str, Any]:
        checkpoint = self.config.checkpoint_path
        return {
            "configured": self.is_configured,
            "checkpoint_exists": bool(checkpoint and checkpoint.is_file()),
            "loaded": self.is_loaded,
            "device": self.config.device,
            "base_area": self.config.base_area,
        }

    def _validate(self) -> torch.device:
        if not self.config.config_path.is_file():
            raise QwenConfigurationError(
                f"Qwen config does not exist: {self.config.config_path}"
            )
        if self.config.checkpoint_path is None:
            raise QwenConfigurationError(
                "PHYEDIT_CHECKPOINT_PATH is required for final image generation"
            )
        if not self.config.checkpoint_path.is_file():
            raise QwenConfigurationError(
                f"PhyEdit checkpoint does not exist: {self.config.checkpoint_path}"
            )
        if self.config.base_area <= 0:
            raise QwenConfigurationError("QWEN_BASE_AREA must be positive")

        try:
            device = torch.device(self.config.device)
        except (RuntimeError, TypeError) as exc:
            raise QwenConfigurationError(
                f"Invalid QWEN_DEVICE: {self.config.device}"
            ) from exc

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise QwenConfigurationError(
                    f"QWEN_DEVICE={self.config.device} requires CUDA"
                )
            device_index = device.index if device.index is not None else 0
            if device_index >= torch.cuda.device_count():
                raise QwenConfigurationError(
                    f"QWEN_DEVICE={self.config.device} is unavailable; "
                    f"visible CUDA device count is {torch.cuda.device_count()}"
                )
        return device

    def _load_pipeline(self, device: torch.device):
        if self._pipeline is None:
            from phyedit.model_deepspeed.model import load_pipeline_from_config

            self._pipeline = load_pipeline_from_config(
                str(self.config.config_path),
                device=device,
                checkpoint_path=str(self.config.checkpoint_path),
                pretrained_model_path=self.config.pretrained_model_path,
            )
            self._pipeline.set_progress_bar_config(disable=True)
        return self._pipeline

    def generate(self, **kwargs):
        from phyedit.model_deepspeed.generate import generate

        with self._lock:
            device = self._validate()
            pipeline = self._load_pipeline(device)
            return generate(
                pipeline,
                device=device,
                base_area=self.config.base_area,
                **kwargs,
            )
