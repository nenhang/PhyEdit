from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

if TYPE_CHECKING:
    from depth_anything_3.api import DepthAnything3


DEFAULT_DEPTH_MODEL = "depth-anything/da3nested-giant-large"


class DepthModel:
    def __init__(self, model_path=None, device: torch.device | str = torch.device("cuda")):
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as error:
            raise ImportError(
                "Depth Anything 3 is required for ManipEval. "
                "Run scripts/setup_depth_anything_3.sh before evaluation."
            ) from error

        if model_path is None:
            model_path = os.getenv("DA_MODEL_PATH", DEFAULT_DEPTH_MODEL)

        self.device = torch.device(device)
        print(f"Loading DepthAnything3 model from {model_path}...")
        self.model: DepthAnything3 = DepthAnything3.from_pretrained(model_path)
        self.model = self.model.to(device=self.device)
        self.model.eval()

    def _prepare_model_inputs(
        self,
        images: list[Image.Image | str],
        process_res: int,
        batch_type: str,
    ) -> torch.Tensor:
        images_bchw, _, _ = self.model.input_processor(
            images,
            process_res=process_res,
            process_res_method="upper_bound_resize",
        )
        images_bchw = images_bchw.to(device=self.device, non_blocking=True).float()

        if batch_type == "view":
            return images_bchw.unsqueeze(0)
        if batch_type == "batch":
            return images_bchw.unsqueeze(1)
        raise ValueError(f"Unsupported batch_type: {batch_type}")

    @staticmethod
    def _select_images(output: torch.Tensor, batch_type: str, name: str) -> torch.Tensor:
        if output.ndim != 4:
            raise RuntimeError(f"Unexpected DA3 {name} shape: {tuple(output.shape)}")

        if batch_type == "view":
            if output.shape[0] != 1:
                raise RuntimeError(f"Expected one DA3 view batch for {name}, got {tuple(output.shape)}")
            return output[0]

        if batch_type == "batch":
            if output.shape[1] != 1:
                raise RuntimeError(f"Expected one DA3 view per batch item for {name}, got {tuple(output.shape)}")
            return output[:, 0]

        raise ValueError(f"Unsupported batch_type: {batch_type}")

    def _run_model(self, inputs: torch.Tensor, batch_type: str) -> tuple[torch.Tensor, ...]:
        autocast_enabled = inputs.device.type == "cuda"
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(
            device_type=inputs.device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            prediction: Any = self.model.model(inputs)

        if prediction.get("non_sky_mask") is None:
            raise RuntimeError(
                "Depth Anything 3 is missing the PhyEdit compatibility patch. "
                "Run scripts/setup_depth_anything_3.sh before evaluation."
            )

        required_outputs = ("depth", "intrinsics", "extrinsics")
        missing_outputs = [name for name in required_outputs if prediction.get(name) is None]
        if missing_outputs:
            raise RuntimeError(f"Depth Anything 3 did not return required outputs: {', '.join(missing_outputs)}")

        depth = self._select_images(prediction["depth"], batch_type, "depth")
        valid_mask = self._select_images(prediction["non_sky_mask"], batch_type, "non_sky_mask")
        intrinsics = self._select_images(prediction["intrinsics"], batch_type, "intrinsics")
        extrinsics = self._select_images(prediction["extrinsics"], batch_type, "extrinsics")
        return depth, valid_mask, intrinsics, extrinsics

    @torch.inference_mode()
    def predict_depth(
        self,
        images: list[Image.Image | str],
        process_res: int,
        batch_type: str = "batch",
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(images) == 0:
            return (
                torch.empty((0,), device=self.device),
                torch.empty((0,), device=self.device, dtype=torch.bool),
                torch.empty((0, 3, 3), device=self.device),
                torch.empty((0, 4, 4), device=self.device),
            )

        if batch_size is None or batch_size <= 0:
            batch_size = len(images)

        output_chunks: list[list[torch.Tensor]] = [[], [], [], []]
        for start_idx in range(0, len(images), batch_size):
            image_chunk = images[start_idx : start_idx + batch_size]
            inputs = self._prepare_model_inputs(image_chunk, process_res, batch_type)
            chunk_outputs = self._run_model(inputs, batch_type)
            for chunks, output in zip(output_chunks, chunk_outputs):
                chunks.append(output)

        return tuple(torch.cat(chunks, dim=0) for chunks in output_chunks)
