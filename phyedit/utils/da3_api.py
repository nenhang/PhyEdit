from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_boundary(images: torch.Tensor, target_size: int, method: str) -> torch.Tensor:
    """Match InputProcessor boundary resize policy with torch interpolate."""
    _, _, h, w = images.shape
    if method in ("upper_bound_resize", "upper_bound_crop"):
        scale = target_size / float(max(h, w))
    elif method in ("lower_bound_resize", "lower_bound_crop"):
        scale = target_size / float(min(h, w))
    else:
        raise ValueError(f"Unsupported resize method: {method}")

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    if new_h == h and new_w == w:
        return images

    if scale > 1.0:
        return F.interpolate(images, size=(new_h, new_w), mode="bicubic", align_corners=False)
    return F.interpolate(images, size=(new_h, new_w), mode="area")


def _make_divisible_by_resize(images: torch.Tensor, patch: int = PATCH_SIZE) -> torch.Tensor:
    """Round H/W to nearest multiple of patch via small differentiable resize."""
    _, _, h, w = images.shape

    def nearest_multiple(x: int, p: int) -> int:
        down = (x // p) * p
        up = down + p
        return up if abs(up - x) <= abs(x - down) else down

    new_h = max(1, nearest_multiple(h, patch))
    new_w = max(1, nearest_multiple(w, patch))
    if new_h == h and new_w == w:
        return images

    upscale = (new_h > h) or (new_w > w)
    if upscale:
        return F.interpolate(images, size=(new_h, new_w), mode="bicubic", align_corners=False)
    return F.interpolate(images, size=(new_h, new_w), mode="area")


def _make_divisible_by_crop(images: torch.Tensor, patch: int = PATCH_SIZE) -> torch.Tensor:
    """Floor H/W to multiple of patch via center crop."""
    _, _, h, w = images.shape
    new_h = (h // patch) * patch
    new_w = (w // patch) * patch
    if new_h == h and new_w == w:
        return images
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return images[:, :, top : top + new_h, left : left + new_w]


def _normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """ImageNet normalize on BCHW tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def preprocess_images_for_da3_model(
    da3: Any,
    images: torch.Tensor,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    batch_type: str = "view",
) -> torch.Tensor:
    """
    Preprocess images into DA3 model input format with gradient support.

    The pipeline mirrors `InputProcessor` logic, but is fully differentiable:
      1) boundary resize (upper/lower bound)
      2) make H/W divisible by 14 (resize or center-crop)
      3) ImageNet normalization
      4) pack to DA3 expected input layout

    Args:
            da3: A `DepthAnything3` instance.
            images: Tensor in RGB with shape (B, 3, H, W). Use float tensor in [0, 1] if you
                want behavior consistent with DA3 default preprocessing.
            process_res: Target processing resolution.
            process_res_method: One of `upper_bound_resize`, `upper_bound_crop`,
                `lower_bound_resize`, `lower_bound_crop`.
            batch_type: `view` -> output shape (1, B, 3, H', W'); `batch` -> (B, 1, 3, H', W').

    Returns:
            Model-ready image tensor that can be fed into `da3.model(...)`, while
            preserving autograd path from output back to input `images`.
    """
    if not hasattr(da3, "model"):
        raise TypeError("`da3` does not look like a DepthAnything3 instance.")

    if images.ndim != 4:
        raise ValueError(f"`images` must be BCHW tensor, got {tuple(images.shape)}")

    imgs_cpu = images
    if not torch.is_floating_point(imgs_cpu):
        imgs_cpu = imgs_cpu.float() / 255.0

    imgs_cpu = _resize_boundary(imgs_cpu, process_res, process_res_method)
    if process_res_method.endswith("resize"):
        imgs_cpu = _make_divisible_by_resize(imgs_cpu, PATCH_SIZE)
    elif process_res_method.endswith("crop"):
        imgs_cpu = _make_divisible_by_crop(imgs_cpu, PATCH_SIZE)
    else:
        raise ValueError(f"Unsupported process_res_method: {process_res_method}")

    imgs_cpu = _normalize_imagenet(imgs_cpu)

    try:
        device = next(da3.parameters()).device
    except StopIteration as error:
        raise TypeError("`da3` does not expose model parameters.") from error

    imgs = imgs_cpu.to(device=device, non_blocking=True).float()
    if batch_type == "view":
        return imgs.unsqueeze(0)
    if batch_type == "batch":
        return imgs.unsqueeze(1)
    raise ValueError(f"Unsupported batch_type: {batch_type}")
