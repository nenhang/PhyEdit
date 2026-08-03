import copy
import math
import os
from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.geometry_utils import translate_objects_3d_batch
from ..utils.image_process import mask_moved_image
from ..utils.text_process import get_edit_prompt, get_edit_prompt_coord_only
from .subset_utils import load_metadata

QWEN_IMAGE_EDIT_BASE_SIZE = 1024
QWEN_IMAGE_EDIT_BASE_AREA = QWEN_IMAGE_EDIT_BASE_SIZE * QWEN_IMAGE_EDIT_BASE_SIZE
QWEN_IMAGE_SIZE_MULTIPLE = 16


def _require_positive_number(value: Any, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _floor_to_multiple(value: int | float, multiple: int) -> int:
    return max(multiple, int(value) // multiple * multiple)


def _round_to_multiple(value: int | float, multiple: int) -> int:
    return max(multiple, round(float(value) / multiple) * multiple)


def calculate_qwen_edit_dimensions(
    target_area: int | float,
    ratio: float,
    multiple: int = QWEN_IMAGE_SIZE_MULTIPLE,
) -> tuple[int, int]:
    """Match Qwen Image Edit's area-based 1024x1024 native size rule."""
    target_area = _require_positive_number(target_area, "target_area")
    ratio = _require_positive_number(ratio, "ratio")
    multiple = int(multiple)
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")

    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return _round_to_multiple(width, multiple), _round_to_multiple(height, multiple)


def resolve_qwen_edit_size(
    source_size: tuple[int, int] | list[int],
    *,
    base_area: int | float | None = QWEN_IMAGE_EDIT_BASE_AREA,
    longer_side: int | None = None,
    height: int | None = None,
    width: int | None = None,
    require_16_multiple: bool = True,
) -> tuple[int, int]:
    """Resolve (width, height) for train/sample images using the Qwen edit convention."""
    if source_size is None or len(source_size) != 2:
        raise ValueError(f"source_size must be a (width, height) pair, got {source_size}")

    src_w = _require_positive_number(source_size[0], "source width")
    src_h = _require_positive_number(source_size[1], "source height")
    multiple = QWEN_IMAGE_SIZE_MULTIPLE if require_16_multiple else 1

    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("height and width must be set together")
        target_w = _require_positive_number(width, "width")
        target_h = _require_positive_number(height, "height")
        if require_16_multiple:
            target_w = _floor_to_multiple(target_w, multiple)
            target_h = _floor_to_multiple(target_h, multiple)
        return int(target_w), int(target_h)

    ratio = src_w / src_h
    if base_area is not None:
        return calculate_qwen_edit_dimensions(base_area, ratio, multiple=multiple)

    if longer_side is not None:
        longer_side = int(_require_positive_number(longer_side, "longer_side"))
        if ratio >= 1.0:
            target_w = longer_side
            target_h = longer_side / ratio
        else:
            target_h = longer_side
            target_w = longer_side * ratio
        if require_16_multiple:
            target_w = _round_to_multiple(target_w, multiple)
            target_h = _round_to_multiple(target_h, multiple)
        return int(target_w), int(target_h)

    target_w, target_h = src_w, src_h
    if require_16_multiple:
        target_w = _floor_to_multiple(target_w, multiple)
        target_h = _floor_to_multiple(target_h, multiple)
    return int(target_w), int(target_h)


def qwen_edit_base_area_from_config(
    config: dict,
    *,
    area_key: str = "image_base_area",
    size_key: str = "image_base_size",
    default_area: int = QWEN_IMAGE_EDIT_BASE_AREA,
) -> int:
    if area_key in config and config[area_key] is not None:
        return int(config[area_key])
    if "image_base_area" in config and config["image_base_area"] is not None:
        return int(config["image_base_area"])
    if size_key in config and config[size_key] is not None:
        base_size = int(config[size_key])
        return base_size * base_size
    if "image_base_size" in config and config["image_base_size"] is not None:
        base_size = int(config["image_base_size"])
        return base_size * base_size
    return int(default_area)


class AspectRatioBatchSampler:
    def __init__(self, metadata, batch_size, shuffle=True, drop_last=True, seed=42):
        self.metadata = metadata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Group indices once; only the shuffled batch order changes per epoch.
        self.groups = defaultdict(list)
        for idx, item in enumerate(self.metadata):
            w, h = item["resolution"]
            ratio = round(w / h, 2)
            self.groups[ratio].append(idx)

        # Delay sharding until DDP has initialized and the actual world size is known.
        self.batches = None
        self.start_batch = 0

    def _get_rank_world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()

        # torchrun exports these values before process-group initialization.
        env_rank = int(os.environ.get("RANK", "0"))
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return env_rank, env_world_size

    def _prepare_batches(self):
        curr_rank, world_size = self._get_rank_world_size()

        all_batches = []
        # Couple the seed to the epoch while keeping all ranks deterministic.
        g = np.random.RandomState(self.seed + self.epoch)

        # Keep aspect-ratio group traversal deterministic.
        sorted_keys = sorted(self.groups.keys())

        for ratio in sorted_keys:
            indices = list(self.groups[ratio])
            if self.shuffle:
                g.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
                elif not self.drop_last and len(batch) > 0:
                    all_batches.append(batch)

        if self.shuffle:
            g.shuffle(all_batches)

        rank_batches = all_batches[curr_rank::world_size]
        start_batch = min(max(int(self.start_batch), 0), len(rank_batches))
        print(
            f"Rank {curr_rank}/{world_size} will use {len(rank_batches) - start_batch} batches "
            f"for epoch {self.epoch} with seed {self.seed + self.epoch}; "
            f"resume_batch={start_batch}"
        )

        return rank_batches[start_batch:]

    def set_epoch(self, epoch, start_batch: int = 0):
        self.epoch = epoch
        self.start_batch = max(int(start_batch), 0)
        self.batches = None

    def set_start_batch(self, start_batch: int):
        self.start_batch = max(int(start_batch), 0)
        self.batches = None

    def __iter__(self):
        # Rebuild from the current epoch and rank before every iteration.
        self.batches = self._prepare_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        if self.batches is None:
            self.batches = self._prepare_batches()
        return len(self.batches)


class ImagePairDataset(Dataset):
    def __init__(self, metadata, text_drop_rate: float = 0.0, need_moved_image: bool = True):
        self.metadata = metadata
        self.text_drop_rate = float(text_drop_rate)
        if not (0.0 <= self.text_drop_rate <= 1.0):
            raise ValueError(f"text_drop_rate must be in [0, 1], got {self.text_drop_rate}")
        self.need_moved_image = need_moved_image

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        f1_image = Image.open(item["f1_path"]).convert("RGB")
        f2_image = Image.open(item["f2_path"]).convert("RGB")

        f1_masks = [Image.open(p).convert("L") for p in item["f1_mask_path"]]
        f2_masks = [Image.open(p).convert("L") for p in item["f2_mask_path"]]

        depth_range_1 = copy.deepcopy(item["f1_depth_range"])
        depth_range_2 = copy.deepcopy(item["f2_depth_range"])
        depth_range = [min(depth_range_1[0], depth_range_2[0]), max(depth_range_1[1], depth_range_2[1])]
        f1_coords = copy.deepcopy(item["f1_coords"])
        f2_coords = copy.deepcopy(item["f2_coords"])

        # norm x, y, z to [0, 1]
        coord_pairs = []
        for f1_coord, f2_coord in zip(f1_coords, f2_coords):
            f1_coord[2] = (f1_coord[2] - depth_range[0]) / (depth_range[1] - depth_range[0])
            f2_coord[2] = (f2_coord[2] - depth_range[0]) / (depth_range[1] - depth_range[0])
            coord_pairs.append((f1_coord, f2_coord))

        edit_prompt = (
            get_edit_prompt(
                object_name=item["object_name"],
                coordinates=coord_pairs,
                object_edit_prompt=item["edit_prompt"],
                additional_prompt=item["additional_prompt"],
            )
            if self.need_moved_image
            else get_edit_prompt_coord_only(
                object_name=item["object_name"],
                coordinates=coord_pairs,
                object_edit_prompt=item["edit_prompt"],
                additional_prompt=item["additional_prompt"],
            )
        )
        if np.random.random() < self.text_drop_rate:
            edit_prompt = ""
        moved_image_pil = None
        if self.need_moved_image:
            moved_image_path = item.get("moved_image_path")
            if moved_image_path:
                moved_image_pil = Image.open(moved_image_path).convert("RGB")
            else:
                moved_image_torch, _, bg_patch_masks, _, moved_obj_masks = translate_objects_3d_batch(
                    images=[item["f1_path"]],
                    masks=[f1_masks],
                    target_coords=[item["f2_coords"]],
                    depths=[item["f1_depth_path"]],
                    intrinsics=[item["f1_intrinsic_path"]],
                    extrinsics=[item["f1_extrinsic_path"]],
                    device="cpu",
                )
                moved_image_torch = mask_moved_image(
                    images_torch=moved_image_torch, obj_masks=moved_obj_masks, bg_patch_masks=bg_patch_masks
                )
                moved_image = (moved_image_torch[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                moved_image_pil = Image.fromarray(moved_image)

        return {
            "f1_image": f1_image,
            "f2_image": f2_image,
            "f1_masks": f1_masks,
            "f2_masks": f2_masks,
            "moved_image": moved_image_pil,
            "prompt": edit_prompt,
            "resolution": item["resolution"],
            "f1_coords": f1_coords,
            "f2_coords": f2_coords,
            "f1_depth_range": item["f1_depth_range"],
            "f2_depth_range": item["f2_depth_range"],
        }


def aspect_ratio_collate_fn(batch, base_area=None, longer_side=None, require_16_multiple=True):
    """Resize and collate one aspect-ratio-compatible batch."""
    if not batch:
        return {}

    # Resolve one shared target size for the batch.
    ratios = [item["resolution"][0] / item["resolution"][1] for item in batch]
    avg_ratio = sum(ratios) / len(ratios)
    if base_area is not None:
        target_w, target_h = calculate_qwen_edit_dimensions(
            base_area,
            avg_ratio,
            multiple=QWEN_IMAGE_SIZE_MULTIPLE if require_16_multiple else 1,
        )
    elif longer_side is not None:
        longer_side = int(_require_positive_number(longer_side, "longer_side"))
        if avg_ratio >= 1.0:
            target_w = longer_side
            target_h = int(longer_side / avg_ratio)
        else:
            target_h = longer_side
            target_w = int(longer_side * avg_ratio)
        if require_16_multiple:
            target_w = _round_to_multiple(target_w, QWEN_IMAGE_SIZE_MULTIPLE)
            target_h = _round_to_multiple(target_h, QWEN_IMAGE_SIZE_MULTIPLE)
    else:
        # Without an explicit policy, use the largest dimensions in the batch.
        max_w = max(item["resolution"][0] for item in batch)
        max_h = max(item["resolution"][1] for item in batch)
        target_w, target_h = max_w, max_h

    if require_16_multiple and base_area is None and longer_side is None:
        target_w = _floor_to_multiple(target_w, QWEN_IMAGE_SIZE_MULTIPLE)
        target_h = _floor_to_multiple(target_h, QWEN_IMAGE_SIZE_MULTIPLE)

    f1_list, f2_list = [], []
    f1_masks_list, f2_masks_list = [], []

    moved_list = []

    for item in batch:
        # Resize RGB images and masks consistently.
        img1 = item["f1_image"].resize((target_w, target_h), resample=Image.BILINEAR)
        img2 = item["f2_image"].resize((target_w, target_h), resample=Image.BILINEAR)
        f1_list.append(img1)
        f2_list.append(img2)

        f1_masks = [mask.resize((target_w, target_h), resample=Image.NEAREST) for mask in item["f1_masks"]]
        f2_masks = [mask.resize((target_w, target_h), resample=Image.NEAREST) for mask in item["f2_masks"]]
        f1_masks_list.append(f1_masks)
        f2_masks_list.append(f2_masks)

        moved_img = (
            item["moved_image"].resize((target_w, target_h), resample=Image.BILINEAR)
            if item["moved_image"] is not None
            else None
        )
        moved_list.append(moved_img)

    return {
        "f1_images": f1_list,
        "f2_images": f2_list,
        "f1_masks": f1_masks_list,
        "f2_masks": f2_masks_list,
        "prompts": [item["prompt"] for item in batch],
        "moved_images": moved_list if "moved_image" in batch[0] else None,
        "resolutions": [(target_w, target_h)] * len(batch),
        "f1_coords": [item["f1_coords"] for item in batch],
        "f2_coords": [item["f2_coords"] for item in batch],
        "f1_depth_ranges": [item["f1_depth_range"] for item in batch],
        "f2_depth_ranges": [item["f2_depth_range"] for item in batch],
    }


class ImagePairDataLoader(DataLoader):
    def __init__(
        self,
        metadata=None,
        batch_size=16,
        shuffle=True,
        num_workers=32,
        base_area=QWEN_IMAGE_EDIT_BASE_AREA,
        longer_side=None,
        use_aspect_ratio_sampler=True,
        require_16_multiple=True,
        text_drop_rate=0.0,
        need_moved_image=True,
        multiprocessing_context=None,
        pin_memory=False,
    ):
        if metadata is None:
            metadata = load_metadata()
        dataset = ImagePairDataset(metadata, text_drop_rate=text_drop_rate, need_moved_image=need_moved_image)

        collate_fn = partial(
            aspect_ratio_collate_fn,
            base_area=base_area,
            longer_side=longer_side,
            require_16_multiple=require_16_multiple,
        )
        dataloader_kwargs = {
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0 and multiprocessing_context is not None:
            dataloader_kwargs["multiprocessing_context"] = multiprocessing_context

        if use_aspect_ratio_sampler:
            batch_sampler = AspectRatioBatchSampler(metadata, batch_size, shuffle=shuffle)
            super().__init__(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
        else:
            # use default sampler
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                **dataloader_kwargs,
            )
