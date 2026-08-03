#!/usr/bin/env python3
"""Precompute moved-image conditioning inputs into a standalone cache directory.

The script leaves the aggregated dataset untouched. It writes moved images under
``cache_root/images`` and emits a derived metadata file with ``moved_image_path``
pointing at that standalone cache. By default, the 3D transform runs at each
source image's original resolution. ``--transform-base-area`` instead resizes
the source image and masks to an aspect-ratio-preserving target area before the
3D transform, matching the public inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phyedit.data.dataset import resolve_qwen_edit_size  # noqa: E402
from phyedit.data.subset_utils import load_aggregated_metadata  # noqa: E402
from phyedit.utils.geometry_utils import translate_objects_3d_batch  # noqa: E402
from phyedit.utils.image_process import mask_moved_image  # noqa: E402

DEFAULT_METADATA_PATH = os.getenv("REALMANIP_METADATA_PATH") or os.getenv(
    "AGGREGATED_METADATA_PATH"
)
DEFAULT_CACHE_ROOT = os.getenv("PHYEDIT_PREVIEW_CACHE_DIR", "./cache/moved_images")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def item_key(item: dict[str, Any], index: int) -> str:
    stable = item.get("source_key") or item.get("sample_id")
    if stable:
        return str(stable)
    payload = {
        "index": index,
        "f1_path": item.get("f1_path"),
        "f2_path": item.get("f2_path"),
        "f1_mask_path": item.get("f1_mask_path"),
        "f2_coords": item.get("f2_coords"),
        "f1_depth_path": item.get("f1_depth_path"),
        "f1_intrinsic_path": item.get("f1_intrinsic_path"),
        "f1_extrinsic_path": item.get("f1_extrinsic_path"),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def cache_path_for(
    item: dict[str, Any],
    index: int,
    cache_root: Path,
    image_format: str,
    transform_base_area: int | None = None,
) -> Path:
    key = item_key(item, index)
    subset = str(item.get("source_subset") or "unknown")
    suffix = ".png" if image_format == "png" else ".jpg"
    image_root = cache_root / "images"
    if transform_base_area is not None:
        image_root = image_root / f"base_area_{int(transform_base_area)}"
    return image_root / subset / key[:2] / f"{key}{suffix}"


def metadata_path_for_cache(
    cache_root: Path, transform_base_area: int | None = None
) -> Path:
    filename = (
        "train_metadata_with_moved.jsonl"
        if transform_base_area is None
        else f"train_metadata_with_moved_base_area_{int(transform_base_area)}.jsonl"
    )
    return cache_root / "metadata" / filename


def resolve_existing_cache_path(
    item: dict[str, Any],
    cache_root: Path,
    index: int,
    transform_base_area: int | None = None,
) -> str | None:
    if transform_base_area is None:
        existing = item.get("moved_image_path")
        if existing and Path(existing).is_file():
            return str(existing)
    for image_format in ("jpg", "png"):
        path = cache_path_for(
            item,
            index,
            cache_root,
            image_format,
            transform_base_area=transform_base_area,
        )
        if path.is_file():
            return path.as_posix()
    return None


def save_image(image: Image.Image, path: Path, image_format: str, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    if image_format == "png":
        image.save(tmp_path, format="PNG", optimize=False)
    else:
        image.save(tmp_path, format="JPEG", quality=quality, subsampling=0)
    os.replace(tmp_path, path)


def render_moved_image(
    item: dict[str, Any],
    device: str,
    transform_base_area: int | None = None,
) -> Image.Image:
    source_image: Image.Image | str = item["f1_path"]
    f1_masks = [Image.open(p).convert("L") for p in item["f1_mask_path"]]
    if transform_base_area is not None:
        source_image = Image.open(item["f1_path"]).convert("RGB")
        target_size = resolve_qwen_edit_size(
            source_image.size,
            base_area=transform_base_area,
        )
        source_image = source_image.resize(target_size, Image.Resampling.BICUBIC)
        f1_masks = [
            mask.resize(target_size, Image.Resampling.NEAREST) for mask in f1_masks
        ]

    with torch.inference_mode():
        moved_image_torch, _, bg_patch_masks, _, moved_obj_masks = (
            translate_objects_3d_batch(
                images=[source_image],
                masks=[f1_masks],
                target_coords=[item["f2_coords"]],
                depths=[item["f1_depth_path"]],
                intrinsics=[item["f1_intrinsic_path"]],
                extrinsics=[item["f1_extrinsic_path"]],
                device=device,
            )
        )
        moved_image_torch = mask_moved_image(
            images_torch=moved_image_torch,
            obj_masks=moved_obj_masks,
            bg_patch_masks=bg_patch_masks,
        )
    moved_image = (
        (moved_image_torch[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(moved_image)


def process_one(
    task: tuple[int, dict[str, Any], str, str, str, int, bool, int | None],
) -> dict[str, Any]:
    (
        index,
        item,
        cache_root_raw,
        image_format,
        device,
        quality,
        overwrite,
        transform_base_area,
    ) = task
    cache_root = Path(cache_root_raw)
    out_path = cache_path_for(
        item,
        index,
        cache_root,
        image_format,
        transform_base_area=transform_base_area,
    )
    if out_path.is_file() and not overwrite:
        return {"status": "exists", "index": index, "path": out_path.as_posix()}

    try:
        image = render_moved_image(
            item,
            device=device,
            transform_base_area=transform_base_area,
        )
        save_image(image, out_path, image_format=image_format, quality=quality)
        return {"status": "written", "index": index, "path": out_path.as_posix()}
    except Exception as exc:
        return {
            "status": "failed",
            "index": index,
            "path": out_path.as_posix(),
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_indices(
    total: int, start: int, end: int | None, stride: int, offset: int, limit: int | None
) -> list[int]:
    stop = total if end is None else min(end, total)
    indices = [i for i in range(start, stop) if (i - offset) % stride == 0]
    if limit is not None:
        indices = indices[:limit]
    return indices


def build_metadata(
    items: list[dict[str, Any]],
    cache_root: Path,
    metadata_out: Path,
    transform_base_area: int | None = None,
) -> dict[str, int]:
    updated = []
    with_cache = 0
    missing = 0
    for index, item in enumerate(items):
        new_item = dict(item)
        cache_path = resolve_existing_cache_path(
            item,
            cache_root,
            index,
            transform_base_area=transform_base_area,
        )
        if cache_path:
            new_item["moved_image_path"] = cache_path
            with_cache += 1
        else:
            new_item.pop("moved_image_path", None)
            missing += 1
        updated.append(new_item)

    write_jsonl(metadata_out, updated)
    write_json(metadata_out.with_suffix(".json"), updated)
    return {
        "items": len(updated),
        "with_moved_image_path": with_cache,
        "missing_moved_image_path": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
    )
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--metadata-out", default=None)
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma-separated GPU ids. Overrides --device for task assignment.",
    )
    parser.add_argument(
        "--transform-base-area",
        "--base-area",
        dest="transform_base_area",
        type=int,
        default=None,
        help=(
            "Resize each source image and mask to this aspect-ratio-preserving area before the 3D transform. "
            "For example, 589824 produces 1024x576 for 16:9 and 768x768 for 1:1. "
            "Omit this option to transform at the original source resolution."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--fail-log", default=None)
    args = parser.parse_args()
    if not args.metadata:
        parser.error(
            "--metadata is required unless REALMANIP_METADATA_PATH or "
            "AGGREGATED_METADATA_PATH is set"
        )
    if args.transform_base_area is not None and args.transform_base_area <= 0:
        parser.error("--transform-base-area must be positive")

    metadata_path = Path(args.metadata).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    metadata_out = (
        Path(args.metadata_out).expanduser().resolve()
        if args.metadata_out
        else metadata_path_for_cache(cache_root, args.transform_base_area)
    )
    fail_log = (
        Path(args.fail_log).expanduser().resolve()
        if args.fail_log
        else cache_root / "metadata" / "moved_image_failures.jsonl"
    )

    items = load_aggregated_metadata(metadata_path)
    stats = {
        "metadata": metadata_path.as_posix(),
        "cache_root": cache_root.as_posix(),
        "metadata_out": metadata_out.as_posix(),
        "format": args.format,
        "quality": args.quality,
        "transform_base_area": args.transform_base_area,
        "transform_mode": "target_area_before_transform"
        if args.transform_base_area is not None
        else "original_resolution",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_metadata_items": len(items),
    }

    if args.metadata_only:
        stats.update(
            build_metadata(
                items,
                cache_root,
                metadata_out,
                transform_base_area=args.transform_base_area,
            )
        )
        stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(cache_root / "metadata" / "moved_image_cache_stats.json", stats)
        print(f"[metadata-only] wrote {metadata_out}", flush=True)
        print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
        return

    indices = select_indices(
        len(items), args.start, args.end, args.stride, args.offset, args.limit
    )
    gpu_ids = [part.strip() for part in args.gpu_ids.split(",") if part.strip()]
    devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids] if gpu_ids else [args.device]
    print(
        f"[cache] metadata_items={len(items):,} selected={len(indices):,} cache_root={cache_root} "
        f"workers={args.workers} devices={devices} transform_base_area={args.transform_base_area or 'original'}",
        flush=True,
    )

    tasks = []
    skipped = 0
    for task_pos, index in enumerate(indices):
        item = items[index]
        out_path = cache_path_for(
            item,
            index,
            cache_root,
            args.format,
            transform_base_area=args.transform_base_area,
        )
        if out_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        device = devices[task_pos % len(devices)]
        tasks.append(
            (
                index,
                item,
                cache_root.as_posix(),
                args.format,
                device,
                args.quality,
                args.overwrite,
                args.transform_base_area,
            )
        )

    counts: dict[str, int] = {"skipped_existing": skipped}
    failures = []
    if tasks:
        if args.workers <= 1:
            results_iter = (process_one(task) for task in tasks)
            for done, result in enumerate(results_iter, 1):
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1
                if status == "failed":
                    failures.append(result)
                if done % max(1, args.progress_every) == 0 or done == len(tasks):
                    print(f"[cache] {done:,}/{len(tasks):,} {counts}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_one, task) for task in tasks]
                for done, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    status = result["status"]
                    counts[status] = counts.get(status, 0) + 1
                    if status == "failed":
                        failures.append(result)
                    if done % max(1, args.progress_every) == 0 or done == len(futures):
                        print(f"[cache] {done:,}/{len(futures):,} {counts}", flush=True)

    if failures:
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        with fail_log.open("w", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
        print(f"[cache] failures={len(failures):,}; wrote {fail_log}", flush=True)

    stats["cache_counts"] = counts
    stats["failures"] = len(failures)
    stats.update(
        build_metadata(
            items,
            cache_root,
            metadata_out,
            transform_base_area=args.transform_base_area,
        )
    )
    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(cache_root / "metadata" / "moved_image_cache_stats.json", stats)
    print(f"[done] metadata_out={metadata_out}", flush=True)
    print(
        f"[done] stats={cache_root / 'metadata' / 'moved_image_cache_stats.json'}",
        flush=True,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
