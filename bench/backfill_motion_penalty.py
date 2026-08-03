"""Stable relocation-aware motion metric engine used by bench.evaluation."""

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from queue import Empty

import numpy as np
import torch
import torch.nn.functional as F

# load .env variables if exists
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.tools import DepthModel  # noqa: E402
from bench.utils.benchmark_metadata import load_benchmark_metadata  # noqa: E402
from bench.utils.coord_distance import (  # noqa: E402
    cal_motion_projection_penalty,
    masked_depth_to_points_3d,
)
from bench.utils.image_process import bbox_to_binary_mask  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

FIXED_IMAGE_SIZE = (768, 432)
FIXED_MASK_SHAPE = (FIXED_IMAGE_SIZE[1], FIXED_IMAGE_SIZE[0])
FILENAME_PATTERNS = {
    "res": re.compile(r"^(?P<bench>\d{4})_res(?P<sample>\d+)\.png$"),
    "seed": re.compile(r"^(?P<bench>\d{4})_seed(?P<sample>\d+)\.png$"),
}


def _match_filename_mode(filename: str):
    for mode, pattern in FILENAME_PATTERNS.items():
        if pattern.match(filename):
            return mode
    return None


def _align_depth_outputs(depths, valid_masks, intrinsics, target_h, target_w):
    if depths.ndim != 3 or valid_masks.ndim != 3:
        return depths, valid_masks, intrinsics

    _, h, w = depths.shape
    if (h, w) == (target_h, target_w):
        return depths, valid_masks, intrinsics

    aligned_depths = F.interpolate(
        depths.unsqueeze(1).float(), size=(target_h, target_w), mode="bilinear", align_corners=False
    ).squeeze(1)
    aligned_valid_masks = F.interpolate(
        valid_masks.unsqueeze(1).float(), size=(target_h, target_w), mode="nearest"
    ).squeeze(1)

    aligned_intrinsics = intrinsics.clone().float()
    sx = target_w / max(w, 1)
    sy = target_h / max(h, 1)
    aligned_intrinsics[:, 0, 0] *= sx
    aligned_intrinsics[:, 1, 1] *= sy
    aligned_intrinsics[:, 0, 2] *= sx
    aligned_intrinsics[:, 1, 2] *= sy
    return aligned_depths, aligned_valid_masks, aligned_intrinsics


def _load_mask_array(mask_like):
    if torch.is_tensor(mask_like):
        mask_np = mask_like.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask_like)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.shape != FIXED_MASK_SHAPE:
        mask_np = np.array(Image.fromarray(mask_np).resize(FIXED_IMAGE_SIZE, resample=Image.Resampling.NEAREST))
    return mask_np


def _scale_bbox_to_fixed(bbox, ref_w, ref_h):
    x_scale = FIXED_IMAGE_SIZE[0] / max(ref_w, 1)
    y_scale = FIXED_IMAGE_SIZE[1] / max(ref_h, 1)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = round(min(max(x1, 0), ref_w - 1) * x_scale)
    y1 = round(min(max(y1, 0), ref_h - 1) * y_scale)
    x2 = round(min(max(x2, 0), ref_w - 1) * x_scale)
    y2 = round(min(max(y2, 0), ref_h - 1) * y_scale)
    return [x1, y1, x2, y2]


def _load_artifact_record_map(artifact_root_dir: Path):
    metadata_path = artifact_root_dir / "localization_results.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find artifact metadata: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    record_map = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        image_name = str(rec.get("image_name", "")).strip()
        if image_name:
            record_map[image_name] = rec
    return record_map


def _build_sample_cache_item(benchmark_item):
    ref_image_raw = Image.open(benchmark_item["f1_path"]).convert("RGB")
    gt_image_raw = Image.open(benchmark_item["f2_path"]).convert("RGB")
    ref_w, ref_h = ref_image_raw.size

    src_image = ref_image_raw.resize(FIXED_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
    gt_image = gt_image_raw.resize(FIXED_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

    src_masks = [_load_mask_array(np.array(Image.open(p).convert("L"))) for p in benchmark_item["f1_mask_path"]]
    gt_masks = [_load_mask_array(np.array(Image.open(p).convert("L"))) for p in benchmark_item["f2_mask_path"]]

    src_bboxes = [_scale_bbox_to_fixed(b, ref_w, ref_h) for b in benchmark_item["f1_obj_bbox"]]
    target_bboxes = [_scale_bbox_to_fixed(b, ref_w, ref_h) for b in benchmark_item["f2_obj_bbox"]]

    return {
        "src_image": src_image,
        "gt_image": gt_image,
        "src_masks": src_masks,
        "gt_masks": gt_masks,
        "src_bboxes": src_bboxes,
        "target_bboxes": target_bboxes,
    }


def _load_pred_obj_mask(image_name, obj_idx, artifact_root_dir, artifact_record, fallback_bbox, device):
    masks_dir = artifact_root_dir / "masks"
    obj_records = artifact_record.get("objects", []) if isinstance(artifact_record, dict) else []

    if obj_idx < len(obj_records) and isinstance(obj_records[obj_idx], dict):
        mask_filename = obj_records[obj_idx].get("mask_filename", f"{image_name}_obj{obj_idx:02d}_mask.png")
        mask_path = masks_dir / str(mask_filename)
        if mask_path.exists():
            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            if mask_np.shape != FIXED_MASK_SHAPE:
                mask_np = np.array(Image.fromarray(mask_np).resize(FIXED_IMAGE_SIZE, resample=Image.Resampling.NEAREST))
            return torch.from_numpy(mask_np).to(device=device, dtype=torch.float32) / 255.0

        final_bbox_abs = obj_records[obj_idx].get("final_bbox_abs")
        if isinstance(final_bbox_abs, (list, tuple)) and len(final_bbox_abs) == 4:
            return bbox_to_binary_mask(FIXED_IMAGE_SIZE, [float(v) for v in final_bbox_abs], device=device)

    return bbox_to_binary_mask(FIXED_IMAGE_SIZE, fallback_bbox, device=device)


def _run_backfill_once(
    benchmark_metadata,
    existing_result_json: Path,
    output_result_json: Path,
    output_image_dir: Path,
    artifact_root_dir: Path,
    depth_model,
    motion_alpha: float,
    motion_beta: float,
    motion_static_motion_ratio: float,
):
    with open(existing_result_json, "r", encoding="utf-8") as f:
        grouped_payload = json.load(f)

    benchmark_by_index = {int(item["bench_index"]): item for item in benchmark_metadata}
    artifact_record_map = _load_artifact_record_map(artifact_root_dir)

    sample_cache = {}
    matched_num_samples = 0
    detected_filename_mode = None

    total_samples = sum(len(group.get("samples", [])) for group in grouped_payload if isinstance(group, dict))
    progress = tqdm(total=total_samples, desc=f"Backfilling {existing_result_json.parent.name}")

    for group in grouped_payload:
        if not isinstance(group, dict):
            progress.update(1)
            continue

        bench_index = int(group.get("bench_index", -1))
        bench_item = benchmark_by_index.get(bench_index)
        if bench_item is None:
            for _ in group.get("samples", []):
                progress.update(1)
            continue

        if bench_index not in sample_cache:
            sample_cache[bench_index] = _build_sample_cache_item(bench_item)
        bench_cache = sample_cache[bench_index]

        src_image = bench_cache["src_image"]
        gt_image = bench_cache["gt_image"]
        src_masks = bench_cache["src_masks"]
        gt_masks = bench_cache["gt_masks"]
        target_bboxes = bench_cache["target_bboxes"]

        for sample_entry in group.get("samples", []):
            filename = str(sample_entry.get("filename", ""))
            matched_mode = _match_filename_mode(filename)
            if matched_mode is None:
                progress.update(1)
                continue

            if detected_filename_mode is None:
                detected_filename_mode = matched_mode
                print(f"Detected filename mode: {detected_filename_mode}")
            elif matched_mode != detected_filename_mode:
                raise ValueError(
                    f"Mixed filename modes in one run: detected={detected_filename_mode}, current={matched_mode}, filename={filename}"
                )

            matched_num_samples += 1

            image_name = Path(filename).stem
            generated_path = output_image_dir / filename
            if not generated_path.exists():
                print(f"Warning: generated image not found, skip sample: {generated_path}")
                progress.update(1)
                continue

            generated_image = Image.open(generated_path).convert("RGB")
            if generated_image.size != FIXED_IMAGE_SIZE:
                generated_image = generated_image.resize(FIXED_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

            batch_depth, batch_valid_masks, batch_intrinsics, batch_extrinsics = depth_model.predict_depth(
                [src_image, gt_image], batch_type="view", process_res=max(FIXED_IMAGE_SIZE)
            )
            batch_depth, batch_valid_masks, batch_intrinsics = _align_depth_outputs(
                batch_depth, batch_valid_masks, batch_intrinsics, FIXED_MASK_SHAPE[0], FIXED_MASK_SHAPE[1]
            )
            src_depth = batch_depth[0]
            src_valid_mask = batch_valid_masks[0] > 0.5
            src_intrinsic = batch_intrinsics[0]
            src_extrinsic = batch_extrinsics[0]
            gt_depth = batch_depth[1]
            gt_valid_mask = batch_valid_masks[1] > 0.5
            gt_intrinsic = batch_intrinsics[1]
            gt_extrinsic = batch_extrinsics[1]

            batch_depth, batch_valid_masks, batch_intrinsics, batch_extrinsics = depth_model.predict_depth(
                [src_image, gt_image, generated_image], batch_type="view", process_res=max(FIXED_IMAGE_SIZE)
            )
            batch_depth, batch_valid_masks, batch_intrinsics = _align_depth_outputs(
                batch_depth, batch_valid_masks, batch_intrinsics, FIXED_MASK_SHAPE[0], FIXED_MASK_SHAPE[1]
            )
            gen_depth = batch_depth[2]
            gen_valid_mask = batch_valid_masks[2] > 0.5
            gen_intrinsic = batch_intrinsics[2]
            gen_extrinsic = batch_extrinsics[2]

            joint_valid_mask = gen_valid_mask & gt_valid_mask
            artifact_record = artifact_record_map.get(image_name, {})

            objs = sample_entry.get("objs", [])
            assert len(src_masks) == len(gt_masks) == len(target_bboxes) == len(objs), (
                f"Mismatch in object count for sample {filename}"
            )
            obj_count = len(objs)
            for obj_idx in range(obj_count):
                pred_mask = _load_pred_obj_mask(
                    image_name=image_name,
                    obj_idx=obj_idx,
                    artifact_root_dir=artifact_root_dir,
                    artifact_record=artifact_record,
                    fallback_bbox=target_bboxes[obj_idx],
                    device=gen_depth.device,
                )

                pts_pred = masked_depth_to_points_3d(
                    depth_map=gen_depth,
                    mask=(torch.as_tensor(pred_mask, device=gen_depth.device) > 0.5) & joint_valid_mask,
                    intrinsic=gen_intrinsic,
                    extrinsic=gen_extrinsic,
                )
                pts_orig = masked_depth_to_points_3d(
                    depth_map=src_depth,
                    mask=(torch.as_tensor(src_masks[obj_idx], device=src_depth.device) > 0.5) & src_valid_mask,
                    intrinsic=src_intrinsic,
                    extrinsic=src_extrinsic,
                )
                pts_gt = masked_depth_to_points_3d(
                    depth_map=gt_depth,
                    mask=(torch.as_tensor(gt_masks[obj_idx], device=gt_depth.device) > 0.5) & joint_valid_mask,
                    intrinsic=gt_intrinsic,
                    extrinsic=gt_extrinsic,
                )

                motion_result = cal_motion_projection_penalty(
                    points_orig=pts_orig,
                    points_pred=pts_pred,
                    points_gt=pts_gt,
                    alpha=motion_alpha,
                    beta=motion_beta,
                    static_motion_ratio=motion_static_motion_ratio,
                    return_details=True,
                )
                if len(motion_result) == 3:
                    motion_proj_raw, motion_proj_penalty, motion_details = motion_result
                else:
                    motion_proj_raw, motion_proj_penalty = motion_result
                    motion_details = {}

                obj_entry = objs[obj_idx]
                obj_entry["motion_proj_ratio_raw"] = motion_proj_raw
                obj_entry["motion_proj_penalty"] = motion_proj_penalty
                obj_entry["motion_proj_formula"] = motion_details.get("motion_proj_formula")
                obj_entry["motion_proj_alpha"] = motion_details.get("alpha")
                obj_entry["motion_proj_beta"] = motion_details.get("beta")
                obj_entry["motion_proj_static_motion_ratio"] = motion_details.get("static_motion_ratio")
                obj_entry["motion_proj_gate"] = motion_details.get("gate")
                obj_entry["motion_proj_err_parallel"] = motion_details.get("err_parallel")
                obj_entry["motion_proj_err_perp"] = motion_details.get("err_perp")
                obj_entry["motion_proj_dot"] = motion_details.get("dot")
                obj_entry["motion_proj_gt_norm"] = motion_details.get("gt_norm")
                obj_entry["motion_proj_pred_norm"] = motion_details.get("pred_norm")
                obj_entry["motion_proj_is_static_case"] = motion_details.get("is_static_case")
                obj_entry["motion_proj_static_threshold"] = motion_details.get("static_threshold")

                if "obj_sim" in obj_entry and obj_entry["obj_sim"] is not None:
                    obj_sim = max(float(obj_entry["obj_sim"]), 0.0)
                    obj_entry["obj_sim_penalized"] = obj_sim * motion_proj_penalty

            progress.update(1)

    progress.close()
    if matched_num_samples == 0:
        print("Warning: no samples matched the expected filename pattern, check if the pattern is correct.")
    else:
        output_result_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_result_json, "w", encoding="utf-8") as f:
            json.dump(grouped_payload, f, ensure_ascii=False, indent=2)

        print(f"Saved backfilled result JSON to: {output_result_json}")


def _parse_gpu_ids(gpu_ids_arg):
    if gpu_ids_arg is None:
        return []
    text = str(gpu_ids_arg).strip()
    if not text:
        return []
    ids = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        ids.append(int(p))
    # Keep user-provided order while removing duplicates.
    unique_ids = []
    seen = set()
    for gpu_id in ids:
        if gpu_id in seen:
            continue
        seen.add(gpu_id)
        unique_ids.append(gpu_id)
    return unique_ids


def _batch_worker(
    worker_name: str,
    gpu_id: int,
    task_queue,
    result_queue,
    benchmark_metadata,
    motion_alpha: float,
    motion_beta: float,
    motion_static_motion_ratio: float,
):
    torch.cuda.set_device(gpu_id)
    try:
        device = torch.device("cuda", gpu_id)
        depth_model = DepthModel(device=device)
    except Exception as e:
        result_queue.put(("worker_failed", worker_name, f"init failed on GPU {gpu_id}: {e}"))
        return

    while True:
        try:
            subdir_str = task_queue.get_nowait()
        except Empty:
            break

        subdir = Path(subdir_str)
        in_json = subdir / "bench_score_new.json"
        images_dir = subdir / "images"
        artifact_dir = subdir / "process_artifacts"
        out_json = subdir / "bench_score_penalty_deqa.json"

        if not in_json.exists():
            result_queue.put(("skipped", subdir.name, "missing bench_score_new.json"))
            continue

        if not images_dir.exists() or not artifact_dir.exists():
            result_queue.put(("skipped", subdir.name, "missing images/ or process_artifacts/"))
            continue

        print(f"[{worker_name}] GPU {gpu_id} processing {subdir.name} ...")
        try:
            _run_backfill_once(
                benchmark_metadata=benchmark_metadata,
                existing_result_json=in_json,
                output_result_json=out_json,
                output_image_dir=images_dir,
                artifact_root_dir=artifact_dir,
                depth_model=depth_model,
                motion_alpha=motion_alpha,
                motion_beta=motion_beta,
                motion_static_motion_ratio=motion_static_motion_ratio,
            )
            result_queue.put(("processed", subdir.name, ""))
        except Exception as e:
            result_queue.put(("failed", subdir.name, str(e)))


def main():
    parser = argparse.ArgumentParser(
        description="Backfill motion projection penalty fields into existing benchmark JSON."
    )
    parser.add_argument("--benchmark-metadata-path", required=True)
    parser.add_argument(
        "--existing-result-json",
        required=True,
    )
    parser.add_argument(
        "--output-result-json",
        required=True,
    )
    parser.add_argument("--output-image-dir", required=True)
    parser.add_argument(
        "--artifact-root-dir",
        required=True,
    )
    parser.add_argument("--batch-root-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3",
        help="Comma-separated physical GPU IDs for parallel batch mode, e.g. '0,1,2'.",
    )
    parser.add_argument("--motion-alpha", type=float, default=1.0)
    parser.add_argument("--motion-beta", type=float, default=0.7)
    parser.add_argument("--motion-static-motion-ratio", type=float, default=0.02)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.append(str(project_root))

    benchmark_metadata_path = Path(args.benchmark_metadata_path)
    existing_result_json = Path(args.existing_result_json)
    output_result_json = Path(args.output_result_json) if args.output_result_json else existing_result_json
    output_image_dir = Path(args.output_image_dir)
    artifact_root_dir = Path(args.artifact_root_dir)
    batch_root_dir = Path(args.batch_root_dir) if args.batch_root_dir else None

    benchmark_metadata = load_benchmark_metadata(benchmark_metadata_path)

    if batch_root_dir is not None:
        if not batch_root_dir.exists():
            raise FileNotFoundError(f"batch root dir not found: {batch_root_dir}")

        candidate_subdirs = sorted([p for p in batch_root_dir.iterdir() if p.is_dir()])
        print(f"Scanning subdirectories under: {batch_root_dir}")

        gpu_ids = _parse_gpu_ids(args.gpu_ids)
        if args.device.startswith("cuda") and len(gpu_ids) > 1:
            print(f"Running multi-GPU pipeline mode on GPUs: {gpu_ids}")
            ctx = mp.get_context("spawn")
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()

            for subdir in candidate_subdirs:
                task_queue.put(str(subdir))

            workers = []
            for idx, gpu_id in enumerate(gpu_ids):
                worker_name = f"worker-{idx}"
                p = ctx.Process(
                    target=_batch_worker,
                    args=(
                        worker_name,
                        gpu_id,
                        task_queue,
                        result_queue,
                        benchmark_metadata,
                        args.motion_alpha,
                        args.motion_beta,
                        args.motion_static_motion_ratio,
                    ),
                )
                p.start()
                workers.append(p)

            for p in workers:
                p.join()

            processed = 0
            skipped = 0
            failed = 0
            worker_failed = 0
            while True:
                try:
                    status, name, msg = result_queue.get_nowait()
                except Empty:
                    break
                if status == "processed":
                    processed += 1
                elif status == "skipped":
                    skipped += 1
                    print(f"Skip {name}: {msg}")
                elif status == "failed":
                    failed += 1
                    print(f"Failed {name}: {msg}")
                elif status == "worker_failed":
                    worker_failed += 1
                    print(f"Worker failure {name}: {msg}")

            print("\n===== Batch Summary =====")
            print(f"Processed subdirs: {processed}")
            print(f"Skipped subdirs: {skipped}")
            print(f"Failed subdirs: {failed}")
            print(f"Worker init failures: {worker_failed}")
            return

        depth_model = DepthModel(device=args.device)

        processed = 0
        skipped = 0
        failed = 0
        for subdir in candidate_subdirs:
            in_json = subdir / "bench_score_new.json"
            images_dir = subdir / "images"
            artifact_dir = subdir / "process_artifacts"
            out_json = subdir / "bench_score_penalty_deqa.json"

            if not in_json.exists():
                skipped += 1
                continue

            if not images_dir.exists() or not artifact_dir.exists():
                print(f"Skip {subdir.name}: missing images/ or process_artifacts/")
                skipped += 1
                continue

            print(f"Processing {subdir.name} ...")
            try:
                _run_backfill_once(
                    benchmark_metadata=benchmark_metadata,
                    existing_result_json=in_json,
                    output_result_json=out_json,
                    output_image_dir=images_dir,
                    artifact_root_dir=artifact_dir,
                    depth_model=depth_model,
                    motion_alpha=args.motion_alpha,
                    motion_beta=args.motion_beta,
                    motion_static_motion_ratio=args.motion_static_motion_ratio,
                )
                processed += 1
            except Exception as e:
                failed += 1
                print(f"Failed {subdir.name}: {e}")

        print("\n===== Batch Summary =====")
        print(f"Processed subdirs: {processed}")
        print(f"Skipped subdirs: {skipped}")
        print(f"Failed subdirs: {failed}")
        return

    depth_model = DepthModel(device=args.device)

    _run_backfill_once(
        benchmark_metadata=benchmark_metadata,
        existing_result_json=existing_result_json,
        output_result_json=output_result_json,
        output_image_dir=output_image_dir,
        artifact_root_dir=artifact_root_dir,
        depth_model=depth_model,
        motion_alpha=args.motion_alpha,
        motion_beta=args.motion_beta,
        motion_static_motion_ratio=args.motion_static_motion_ratio,
    )


if __name__ == "__main__":
    main()
