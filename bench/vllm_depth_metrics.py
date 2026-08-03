"""Stable core geometry/DINO metric engine used by bench.evaluation."""

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.evaluation.grounding_cache import (  # noqa: E402
    GroundingResultCache,
    build_grounding_cache_signature,
)
from bench.tools import DepthModel, DINOModel, SAMModel  # noqa: E402
from bench.utils.benchmark_metadata import load_benchmark_metadata  # noqa: E402
from bench.utils.coord_distance import (  # noqa: E402
    cal_bbox_diou,
    cal_chamfer_and_centroid_distance,
    cal_depth_absrel_and_delta,
    cal_mask_iou,
    cal_motion_projection_penalty,
    cal_pointcloud_3d_metrics,
    masked_depth_to_points_3d,
    norm_z_coords,
)
from bench.utils.image_process import bbox_to_binary_mask, save_mask_overlay_debug  # noqa: E402
from bench.utils.text_process import get_simple_edit_prompt  # noqa: E402
from bench.vlm import VLMModel  # noqa: E402
from bench.vlm.vlm import GROUNDING_PROMPT_TEMPLATE  # noqa: E402


class MetricSuite:
    def init_models(self, device: torch.device | str = torch.device("cuda")):
        self.sam_model = SAMModel(device=device)
        self.depth_model = DepthModel(device=device)
        self.dino_model = DINOModel(device=device)

    def init_vlm_model(self):
        self.vlm_model = VLMModel()

    def cal_multi_metrics(
        self,
        generated_images,
        src_images,
        coordinate_lists,
        src_obj_bboxes,
        src_obj_masks,
        obj_names,
        gt_images,
        gt_obj_masks,
        recorded_src_depths,
        recorded_gt_depths,
        target_obj_bboxes,
        process_artifact_dir: str | Path | None = None,
        artifact_image_names: list[str] | None = None,
        debug_dir: str | Path | None = None,
    ):
        MAX_DINO_BATCH_SIZE = 128
        MAX_SAM_BBOX_BATCH_SIZE = 4
        fixed_image_size = (768, 432)
        fixed_mask_shape = (fixed_image_size[1], fixed_image_size[0])

        artifact_root_dir = Path(process_artifact_dir) if process_artifact_dir is not None else None
        artifact_masks_dir = None
        artifact_metadata_path = None
        if artifact_root_dir is not None:
            artifact_masks_dir = artifact_root_dir / "masks"
            artifact_metadata_path = artifact_root_dir / "localization_results.json"
            artifact_root_dir.mkdir(parents=True, exist_ok=True)
            artifact_masks_dir.mkdir(parents=True, exist_ok=True)

        full_results = []
        # [{"deqa_score":..., "clip_score": ..., ... , {objs: [ {"2d_dist":..., "depth_dist":..., "obj_sim":...}, ... ] } ]
        # Build local, resized copies to avoid mutating shared input references.
        processed_generated_images = []
        processed_src_images = []
        processed_src_obj_masks = []
        processed_src_obj_bboxes = []
        processed_target_obj_bboxes = []
        processed_obj_names = []

        for i in range(len(generated_images)):
            generated_image = generated_images[i]
            src_image = src_images[i]

            ref_w, ref_h = src_image.size
            x_scale = fixed_image_size[0] / max(ref_w, 1)
            y_scale = fixed_image_size[1] / max(ref_h, 1)

            if generated_image.size != fixed_image_size:
                generated_image = generated_image.resize(fixed_image_size, resample=Image.Resampling.BILINEAR)
            if src_image.size != fixed_image_size:
                src_image = src_image.resize(fixed_image_size, resample=Image.Resampling.BILINEAR)

            def _scale_bbox_to_fixed(bbox):
                x1, y1, x2, y2 = [float(v) for v in bbox]
                x1 = round(min(max(x1, 0), ref_w - 1) * x_scale)
                y1 = round(min(max(y1, 0), ref_h - 1) * y_scale)
                x2 = round(min(max(x2, 0), ref_w - 1) * x_scale)
                y2 = round(min(max(y2, 0), ref_h - 1) * y_scale)
                return [x1, y1, x2, y2]

            image_src_masks = []
            image_src_bboxes = []
            image_target_bboxes = []
            image_obj_names = list(obj_names[i])

            for obj_idx in range(len(src_obj_masks[i])):
                obj_mask = src_obj_masks[i][obj_idx]
                if torch.is_tensor(obj_mask):
                    mask_np = obj_mask.detach().cpu().numpy()
                else:
                    mask_np = np.asarray(obj_mask)

                if mask_np.ndim > 2:
                    mask_np = np.squeeze(mask_np)

                if mask_np.shape != fixed_mask_shape:
                    mask_np = np.array(
                        Image.fromarray(mask_np).resize(fixed_image_size, resample=Image.Resampling.NEAREST)
                    )

                image_src_masks.append(torch.from_numpy(mask_np))

                src_bbox = deepcopy(src_obj_bboxes[i][obj_idx])
                image_src_bboxes.append(_scale_bbox_to_fixed(src_bbox))

                raw_target_bbox = (
                    target_obj_bboxes[i][obj_idx] if target_obj_bboxes is not None else src_obj_bboxes[i][obj_idx]
                )
                image_target_bboxes.append(_scale_bbox_to_fixed(deepcopy(raw_target_bbox)))

            processed_generated_images.append(generated_image)
            processed_src_images.append(src_image)
            processed_src_obj_masks.append(image_src_masks)
            processed_src_obj_bboxes.append(image_src_bboxes)
            processed_target_obj_bboxes.append(image_target_bboxes)
            processed_obj_names.append(image_obj_names)

            image_obj_num = len(image_src_bboxes)
            single_image_results = {
                "objs": [
                    {
                        "may_lost": False,
                        "used_cached_localization": False,
                        "localization_source": "target_bbox_init",
                    }
                    for _ in range(image_obj_num)
                ]
            }
            full_results.append(single_image_results)

        generated_images = processed_generated_images
        src_images = processed_src_images
        src_obj_masks = processed_src_obj_masks
        src_obj_bboxes = processed_src_obj_bboxes
        target_obj_bboxes = processed_target_obj_bboxes
        obj_names = processed_obj_names

        src_depths = []
        src_intrinsics = []
        src_extrinsics = []
        gen_depths = []
        gen_valid_masks = []
        gen_intrinsics = []
        gen_extrinsics = []
        gt_depths = []
        gt_valid_masks = []
        gt_intrinsics = []
        gt_extrinsics = []
        target_h, target_w = fixed_mask_shape

        def _align_depth_outputs(depths, valid_masks, intrinsics):
            # Align depth/mask to evaluation resolution and keep intrinsics consistent.
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

        for src_image, gt_image, generated_image in tqdm(
            zip(src_images, gt_images, generated_images), total=len(generated_images), desc="Predicting depth maps"
        ):
            batch_depth, batch_valid_masks, batch_intrinsics, batch_extrinsics = self.depth_model.predict_depth(
                [src_image, gt_image], batch_type="view", process_res=max(fixed_image_size)
            )
            batch_depth, batch_valid_masks, batch_intrinsics = _align_depth_outputs(
                batch_depth, batch_valid_masks, batch_intrinsics
            )
            src_depths.append(batch_depth[0])
            src_intrinsics.append(batch_intrinsics[0])
            src_extrinsics.append(batch_extrinsics[0])
            gt_depths.append(batch_depth[1])
            gt_valid_masks.append(batch_valid_masks[1])
            gt_intrinsics.append(batch_intrinsics[1])
            gt_extrinsics.append(batch_extrinsics[1])

            batch_depth, batch_valid_masks, batch_intrinsics, batch_extrinsics = self.depth_model.predict_depth(
                [src_image, gt_image, generated_image], batch_type="view", process_res=max(fixed_image_size)
            )
            batch_depth, batch_valid_masks, batch_intrinsics = _align_depth_outputs(
                batch_depth, batch_valid_masks, batch_intrinsics
            )
            gen_depths.append(batch_depth[2])
            gen_valid_masks.append(batch_valid_masks[2])
            gen_intrinsics.append(batch_intrinsics[2])
            gen_extrinsics.append(batch_extrinsics[2])

        src_depths = torch.stack(src_depths, dim=0)
        src_intrinsics = torch.stack(src_intrinsics, dim=0)
        src_extrinsics = torch.stack(src_extrinsics, dim=0)
        gen_depths = torch.stack(gen_depths, dim=0)
        gen_valid_masks = torch.stack(gen_valid_masks, dim=0)
        gen_intrinsics = torch.stack(gen_intrinsics, dim=0)
        gen_extrinsics = torch.stack(gen_extrinsics, dim=0)
        gt_depths = torch.stack(gt_depths, dim=0)
        gt_valid_masks = torch.stack(gt_valid_masks, dim=0)
        gt_intrinsics = torch.stack(tensors=gt_intrinsics, dim=0)
        gt_extrinsics = torch.stack(gt_extrinsics, dim=0)

        simple_edit_prompts = []
        normed_coordinate_lists = norm_z_coords(
            coordinate_lists=coordinate_lists, reference_depths=recorded_src_depths, gt_depths=recorded_gt_depths
        )
        for i in range(len(generated_images)):
            object_edit_prompt: list[str | None] = [None] * len(obj_names[i])
            edit_prompt = get_simple_edit_prompt(
                object_name=obj_names[i],
                coordinates=normed_coordinate_lists[i],
                object_edit_prompt=object_edit_prompt,
                additional_prompt="",
                xy_coord_range="neg1_1",
            )
            simple_edit_prompts.append(edit_prompt)

        generated_obj_masks = []
        fallback_obj_bboxes = []
        vlm_detected_norm_boxes: list[list[list[float] | None]] = [
            [None for _ in image_bboxes] for image_bboxes in src_obj_bboxes
        ]
        vlm_grounding_texts: list[list[str | None]] = [[None for _ in image_bboxes] for image_bboxes in src_obj_bboxes]
        for i in range(len(src_obj_bboxes)):
            image_masks = []
            image_fallback_bboxes = []
            for obj_idx in range(len(src_obj_bboxes[i])):
                fallback_bbox = (
                    target_obj_bboxes[i][obj_idx] if target_obj_bboxes is not None else src_obj_bboxes[i][obj_idx]
                )

                fallback_bbox = [int(v) for v in fallback_bbox]
                image_fallback_bboxes.append(fallback_bbox)

                image_masks.append(
                    bbox_to_binary_mask(
                        image_size=generated_images[i].size,
                        bbox=fallback_bbox,
                        device=gen_depths.device,
                    )
                )
                vlm_grounding_texts[i][obj_idx] = obj_names[i][obj_idx]
            generated_obj_masks.append(image_masks)
            fallback_obj_bboxes.append(image_fallback_bboxes)

        detected_abs_boxes = [[bbox[:] for bbox in image_bboxes] for image_bboxes in fallback_obj_bboxes]
        vlm_request_failed_flags = [[False for _ in image_bboxes] for image_bboxes in src_obj_bboxes]
        reused_obj_flags = [[False for _ in image_bboxes] for image_bboxes in src_obj_bboxes]
        reused_obj_count = 0
        total_obj_count = sum(len(image_bboxes) for image_bboxes in src_obj_bboxes)
        if artifact_metadata_path is not None and artifact_masks_dir is not None:
            if artifact_metadata_path.exists() and artifact_masks_dir.exists():
                print(f"Found localization artifacts, trying to reuse: {artifact_metadata_path}")
                try:
                    with open(artifact_metadata_path, "r", encoding="utf-8") as f:
                        cached_records = json.load(f)

                    record_map = {
                        str(record.get("image_name", "")): record
                        for record in cached_records
                        if isinstance(record, dict)
                    }
                    for img_idx in range(len(generated_images)):
                        image_name = (
                            artifact_image_names[img_idx]
                            if artifact_image_names is not None and img_idx < len(artifact_image_names)
                            else f"img_{img_idx:04d}"
                        )
                        image_record = record_map.get(str(image_name))
                        if image_record is None:
                            continue

                        obj_records = image_record.get("objects", [])
                        if not isinstance(obj_records, list):
                            continue

                        for obj_idx in range(len(src_obj_bboxes[img_idx])):
                            if obj_idx >= len(obj_records):
                                continue

                            obj_record = obj_records[obj_idx]
                            if not isinstance(obj_record, dict):
                                continue
                            if bool(obj_record.get("vlm_failed", False)):
                                continue

                            cached_bbox = obj_record.get("final_bbox_abs")
                            has_valid_bbox = isinstance(cached_bbox, (list, tuple)) and len(cached_bbox) == 4
                            if not has_valid_bbox:
                                continue

                            assert cached_bbox is not None
                            bbox_values = list(cached_bbox)
                            bbox_abs = [int(round(float(v))) for v in bbox_values]

                            full_results[img_idx]["objs"][obj_idx]["may_lost"] = bool(obj_record.get("may_lost", False))
                            full_results[img_idx]["objs"][obj_idx]["used_cached_localization"] = True
                            full_results[img_idx]["objs"][obj_idx]["localization_source"] = "cached_mask"
                            vlm_bbox_norm = obj_record.get("vlm_bbox_norm")
                            if isinstance(vlm_bbox_norm, (list, tuple)) and len(vlm_bbox_norm) == 4:
                                vlm_detected_norm_boxes[img_idx][obj_idx] = [float(v) for v in vlm_bbox_norm]

                            grounding_text = str(obj_record.get("grounding_text", "")).strip()
                            vlm_grounding_texts[img_idx][obj_idx] = grounding_text or obj_names[img_idx][obj_idx]

                            mask_filename = obj_record.get("mask_filename", f"{image_name}_obj{obj_idx:02d}_mask.png")
                            mask_path = artifact_masks_dir / str(mask_filename)
                            if not mask_path.exists():
                                continue

                            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
                            if mask_np.shape != fixed_mask_shape:
                                mask_np = np.array(
                                    Image.fromarray(mask_np).resize(fixed_image_size, resample=Image.Resampling.NEAREST)
                                )

                            detected_abs_boxes[img_idx][obj_idx] = bbox_abs
                            generated_obj_masks[img_idx][obj_idx] = (
                                torch.from_numpy(mask_np).to(device=gen_depths.device, dtype=torch.float32) / 255.0
                            )
                            reused_obj_flags[img_idx][obj_idx] = True
                            reused_obj_count += 1

                    print(f"Reused cached localization objects: {reused_obj_count}/{total_obj_count}")
                except Exception as e:
                    print(f"Failed to load cached localization artifacts ({e}), fallback to VLM/SAM for all objects.")

        missing_obj_count = total_obj_count - reused_obj_count
        if missing_obj_count > 0:
            print(f"Detecting remaining objects with VLM/SAM... ({missing_obj_count} objects)")
            flat_indices = []
            detect_refs = []
            detect_gens = []
            detect_orig_boxes = []
            detect_target_boxes = []
            detect_obj_names = []
            detect_cache_keys = []
            for i in range(len(generated_images)):
                for obj_idx in range(len(src_obj_bboxes[i])):
                    if reused_obj_flags[i][obj_idx]:
                        continue

                    flat_indices.append((i, obj_idx))
                    detect_refs.append(src_images[i])
                    detect_gens.append(generated_images[i])
                    detect_orig_boxes.append(src_obj_bboxes[i][obj_idx])

                    current_target_bbox = target_obj_bboxes[i][obj_idx] if target_obj_bboxes is not None else None
                    detect_target_boxes.append(current_target_bbox)

                    detect_obj_names.append(obj_names[i][obj_idx])

                    image_name = (
                        artifact_image_names[i]
                        if artifact_image_names is not None and i < len(artifact_image_names)
                        else f"img_{i:04d}"
                    )
                    detect_cache_keys.append(f"{image_name}:obj{obj_idx:02d}")

            grounding_cache = None
            cached_grounding_results = None
            if artifact_root_dir is not None:
                grounding_cache = GroundingResultCache(
                    artifact_root_dir / "grounding_results.json",
                    build_grounding_cache_signature(
                        base_url=self.vlm_model.base_url,
                        model_name=self.vlm_model.model_name,
                        prompt_template=GROUNDING_PROMPT_TEMPLATE,
                    ),
                )
                cached_grounding_results = grounding_cache.successful_items()

            vlm_detected_results = self.vlm_model.detect_object_vllm(
                original_images=detect_refs,
                edited_images=detect_gens,
                orig_bboxes=detect_orig_boxes,
                obj_names=detect_obj_names,
                target_bboxes=detect_target_boxes,
                cache_keys=detect_cache_keys,
                cached_results=cached_grounding_results,
                on_result=grounding_cache.record if grounding_cache is not None else None,
            )

            if grounding_cache is not None:
                cache_counts = grounding_cache.counts()
                print(
                    "Grounding cache: "
                    f"successful={cache_counts['successful']}, "
                    f"failed={cache_counts['failed']}, total={cache_counts['total']}"
                )

            valid_sam_indices = []
            valid_sam_images = []
            valid_sam_bboxes = []
            valid_sam_obj_names = []
            for flat_idx, (img_idx, obj_idx) in enumerate(flat_indices):
                vlm_result = vlm_detected_results[flat_idx]
                if isinstance(vlm_result, dict):
                    vlm_box = vlm_result.get("bbox")
                    sam_text = str(vlm_result.get("object_description", "")).strip()
                    vlm_request_failed_flags[img_idx][obj_idx] = bool(vlm_result.get("vlm_failed", False))
                else:
                    vlm_box = vlm_result
                    sam_text = ""

                if not sam_text:
                    sam_text = obj_names[img_idx][obj_idx]

                is_valid_box = (
                    isinstance(vlm_box, (list, tuple))
                    and len(vlm_box) == 4
                    and not (vlm_box[0] == 0 and vlm_box[1] == 0 and vlm_box[2] == 0 and vlm_box[3] == 0)
                    and (vlm_box[2] > vlm_box[0])
                    and (vlm_box[3] > vlm_box[1])
                )

                if is_valid_box:
                    assert vlm_box is not None
                    vlm_box_values = [float(v) for v in vlm_box]
                    vlm_detected_norm_boxes[img_idx][obj_idx] = vlm_box_values
                    vlm_grounding_texts[img_idx][obj_idx] = sam_text
                    width, height = generated_images[img_idx].size
                    abs_box = [
                        round(min(max(0, vlm_box_values[0] * width / 1000), width - 1)),
                        round(min(max(0, vlm_box_values[1] * height / 1000), height - 1)),
                        round(min(max(0, vlm_box_values[2] * width / 1000), width - 1)),
                        round(min(max(0, vlm_box_values[3] * height / 1000), height - 1)),
                    ]
                    detected_abs_boxes[img_idx][obj_idx] = abs_box
                    valid_sam_indices.append((img_idx, obj_idx))
                    valid_sam_images.append(generated_images[img_idx])
                    valid_sam_bboxes.append([abs_box])
                    valid_sam_obj_names.append(sam_text)
                else:
                    vlm_grounding_texts[img_idx][obj_idx] = sam_text
                    full_results[img_idx]["objs"][obj_idx]["may_lost"] = True
                    full_results[img_idx]["objs"][obj_idx]["localization_source"] = "target_bbox_init"
                    full_results[img_idx]["objs"][obj_idx]["vlm_failed"] = vlm_request_failed_flags[img_idx][obj_idx]

            failed_request_count = sum(sum(flags) for flags in vlm_request_failed_flags)
            if failed_request_count:
                print(
                    f"Warning: {failed_request_count} VLM grounding requests failed after retries; "
                    "they will remain retryable in localization artifacts."
                )

            if valid_sam_indices:
                sam_results = self.sam_model.get_masks_by_bboxes(
                    images=valid_sam_images,
                    texts=valid_sam_obj_names,
                    bboxes=valid_sam_bboxes,
                    batch_size=MAX_SAM_BBOX_BATCH_SIZE,
                )
                for (img_idx, obj_idx), sam_result in zip(valid_sam_indices, sam_results):
                    current_vlm_abs_box = detected_abs_boxes[img_idx][obj_idx]
                    scores = sam_result["scores"]
                    if len(scores) > 0:
                        refined_bbox = sam_result["boxes"]
                        refined_mask = sam_result["masks"]

                        # sort by scores and take the best one
                        sorted_indices = torch.argsort(scores, descending=True)
                        refined_mask = refined_mask[sorted_indices[0]]
                        refined_bbox = refined_bbox[sorted_indices[0]]
                        detected_abs_boxes[img_idx][obj_idx] = refined_bbox.cpu().numpy().tolist()

                        # if torch.sum(refined_mask > 0.5) >= 64:
                        if torch.sum(refined_mask > 0.5) > 0:
                            generated_obj_masks[img_idx][obj_idx] = refined_mask
                            full_results[img_idx]["objs"][obj_idx]["localization_source"] = "sam_mask"

                        elif (refined_bbox[2] - refined_bbox[0]) * (refined_bbox[3] - refined_bbox[1]) / (
                            generated_images[img_idx].size[0] * generated_images[img_idx].size[1]
                        ) >= 0.001:
                            # use refined bbox as the mask if the bbox is reasonably large, even if the mask is not good
                            generated_obj_masks[img_idx][obj_idx] = bbox_to_binary_mask(
                                image_size=generated_images[img_idx].size,
                                bbox=refined_bbox.cpu().numpy().tolist(),
                                device=refined_mask.device,
                            )
                            full_results[img_idx]["objs"][obj_idx]["localization_source"] = "sam_bbox_fallback"
                    else:
                        full_results[img_idx]["objs"][obj_idx]["may_lost"] = True
                        if current_vlm_abs_box is not None:
                            # if SAM fails to give any mask, we can at least use the VLM detected box as a fallback mask
                            generated_obj_masks[img_idx][obj_idx] = bbox_to_binary_mask(
                                image_size=generated_images[img_idx].size,
                                bbox=current_vlm_abs_box,
                                # Keep masks on the metric device for the later coordinate calculations.
                                device=gen_depths.device,
                            )
                            detected_abs_boxes[img_idx][obj_idx] = current_vlm_abs_box
                            full_results[img_idx]["objs"][obj_idx]["localization_source"] = "vlm_bbox_fallback"
                            print(
                                f"Warning: SAM failed to detect any mask for image {img_idx} obj {obj_idx}, using VLM detected box as fallback."
                            )
                        else:
                            # use the src bbox as the fallback, as a punishment on distance
                            generated_obj_masks[img_idx][obj_idx] = bbox_to_binary_mask(
                                image_size=generated_images[img_idx].size,
                                bbox=src_obj_bboxes[img_idx][obj_idx],
                                device=gen_depths.device,
                            )
                            detected_abs_boxes[img_idx][obj_idx] = src_obj_bboxes[img_idx][obj_idx]
                            full_results[img_idx]["objs"][obj_idx]["localization_source"] = "src_bbox_fallback"
                            print(
                                f"Warning: VLM also failed to detect any box for image {img_idx} obj {obj_idx}, using original box as fallback as a punishment on distance."
                            )
        else:
            print("All objects are fully covered by cached localization artifacts, skip VLM/SAM detection.")

        print("Preparing GT object masks for metric computation...")
        if gt_obj_masks is None or len(gt_obj_masks) != len(gt_images):
            raise ValueError("gt_obj_masks must be provided for every sample and match gt_images length.")

        processed_gt_obj_masks = []
        for img_idx in range(len(gt_images)):
            if gt_obj_masks[img_idx] is None or len(gt_obj_masks[img_idx]) != len(src_obj_bboxes[img_idx]):
                raise ValueError(
                    f"gt_obj_masks[{img_idx}] must contain one mask per object, "
                    f"expected {len(src_obj_bboxes[img_idx])}, got "
                    f"{0 if gt_obj_masks[img_idx] is None else len(gt_obj_masks[img_idx])}."
                )

            image_masks = []
            for obj_idx in range(len(src_obj_bboxes[img_idx])):
                provided_gt_mask = gt_obj_masks[img_idx][obj_idx]
                if torch.is_tensor(provided_gt_mask):
                    gt_mask_np = provided_gt_mask.detach().cpu().numpy()
                else:
                    gt_mask_np = np.asarray(provided_gt_mask)

                if gt_mask_np.ndim > 2:
                    gt_mask_np = np.squeeze(gt_mask_np)
                if gt_mask_np.shape != fixed_mask_shape:
                    gt_mask_np = np.array(
                        Image.fromarray(gt_mask_np).resize(fixed_image_size, resample=Image.Resampling.NEAREST)
                    )

                image_masks.append(torch.from_numpy(gt_mask_np).to(device=gen_depths.device, dtype=torch.float32))
            processed_gt_obj_masks.append(image_masks)

        if debug_dir is not None:
            print(f"Saving debug visualization to: {debug_dir}")
            debug_mask_overlay_dir = Path(debug_dir) / "mask_overlay"
            if debug_mask_overlay_dir.exists():
                shutil.rmtree(debug_mask_overlay_dir)
            debug_mask_overlay_dir.mkdir(parents=True, exist_ok=True)
            save_mask_overlay_debug(
                generated_images=generated_images,
                generated_obj_masks=generated_obj_masks,
                generated_obj_bboxes=detected_abs_boxes,
                output_dir=debug_mask_overlay_dir,
            )

        if artifact_root_dir is not None:
            assert artifact_masks_dir is not None
            assert artifact_metadata_path is not None
            print(f"Saving localization artifacts to: {artifact_root_dir}")
            for img_idx, image_masks in enumerate(generated_obj_masks):
                image_name = (
                    artifact_image_names[img_idx]
                    if artifact_image_names is not None and img_idx < len(artifact_image_names)
                    else f"img_{img_idx:04d}"
                )
                for obj_idx, mask in enumerate(image_masks):
                    if torch.is_tensor(mask):
                        mask_tensor = mask.detach().cpu()
                    else:
                        mask_tensor = torch.as_tensor(mask)

                    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
                        mask_tensor = mask_tensor.squeeze(0)
                    elif mask_tensor.ndim > 2:
                        mask_tensor = mask_tensor.squeeze()

                    mask_uint8 = (mask_tensor > 0.5).to(torch.uint8).numpy() * 255
                    Image.fromarray(mask_uint8).convert("1").save(
                        artifact_masks_dir / f"{image_name}_obj{obj_idx:02d}_mask.png"
                    )

            intermediate_records = []
            for img_idx in range(len(generated_images)):
                image_name = (
                    artifact_image_names[img_idx]
                    if artifact_image_names is not None and img_idx < len(artifact_image_names)
                    else f"img_{img_idx:04d}"
                )
                object_records = []
                for obj_idx in range(len(src_obj_bboxes[img_idx])):
                    mask_filename = f"{image_name}_obj{obj_idx:02d}_mask.png"
                    object_records.append(
                        {
                            "obj_index": obj_idx,
                            "obj_name": obj_names[img_idx][obj_idx],
                            "grounding_text": vlm_grounding_texts[img_idx][obj_idx],
                            "vlm_bbox_norm": vlm_detected_norm_boxes[img_idx][obj_idx],
                            "final_bbox_abs": detected_abs_boxes[img_idx][obj_idx],
                            "source_bbox_abs": src_obj_bboxes[img_idx][obj_idx],
                            "target_bbox_abs": target_obj_bboxes[img_idx][obj_idx],
                            "may_lost": bool(full_results[img_idx]["objs"][obj_idx].get("may_lost", False)),
                            "vlm_failed": bool(vlm_request_failed_flags[img_idx][obj_idx]),
                            "mask_filename": mask_filename,
                        }
                    )

                intermediate_records.append(
                    {
                        "image_index": img_idx,
                        "image_name": image_name,
                        "objects": object_records,
                    }
                )

            with open(artifact_metadata_path, "w", encoding="utf-8") as f:
                json.dump(intermediate_records, f, ensure_ascii=False, indent=2)
            print(f"Saved localization metadata to: {artifact_metadata_path}")
            print(f"Saved object masks to: {artifact_masks_dir}")

        # multi-view metrics
        print("Calculating 2D/depth/3D metrics...")
        for i, (
            image_coord_list,
            reference_image,
            gt_image,
            generated_image_depth,
            generated_image_obj_masks,
            reference_depth,
            gt_depth,
        ) in enumerate(
            tqdm(
                zip(
                    coordinate_lists,
                    src_images,
                    gt_images,
                    gen_depths,
                    generated_obj_masks,
                    src_depths,
                    gt_depths,
                )
            )
        ):
            min_depth = torch.min(torch.cat([reference_depth.flatten(), gt_depth.flatten()]))
            max_depth = torch.max(torch.cat([reference_depth.flatten(), gt_depth.flatten()]))
            # Per-sample normalization baseline from GT scene valid points.
            gt_scene_points = masked_depth_to_points_3d(
                depth_map=gt_depth, mask=gt_valid_masks[i] > 0.5, intrinsic=gt_intrinsics[i], extrinsic=gt_extrinsics[i]
            )
            if gt_scene_points.shape[0] > 0:
                # Center the cloud to remove camera translation.
                pc_mean = gt_scene_points.mean(dim=0, keepdim=True)
                centered_pc = gt_scene_points - pc_mean
                # Use the enclosing radius as a per-sample scale unit.
                scene_norm_base = torch.norm(centered_pc, dim=-1).max().clamp(min=1e-6)
            else:
                scene_norm_base = torch.tensor(1.0, device=generated_image_depth.device)

            full_results[i]["scene_norm_base"] = float(scene_norm_base.item())
            for obj_idx, (coord_pair, obj_mask) in enumerate(zip(image_coord_list, generated_image_obj_masks)):
                target_bbox = (
                    target_obj_bboxes[i][obj_idx] if target_obj_bboxes is not None else src_obj_bboxes[i][obj_idx]
                )
                gt_obj_mask = processed_gt_obj_masks[i][obj_idx]
                gen_valid_mask = gen_valid_masks[i] > 0.5
                gt_valid_mask = gt_valid_masks[i] > 0.5
                joint_valid_mask = gen_valid_mask & gt_valid_mask

                bbox_diou = cal_bbox_diou(
                    pred_bbox=detected_abs_boxes[i][obj_idx],
                    gt_bbox=target_bbox,
                )

                mask_iou = cal_mask_iou(
                    pred_mask=obj_mask,
                    gt_mask=gt_obj_mask,
                )

                depth_eval_mask = (torch.as_tensor(gt_obj_mask, device=generated_image_depth.device) > 0.5) & (
                    joint_valid_mask > 0
                )
                depth_abs_rel, depth_delta_1_25 = cal_depth_absrel_and_delta(
                    pred_depth=generated_image_depth,
                    gt_depth=gt_depth,
                    eval_mask=depth_eval_mask,
                    eps=1e-6,
                    delta_thresh=1.25,
                )

                normed_dist_depth = (depth_abs_rel / (max_depth - min_depth + 1e-8)).item()

                pts_pred = masked_depth_to_points_3d(
                    depth_map=generated_image_depth,
                    mask=(torch.as_tensor(obj_mask, device=generated_image_depth.device) > 0.5) & joint_valid_mask,
                    intrinsic=gen_intrinsics[i],
                    extrinsic=gen_extrinsics[i],
                )
                pts_orig = masked_depth_to_points_3d(
                    depth_map=reference_depth,
                    mask=(torch.as_tensor(src_obj_masks[i][obj_idx], device=reference_depth.device) > 0.5)
                    & (gt_valid_mask > 0),
                    intrinsic=src_intrinsics[i],
                    extrinsic=src_extrinsics[i],
                )
                pts_gt = masked_depth_to_points_3d(
                    depth_map=gt_depth,
                    mask=(torch.as_tensor(gt_obj_mask, device=gt_depth.device) > 0.5) & joint_valid_mask,
                    intrinsic=gt_intrinsics[i],
                    extrinsic=gt_extrinsics[i],
                )

                pts_pred_norm = pts_pred / scene_norm_base
                pts_gt_norm = pts_gt / scene_norm_base
                chamfer_l2, centroid_dist = cal_chamfer_and_centroid_distance(pts_pred_norm, pts_gt_norm)

                # Additional 3D metrics with object-relative thresholds to improve discriminability.
                if pts_gt.shape[0] > 0:
                    gt_center = pts_gt.mean(dim=0, keepdim=True)
                    obj_scale = torch.norm(pts_gt - gt_center, dim=-1).max().clamp(min=1e-6).item()
                else:
                    obj_scale = 1.0

                pointcloud_metrics = cal_pointcloud_3d_metrics(
                    points_a=pts_pred,
                    points_b=pts_gt,
                    obj_scale=obj_scale,
                    threshold_ratios=(0.01, 0.02, 0.05),
                    max_points_for_nn=2048,
                )

                motion_proj_raw, motion_proj_penalty = cal_motion_projection_penalty(
                    points_orig=pts_orig,
                    points_pred=pts_pred,
                    points_gt=pts_gt,
                )

                full_results[i]["objs"][obj_idx].update(
                    {
                        "bbox_diou": bbox_diou,
                        "mask_iou": mask_iou,
                        "depth_abs_rel": depth_abs_rel,
                        "depth_delta_1_25": depth_delta_1_25,
                        "chamfer_l2": chamfer_l2,
                        "centroid_dist": centroid_dist,
                        "depth_dist": normed_dist_depth,
                        "obj_scale": obj_scale,
                        "nn_l2_mean": pointcloud_metrics["nn_l2_mean"],
                        "nn_l2_p95": pointcloud_metrics["nn_l2_p95"],
                        "hausdorff_l2": pointcloud_metrics["hausdorff_l2"],
                        "precision_1pct": pointcloud_metrics["precision_1pct"],
                        "recall_1pct": pointcloud_metrics["recall_1pct"],
                        "fscore_1pct": pointcloud_metrics["fscore_1pct"],
                        "precision_2pct": pointcloud_metrics["precision_2pct"],
                        "recall_2pct": pointcloud_metrics["recall_2pct"],
                        "fscore_2pct": pointcloud_metrics["fscore_2pct"],
                        "precision_5pct": pointcloud_metrics["precision_5pct"],
                        "recall_5pct": pointcloud_metrics["recall_5pct"],
                        "fscore_5pct": pointcloud_metrics["fscore_5pct"],
                        "motion_proj_ratio_raw": motion_proj_raw,
                        "motion_proj_penalty": motion_proj_penalty,
                    }
                )

        # dino obj similarity
        print("Extracting DINO features for object similarity...")
        original_obj_features = self.dino_model.extract_masked_features(
            images=src_images,
            masks=src_obj_masks,
            bboxes=src_obj_bboxes,
            batch_size=MAX_DINO_BATCH_SIZE,
        )

        generated_obj_features = self.dino_model.extract_masked_features(
            images=generated_images,
            masks=generated_obj_masks,
            bboxes=detected_abs_boxes,
            batch_size=MAX_DINO_BATCH_SIZE,
        )

        for i in range(len(generated_images)):
            for obj_idx, (orig_feat, gen_feat) in enumerate(zip(original_obj_features[i], generated_obj_features[i])):
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_feat.unsqueeze(0), gen_feat.unsqueeze(0), dim=1
                ).item()
                motion_penalty = float(full_results[i]["objs"][obj_idx].get("motion_proj_penalty", 0.0))
                full_results[i]["objs"][obj_idx]["obj_sim"] = cos_sim
                full_results[i]["objs"][obj_idx]["obj_sim_penalized"] = cos_sim * motion_penalty

        # # vlm quality scores
        # print("Scoring editing quality with VLM...")
        # vlm_outputs = self.vlm_model.score_editing_batch(
        #     original_images=src_images,
        #     edited_images=generated_images,
        #     edit_prompts=simple_edit_prompts,
        #     orig_bboxes=src_obj_bboxes,
        #     target_bboxes=detected_abs_boxes,
        # )

        # for i in range(len(generated_images)):
        #     full_results[i]["vlm_score"] = vlm_outputs[i]

        return full_results


if __name__ == "__main__":
    import argparse
    import glob
    import re
    from collections import Counter

    from dotenv import load_dotenv
    from PIL import Image
    from tqdm import tqdm

    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Evaluate generated benchmark images with VLM/depth metrics.")
    parser.add_argument("--backbone", default="depth_anything_v2", help="Backbone subdirectory under --ablation-root.")
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=None,
        help="Optional root containing per-backbone samples/process_artifacts/debug directories.",
    )
    parser.add_argument(
        "--benchmark-metadata",
        type=Path,
        required=True,
        help="Benchmark metadata JSON used by the existing metric logic.",
    )
    parser.add_argument("--output-image-dir", type=Path, default=None, help="Directory containing generated *.png files.")
    parser.add_argument("--output-result-json", type=Path, default=None, help="Path to write grouped metric JSON.")
    parser.add_argument("--process-artifact-dir", type=Path, default=None, help="Directory for localization artifacts.")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Directory for debug visualizations.")
    parser.add_argument("--max-samples-per-bench-item", type=int, default=8)
    parser.add_argument("--device", default="cuda", help="Torch device for SAM/depth/DINO models.")
    args = parser.parse_args()

    backbone_root = args.ablation_root / args.backbone if args.ablation_root is not None else None
    if backbone_root is None:
        missing_output_args = [
            name
            for name, value in (
                ("--output-image-dir", args.output_image_dir),
                ("--output-result-json", args.output_result_json),
                ("--process-artifact-dir", args.process_artifact_dir),
                ("--debug-dir", args.debug_dir),
            )
            if value is None
        ]
        if missing_output_args:
            parser.error(
                "Without --ablation-root, explicit output paths are required: "
                + ", ".join(missing_output_args)
            )

    BENCHMARK_METADATA_PATH = args.benchmark_metadata
    OUTPUT_IMAGE_DIR = args.output_image_dir or (backbone_root / "samples")
    OUTPUT_RESULT_JSON_PATH = args.output_result_json or (backbone_root / "bench_score_new.json")
    PROCESS_ARTIFACT_DIR = args.process_artifact_dir or (backbone_root / "process_artifacts")
    DEBUG_DIR = args.debug_dir or (backbone_root / "debug")
    MAX_SAMPLES_PER_BENCH_ITEM = args.max_samples_per_bench_item

    print("\n===== Metric Config =====")
    print(f"Backbone: {args.backbone}")
    print(f"Benchmark metadata: {BENCHMARK_METADATA_PATH}")
    print(f"Output image dir: {OUTPUT_IMAGE_DIR}")
    print(f"Output result json: {OUTPUT_RESULT_JSON_PATH}")
    print(f"Process artifact dir: {PROCESS_ARTIFACT_DIR}")
    print(f"Debug dir: {DEBUG_DIR}")
    print(f"Device: {args.device}")

    benchmark_metadata = load_benchmark_metadata(BENCHMARK_METADATA_PATH)
    # benchmark_metadata = benchmark_metadata[-5:]  # for quick testing, only process the last 5 samples
    # in form like 0101_seed48.png, but there may be some extra images in the folder, so we need to filter by regex
    pattern = re.compile(r"^(?P<bench>\d{4})_seed(?P<sample>\d+)\.png$")
    generated_image_paths = glob.glob(str(Path(OUTPUT_IMAGE_DIR) / "*.png"))
    generated_image_paths = [p for p in generated_image_paths if pattern.match(Path(p).name)]

    benchmark_by_index = {}
    for idx, item in enumerate(benchmark_metadata):
        bench_index = item["bench_index"]
        benchmark_by_index[int(bench_index)] = item

    generated_image_paths.sort()
    aligned_generated_paths = []
    aligned_sample_indices = []
    generated_images = []
    per_bench_loaded_sample_counts = Counter()
    for path in tqdm(generated_image_paths, desc="Loading generated images"):
        filename = Path(path).name
        matched = pattern.match(filename)
        if not matched:
            continue

        bench_index = int(matched.group("bench"))
        benchmark_item = benchmark_by_index.get(bench_index)
        if benchmark_item is None:
            print(f"Warning: cannot find benchmark item for generated image {filename}, skip.")
            continue

        if (
            MAX_SAMPLES_PER_BENCH_ITEM is not None
            and MAX_SAMPLES_PER_BENCH_ITEM > 0
            and per_bench_loaded_sample_counts[bench_index] >= MAX_SAMPLES_PER_BENCH_ITEM
        ):
            continue

        aligned_generated_paths.append(path)
        aligned_sample_indices.append(bench_index)
        generated_images.append(Image.open(path).convert("RGB"))
        per_bench_loaded_sample_counts[bench_index] += 1

    if not aligned_generated_paths:
        raise ValueError("No generated images can be aligned with benchmark metadata.")

    sample_image_counts = Counter(aligned_sample_indices)
    unique_sample_num = len(sample_image_counts)
    avg_images_per_sample = len(aligned_sample_indices) / max(unique_sample_num, 1)
    min_images_per_sample = min(sample_image_counts.values())
    max_images_per_sample = max(sample_image_counts.values())
    missing_benchmark_samples = len(set(benchmark_by_index.keys()) - set(sample_image_counts.keys()))

    print("\n===== Alignment Summary =====")
    print(f"Benchmark samples: {len(benchmark_by_index)}")
    print(f"Generated images (matched by pattern): {len(generated_image_paths)}")
    print(f"Aligned generated images: {len(aligned_generated_paths)}")
    print(f"Max samples per bench item: {MAX_SAMPLES_PER_BENCH_ITEM}")
    print(f"Unique aligned samples: {unique_sample_num}")
    print(f"Average images per sample: {avg_images_per_sample:.2f}")
    print(f"Min/Max images per sample: {min_images_per_sample}/{max_images_per_sample}")
    print(f"Benchmark samples without generated images: {missing_benchmark_samples}")

    reference_images = []
    coords_lists = []
    obj_bboxes = []
    obj_names = []
    obj_masks = []
    gt_obj_masks = []
    gt_images = []
    reference_depths = []
    gt_depths = []
    target_bboxes = []
    sample_cache = {}
    for sample_index in tqdm(aligned_sample_indices, desc="Loading benchmark data"):
        if sample_index not in sample_cache:
            item = benchmark_by_index[sample_index]

            coords_list = []
            for f1_coord, f2_coord in zip(item["f1_coords"], item["f2_coords"]):
                coords_list.append([f1_coord, f2_coord])

            single_image_mask_list = []
            for mask_path in item["f1_mask_path"]:
                single_image_mask_list.append(np.array(Image.open(mask_path).convert("L")))

            single_gt_mask_list = []
            for mask_path in item["f2_mask_path"]:
                single_gt_mask_list.append(np.array(Image.open(mask_path).convert("L")))

            sample_cache[sample_index] = {
                "coords_list": coords_list,
                "reference_image": Image.open(item["f1_path"]).convert("RGB"),
                "gt_image": Image.open(item["f2_path"]).convert("RGB"),
                "obj_masks": single_image_mask_list,
                "gt_obj_masks": single_gt_mask_list,
                "obj_bboxes": item["f1_obj_bbox"],
                "target_bboxes": item["f2_obj_bbox"],
                "obj_names": item["object_name"],
                "reference_depth": torch.from_numpy(np.load(item["f1_depth_path"])),
                "gt_depth": torch.from_numpy(np.load(item["f2_depth_path"])),
            }

        sample_data = sample_cache[sample_index]
        coords_lists.append(sample_data["coords_list"])
        reference_images.append(sample_data["reference_image"])
        gt_images.append(sample_data["gt_image"])
        obj_masks.append(sample_data["obj_masks"])
        gt_obj_masks.append(sample_data["gt_obj_masks"])
        obj_bboxes.append(sample_data["obj_bboxes"])
        target_bboxes.append(sample_data["target_bboxes"])
        obj_names.append(sample_data["obj_names"])
        reference_depths.append(sample_data["reference_depth"])
        gt_depths.append(sample_data["gt_depth"])

    avoided_reloads = len(aligned_sample_indices) - len(sample_cache)
    print("\n===== Cache Summary =====")
    print(f"Unique samples loaded from benchmark: {len(sample_cache)}")
    print(f"Benchmark reloads avoided by cache: {avoided_reloads}")

    metric_suite = MetricSuite()

    metric_suite.init_vlm_model()
    metric_suite.init_models(device=args.device)
    full_results = metric_suite.cal_multi_metrics(
        generated_images=generated_images,
        src_images=reference_images,
        coordinate_lists=coords_lists,
        src_obj_bboxes=obj_bboxes,
        src_obj_masks=obj_masks,
        obj_names=obj_names,
        gt_images=gt_images,
        gt_obj_masks=gt_obj_masks,
        recorded_src_depths=reference_depths,
        recorded_gt_depths=gt_depths,
        target_obj_bboxes=target_bboxes,
        process_artifact_dir=PROCESS_ARTIFACT_DIR,
        artifact_image_names=[Path(p).stem for p in aligned_generated_paths],
        debug_dir=DEBUG_DIR,
    )

    grouped_payload_map = {}
    for image_path, image_result in zip(aligned_generated_paths, full_results):
        filename = Path(image_path).name
        matched = pattern.match(filename)
        if not matched:
            continue

        bench_index = int(matched.group("bench"))
        sample_index = int(matched.group("sample"))

        if bench_index not in grouped_payload_map:
            grouped_payload_map[bench_index] = {
                "bench_index": bench_index,
                "samples": [],
            }

        sample_entry = {
            "sample_index": sample_index,
            "filename": filename,
            **image_result,
        }
        grouped_payload_map[bench_index]["samples"].append(sample_entry)

    grouped_payload = [grouped_payload_map[k] for k in sorted(grouped_payload_map.keys())]

    for item in grouped_payload:
        item["samples"].sort(key=lambda sample: (sample["sample_index"], sample["filename"]))

    OUTPUT_RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULT_JSON_PATH, "w") as f:
        json.dump(grouped_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark results to: {OUTPUT_RESULT_JSON_PATH}")
