"""Stable physical-plausibility VLM metric engine used by bench.evaluation."""

import argparse
import concurrent.futures
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.utils.benchmark_metadata import load_benchmark_metadata  # noqa: E402
from bench.vlm.vlm import VLMModel, normalize_image_for_eval  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

FILENAME_PATTERNS = {
    "res": re.compile(r"^(?P<bench>\d{4})_res(?P<sample>\d+)\.png$"),
    "seed": re.compile(r"^(?P<bench>\d{4})_seed(?P<sample>\d+)\.png$"),
}


def _match_filename_mode(filename: str):
    for mode, pattern in FILENAME_PATTERNS.items():
        if pattern.match(filename):
            return mode
    return None


def _clamp_bbox(bbox: list[float], width: int, height: int) -> list[int] | None:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = int(round(min(max(x1, 0.0), width - 1)))
    y1 = int(round(min(max(y1, 0.0), height - 1)))
    x2 = int(round(min(max(x2, 0.0), width - 1)))
    y2 = int(round(min(max(y2, 0.0), height - 1)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _load_artifact_record_map(artifact_root_dir: Path) -> dict[str, dict]:
    metadata_path = artifact_root_dir / "localization_results.json"
    if not metadata_path.exists():
        return {}
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


def _load_benchmark_metadata_map(benchmark_metadata_path: Path | None) -> dict[int, dict]:
    if benchmark_metadata_path is None or not benchmark_metadata_path.exists():
        return {}
    payload = load_benchmark_metadata(benchmark_metadata_path)
    metadata_map: dict[int, dict] = {}
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            bench_index_value = item.get("bench_index")
            if bench_index_value is None:
                continue
            try:
                bench_index = int(bench_index_value)
            except Exception:
                continue
            metadata_map[bench_index] = item
    return metadata_map


def _build_sample_cache_item(benchmark_item: dict) -> dict:
    src_w = src_h = None
    try:
        src_image = Image.open(benchmark_item["f1_path"]).convert("RGB")
        src_w, src_h = src_image.size
    except Exception:
        pass

    def _as_box_list(raw_boxes) -> list[list[float]]:
        if not isinstance(raw_boxes, list):
            return []
        output = []
        for item in raw_boxes:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                output.append([float(v) for v in item])
        return output

    return {
        "src_size": (src_w, src_h),
        "target_bboxes": _as_box_list(benchmark_item.get("f2_obj_bbox")),
        "source_bboxes": _as_box_list(benchmark_item.get("f1_obj_bbox")),
    }


def _collect_boxes_for_sample(
    sample_entry: dict,
    image_name: str,
    artifact_record_map: dict[str, dict],
    fallback_boxes: list[list[float]] | None = None,
) -> list[list[float]]:
    boxes: list[list[float]] = []
    objs = sample_entry.get("objs", [])
    if isinstance(objs, list):
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            for key in ("final_bbox_abs", "generated_obj_bbox", "pred_bbox_abs", "target_bbox_abs", "target_bbox"):
                candidate = obj.get(key)
                if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                    boxes.append([float(v) for v in candidate])
                    break

    artifact_record = artifact_record_map.get(image_name, {})
    obj_records = artifact_record.get("objects", []) if isinstance(artifact_record, dict) else []
    if isinstance(obj_records, list):
        for obj_record in obj_records:
            if not isinstance(obj_record, dict):
                continue
            candidate = obj_record.get("final_bbox_abs")
            if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                boxes.append([float(v) for v in candidate])

    if fallback_boxes:
        boxes.extend(
            [[float(v) for v in box] for box in fallback_boxes if isinstance(box, (list, tuple)) and len(box) == 4]
        )

    # Deduplicate
    unique = []
    seen = set()
    for box in boxes:
        key = tuple(round(float(v), 2) for v in box)
        if key in seen:
            continue
        seen.add(key)
        unique.append(box)
    return unique


def _draw_boxes(image: Image.Image, boxes: list[list[float]]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    for box in boxes:
        clamped = _clamp_bbox(box, w, h)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    return canvas


def _edited_and_annotated_for_sample(
    grouped_payload: list,
    group_idx: int,
    sample_idx: int,
    filename: str,
    generated_path: Path,
    bench_index: int | None,
    sample_cache: dict[int, dict],
    artifact_record_map: dict[str, dict],
) -> tuple[Image.Image, Image.Image]:
    image_name = Path(filename).stem
    sample_entry = grouped_payload[group_idx]["samples"][sample_idx]
    cache_item = sample_cache.get(bench_index or -1, {})
    fallback_boxes = cache_item.get("target_bboxes", []) if isinstance(cache_item, dict) else []

    edited_raw = Image.open(generated_path).convert("RGB")
    original_w, original_h = edited_raw.size
    edited = normalize_image_for_eval(edited_raw)

    boxes = _collect_boxes_for_sample(
        sample_entry,
        image_name,
        artifact_record_map,
        fallback_boxes=fallback_boxes,
    )
    boxes_scaled = []
    for b in boxes:
        nw, nh = edited.size
        sx = nw / max(original_w, 1)
        sy = nh / max(original_h, 1)
        boxes_scaled.append([float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy])
    annotated = _draw_boxes(edited, boxes_scaled)
    return edited, annotated


def _run_backfill_once(
    existing_result_json: Path,
    output_result_json: Path,
    output_image_dir: Path,
    artifact_root_dir: Path | None,
    vlm_model: VLMModel | None,
    benchmark_metadata_map: dict[int, dict] | None = None,
    debug_visualize_dir: Path | None = None,
    debug_max_samples: int = 0,
    run_name: str = "",
    visualize_only: bool = False,
    repair_failed_only: bool = False,
) -> None:
    with open(existing_result_json, "r", encoding="utf-8") as f:
        grouped_payload = json.load(f)

    artifact_record_map = _load_artifact_record_map(artifact_root_dir) if artifact_root_dir is not None else {}

    benchmark_metadata_map = benchmark_metadata_map or {}
    sample_cache: dict[int, dict] = {}
    tasks = []
    for group_idx, group in enumerate(grouped_payload):
        if not isinstance(group, dict):
            continue
        for sample_idx, sample_entry in enumerate(group.get("samples", [])):
            if not isinstance(sample_entry, dict):
                continue
            if repair_failed_only:
                vlm_score = sample_entry.get("vlm_score")
                if isinstance(vlm_score, dict):
                    has_score = vlm_score.get("logic_consistency_score") is not None
                    failed = bool(vlm_score.get("logic_consistency_vlm_failed", False))
                    if has_score and not failed:
                        continue
            filename = str(sample_entry.get("filename", "")).strip()
            if _match_filename_mode(filename) is None:
                continue
            generated_path = output_image_dir / filename
            if not generated_path.exists():
                continue
            bench_index = None
            bench_index_value = group.get("bench_index")
            if bench_index_value is not None:
                try:
                    bench_index = int(bench_index_value)
                except Exception:
                    bench_index = None
            if bench_index is None:
                matched = FILENAME_PATTERNS["res"].match(filename) or FILENAME_PATTERNS["seed"].match(filename)
                if matched is not None:
                    bench_str = matched.group("bench")
                    if bench_str is not None:
                        bench_index = int(bench_str)

            if bench_index is not None and bench_index not in sample_cache:
                bench_item = benchmark_metadata_map.get(bench_index)
                if isinstance(bench_item, dict):
                    sample_cache[bench_index] = _build_sample_cache_item(bench_item)
                else:
                    sample_cache[bench_index] = {}

            tasks.append((group_idx, sample_idx, filename, generated_path, bench_index))

    if not tasks:
        if repair_failed_only:
            print(f"No failed/missing logic consistency samples found in {existing_result_json}")
        else:
            print(f"No valid samples found in {existing_result_json}")
        return
    if repair_failed_only:
        print(f"Repairing {len(tasks)} failed/missing logic consistency samples in {existing_result_json}")

    if visualize_only:
        if debug_visualize_dir is None:
            raise ValueError("visualize_only requires debug_visualize_dir")
        cap = max(int(debug_max_samples), 1)
        debug_visualize_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        desc = f"Visualize-only {run_name or existing_result_json.parent.name}"
        for group_idx, sample_idx, filename, generated_path, bench_index in tqdm(tasks, desc=desc):
            if saved >= cap:
                break
            edited, annotated = _edited_and_annotated_for_sample(
                grouped_payload,
                group_idx,
                sample_idx,
                filename,
                generated_path,
                bench_index,
                sample_cache,
                artifact_record_map,
            )
            stem = Path(filename).stem
            edited.save(debug_visualize_dir / f"{stem}_edited.png")
            annotated.save(debug_visualize_dir / f"{stem}_annotated_bbox.png")
            saved += 1
        print(f"Saved {saved} visualization pairs to {debug_visualize_dir}")
        return

    if vlm_model is None:
        raise ValueError("vlm_model is required when not using visualize_only")

    edited_images = []
    annotated_images = []
    task_refs = []
    debug_saved = 0

    for group_idx, sample_idx, filename, generated_path, bench_index in tqdm(
        tasks, desc=f"Loading {existing_result_json.parent.name}"
    ):
        edited, annotated = _edited_and_annotated_for_sample(
            grouped_payload,
            group_idx,
            sample_idx,
            filename,
            generated_path,
            bench_index,
            sample_cache,
            artifact_record_map,
        )

        if debug_visualize_dir is not None and debug_saved < max(int(debug_max_samples), 0):
            debug_visualize_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(filename).stem
            edited.save(debug_visualize_dir / f"{stem}_edited.png")
            annotated.save(debug_visualize_dir / f"{stem}_annotated_bbox.png")
            debug_saved += 1

        edited_images.append(edited)
        annotated_images.append(annotated)
        task_refs.append((group_idx, sample_idx))

    vlm_outputs = vlm_model.score_logic_consistency_batch(
        edited_images=edited_images,
        annotated_images=annotated_images,
        run_name=run_name,
    )

    for (group_idx, sample_idx), vlm_out in zip(task_refs, vlm_outputs):
        sample_entry = grouped_payload[group_idx]["samples"][sample_idx]
        vlm_score = sample_entry.get("vlm_score")
        if not isinstance(vlm_score, dict):
            vlm_score = {}
            sample_entry["vlm_score"] = vlm_score

        score_value = vlm_out.get("logic_consistency_score")
        if score_value is not None:
            try:
                score_value = int(score_value)
            except Exception:
                score_value = None
        vlm_score["logic_consistency_score"] = score_value
        vlm_score["logic_consistency_explanation"] = str(vlm_out.get("explanation", "")).strip()
        vlm_score["logic_consistency_vlm_failed"] = bool(vlm_out.get("vlm_failed", False))
        if vlm_out.get("vlm_error"):
            vlm_score["logic_consistency_vlm_error"] = str(vlm_out.get("vlm_error"))

    output_result_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_result_json, "w", encoding="utf-8") as f:
        json.dump(grouped_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved backfilled logic consistency JSON to: {output_result_json}")


def _is_baseline_method_dir(method_dir: Path) -> bool:
    return not method_dir.name.startswith(".")


def _process_one_method_dir(
    subdir: Path,
    benchmark_metadata_map: dict[int, dict],
    debug_visualize_dir: Path | None = None,
    debug_max_samples: int = 0,
    visualize_only: bool = False,
    repair_failed_only: bool = False,
) -> tuple[str, bool, str]:
    in_json = subdir / "bench_score_penalty_deqa.json"
    if not in_json.exists():
        in_json = subdir / "bench_score_new.json"
    out_json = subdir / "bench_score_vlm_logic.json"
    images_dir = subdir / "images"
    artifact_dir = subdir / "process_artifacts"
    if not in_json.exists() or not images_dir.exists():
        return subdir.name, False, "missing bench_score_penalty_deqa.json/bench_score_new.json or images/"

    try:
        vlm_model = None if visualize_only else VLMModel()
        _run_backfill_once(
            existing_result_json=in_json,
            output_result_json=out_json,
            output_image_dir=images_dir,
            artifact_root_dir=artifact_dir if artifact_dir.exists() else None,
            benchmark_metadata_map=benchmark_metadata_map,
            vlm_model=vlm_model,
            debug_visualize_dir=debug_visualize_dir,
            debug_max_samples=debug_max_samples,
            run_name=subdir.name,
            visualize_only=visualize_only,
            repair_failed_only=repair_failed_only,
        )
        return subdir.name, True, ""
    except Exception as e:
        return subdir.name, False, str(e)


def _parse_csv_set(text: str) -> set[str]:
    values = {item.strip() for item in str(text).split(",") if item.strip()}
    return values


def main():
    parser = argparse.ArgumentParser(description="Backfill VLM logic consistency score into benchmark JSON.")
    parser.add_argument(
        "--existing-result-json",
        default="",
    )
    parser.add_argument(
        "--output-result-json",
        default="",
    )
    parser.add_argument("--output-image-dir", default="")
    parser.add_argument(
        "--artifact-root-dir",
        default="",
    )
    parser.add_argument("--benchmark-metadata-path", required=True)
    parser.add_argument("--batch-root-dir", default="", help="If set, process subdirs under this root.")
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method folder names to backfill in batch mode, e.g. method_a,method_b.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel method workers in batch mode (CPU/network parallelism, no GPU needed).",
    )
    parser.add_argument(
        "--debug-visualize-dir",
        type=str,
        default="",
        help="Optional directory to save VLM input visualization pairs (edited + annotated bbox).",
    )
    parser.add_argument(
        "--debug-max-samples",
        type=int,
        default=0,
        help="Max number of samples to dump for visualization in each processed method.",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help=(
            "Only export edited + bbox visualization PNGs per method; do not call VLM or write bench_score JSON. "
            "Requires --debug-visualize-dir. If --debug-max-samples is 0, defaults to 10 per method."
        ),
    )
    parser.add_argument(
        "--repair-failed-only",
        action="store_true",
        help="Only rerun samples with missing/failed logic_consistency_score; keep successful existing scores unchanged.",
    )
    args = parser.parse_args()
    benchmark_metadata_map = _load_benchmark_metadata_map(Path(args.benchmark_metadata_path))
    debug_visualize_dir = Path(args.debug_visualize_dir) if str(args.debug_visualize_dir).strip() else None
    debug_max_samples = max(int(args.debug_max_samples), 0)
    if args.visualize_only:
        if debug_visualize_dir is None:
            parser.error("--visualize-only requires --debug-visualize-dir")
        if debug_max_samples == 0:
            debug_max_samples = 10

    batch_root = Path(args.batch_root_dir).expanduser() if str(args.batch_root_dir).strip() else None
    if batch_root is not None:
        if not batch_root.exists():
            raise FileNotFoundError(f"batch root dir not found: {batch_root}")

        selected_methods = _parse_csv_set(args.methods)
        candidates = [p for p in sorted(batch_root.iterdir()) if p.is_dir()]
        if selected_methods:
            candidates = [p for p in candidates if p.name in selected_methods]
        else:
            candidates = [p for p in candidates if _is_baseline_method_dir(p)]

        if not candidates:
            print("No method folders matched for batch mode.")
            return

        worker_count = max(int(args.num_workers), 1)
        print(f"Batch methods: {len(candidates)}, workers: {worker_count}")
        if worker_count == 1:
            for subdir in candidates:
                print(f"Processing {subdir.name} ...")
                method_debug_dir = (debug_visualize_dir / subdir.name) if debug_visualize_dir is not None else None
                name, ok, msg = _process_one_method_dir(
                    subdir,
                    benchmark_metadata_map=benchmark_metadata_map,
                    debug_visualize_dir=method_debug_dir,
                    debug_max_samples=debug_max_samples,
                    visualize_only=args.visualize_only,
                    repair_failed_only=args.repair_failed_only,
                )
                if not ok:
                    print(f"Failed {name}: {msg}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _process_one_method_dir,
                        subdir,
                        benchmark_metadata_map,
                        (debug_visualize_dir / subdir.name) if debug_visualize_dir is not None else None,
                        debug_max_samples,
                        args.visualize_only,
                        args.repair_failed_only,
                    ): subdir.name
                    for subdir in candidates
                }
                for future in concurrent.futures.as_completed(future_map):
                    name = future_map[future]
                    try:
                        _, ok, msg = future.result()
                    except Exception as e:
                        ok, msg = False, str(e)
                    if ok:
                        print(f"Done {name}")
                    else:
                        print(f"Failed {name}: {msg}")
        return

    missing_single_run_args = [
        name
        for name, value in (
            ("--existing-result-json", args.existing_result_json),
            ("--output-result-json", args.output_result_json),
            ("--output-image-dir", args.output_image_dir),
            ("--artifact-root-dir", args.artifact_root_dir),
        )
        if not str(value).strip()
    ]
    if missing_single_run_args:
        parser.error(
            "Single-method mode requires: " + ", ".join(missing_single_run_args)
        )

    vlm_model = None if args.visualize_only else VLMModel()
    _run_backfill_once(
        existing_result_json=Path(args.existing_result_json),
        output_result_json=Path(args.output_result_json),
        output_image_dir=Path(args.output_image_dir),
        artifact_root_dir=(Path(args.artifact_root_dir) if Path(args.artifact_root_dir).exists() else None),
        benchmark_metadata_map=benchmark_metadata_map,
        vlm_model=vlm_model,
        debug_visualize_dir=debug_visualize_dir,
        debug_max_samples=debug_max_samples,
        run_name=Path(args.output_result_json).parent.name,
        visualize_only=args.visualize_only,
        repair_failed_only=args.repair_failed_only,
    )


if __name__ == "__main__":
    main()
