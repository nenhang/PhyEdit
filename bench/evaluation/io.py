from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_NAME_PATTERN = re.compile(r"^(?P<bench>\d{4})_seed(?P<seed>\d+)\.png$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def is_valid_grouped_result(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, list) and all(isinstance(item, dict) for item in payload)


def summarize_inputs(
    benchmark_metadata: Path,
    image_dir: Path,
    max_samples_per_bench_item: int,
) -> dict[str, Any]:
    payload = load_json(benchmark_metadata)
    if not isinstance(payload, list):
        raise ValueError(
            f"Benchmark metadata must be a JSON array: {benchmark_metadata}"
        )

    benchmark_indices = []
    for item in payload:
        if not isinstance(item, dict) or item.get("bench_index") is None:
            raise ValueError("Every benchmark metadata item must contain bench_index")
        benchmark_indices.append(int(item["bench_index"]))

    duplicate_indices = sorted(
        index for index, count in Counter(benchmark_indices).items() if count > 1
    )
    if duplicate_indices:
        raise ValueError(
            f"Duplicate bench_index values in metadata: {duplicate_indices}"
        )

    benchmark_index_set = set(benchmark_indices)
    per_bench_counts: Counter[int] = Counter()
    generated_pattern_count = 0
    extra_generated_count = 0
    for image_path in image_dir.glob("*.png"):
        matched = IMAGE_NAME_PATTERN.match(image_path.name)
        if matched is None:
            continue
        generated_pattern_count += 1
        bench_index = int(matched.group("bench"))
        if bench_index in benchmark_index_set:
            per_bench_counts[bench_index] += 1
        else:
            extra_generated_count += 1

    aligned_image_count = sum(
        min(per_bench_counts[index], max_samples_per_bench_item)
        for index in benchmark_indices
    )
    missing_bench_indices = sorted(
        index for index in benchmark_indices if per_bench_counts[index] == 0
    )
    if aligned_image_count == 0:
        raise ValueError(
            f"No generated images in {image_dir} align with metadata {benchmark_metadata}"
        )

    return {
        "benchmark_items": len(benchmark_indices),
        "generated_images_matching_pattern": generated_pattern_count,
        "aligned_images_after_cap": aligned_image_count,
        "unique_aligned_bench_items": len(per_bench_counts),
        "missing_bench_items": len(missing_bench_indices),
        "missing_bench_indices": missing_bench_indices,
        "extra_generated_images": extra_generated_count,
        "max_samples_per_bench_item": max_samples_per_bench_item,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None
