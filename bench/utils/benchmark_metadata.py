import json
import os
from pathlib import Path
from typing import Any

SCALAR_PATH_FIELDS = (
    "f1_path",
    "f2_path",
    "f1_depth_path",
    "f2_depth_path",
    "f1_intrinsic_path",
    "f2_intrinsic_path",
    "f1_extrinsic_path",
    "f2_extrinsic_path",
)
LIST_PATH_FIELDS = ("f1_mask_path", "f2_mask_path")


def _resolve_metadata_path(value: Any, metadata_dir: Path) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return str(path)
    base_dir = metadata_dir.parent if metadata_dir.name == "metadata" and path.parts[0] == "assets" else metadata_dir
    return str((base_dir / path).resolve())


def load_benchmark_metadata(path: str | Path) -> list[dict[str, Any]]:
    metadata_path = Path(path).expanduser().resolve()
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(
            f"Benchmark metadata must be a JSON array of objects: {metadata_path}"
        )

    base_dir = metadata_path.parent
    for item in payload:
        for field in SCALAR_PATH_FIELDS:
            if field in item:
                item[field] = _resolve_metadata_path(item[field], base_dir)
        for field in LIST_PATH_FIELDS:
            values = item.get(field)
            if isinstance(values, list):
                item[field] = [
                    _resolve_metadata_path(value, base_dir) for value in values
                ]

    return payload
