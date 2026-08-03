"""Portable metadata loaders for RealManip-40K and ManipEval."""

from __future__ import annotations

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
    "moved_image_path",
)
LIST_PATH_FIELDS = ("f1_mask_path", "f2_mask_path")


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
                records.append(record)
    else:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Metadata must be a JSON array or JSONL sequence of objects: {path}")
    return records


def _resolve_asset_path(value: Any, metadata_path: Path) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value

    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return str(path)

    if metadata_path.parent.name == "metadata" and path.parts and path.parts[0] == "assets":
        base_dir = metadata_path.parent.parent
    else:
        base_dir = metadata_path.parent
    return str((base_dir / path).resolve())


def _resolve_record_paths(records: list[dict[str, Any]], metadata_path: Path) -> list[dict[str, Any]]:
    for item in records:
        for field in SCALAR_PATH_FIELDS:
            if field in item:
                item[field] = _resolve_asset_path(item[field], metadata_path)
        for field in LIST_PATH_FIELDS:
            values = item.get(field)
            if isinstance(values, list):
                item[field] = [_resolve_asset_path(value, metadata_path) for value in values]
    return records


def _required_metadata_path(path: str | os.PathLike[str] | None, env_names: tuple[str, ...]) -> Path:
    value = path
    if value is None:
        for env_name in env_names:
            value = os.getenv(env_name)
            if value:
                break
    if not value:
        names = " or ".join(env_names)
        raise ValueError(f"Metadata path is required; pass it explicitly or set {names}")

    resolved = Path(value).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_aggregated_metadata(
    metadata_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    path = _required_metadata_path(
        metadata_path,
        ("REALMANIP_METADATA_PATH", "AGGREGATED_METADATA_PATH"),
    )
    return _resolve_record_paths(_read_records(path), path)


def load_metadata(
    metadata_path: str | os.PathLike[str] | None = None,
    add_benchmark: bool = False,
) -> list[dict[str, Any]]:
    if add_benchmark:
        raise ValueError("Training and benchmark metadata must be loaded separately")
    return load_aggregated_metadata(metadata_path)


def load_benchmark_metadata(
    metadata_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    path = _required_metadata_path(metadata_path, ("MANIPEVAL_METADATA_PATH",))
    return _resolve_record_paths(_read_records(path), path)
