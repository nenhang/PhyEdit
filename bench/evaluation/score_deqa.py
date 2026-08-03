#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parents[2].resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench.evaluation.io import atomic_write_json, load_json  # noqa: E402


def _load_grouped_payload(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"Expected a grouped benchmark JSON array: {path}")
    return payload


def _to_python_scalar(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.floating):
        return float(value)
    if np is not None and isinstance(value, np.integer):
        return int(value)
    return value


def _build_model(model_path: str, device: str, attention_implementation: str):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": device,
    }
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    if not hasattr(model, "score"):
        raise AttributeError(f"Loaded DeQA model does not expose score(): {model_path}")
    return model


def _collect_pending_samples(
    payload: list[dict[str, Any]],
    image_dir: Path,
    *,
    resume: bool,
    allow_missing_images: bool,
) -> list[tuple[dict[str, Any], Path]]:
    pending: list[tuple[dict[str, Any], Path]] = []
    missing: list[Path] = []

    for group in payload:
        samples = group.get("samples", [])
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            if resume and sample.get("deqa_score") is not None:
                continue
            filename = str(sample.get("filename", "")).strip()
            if not filename:
                continue
            image_path = image_dir / filename
            if not image_path.is_file():
                missing.append(image_path)
                continue
            pending.append((sample, image_path))

    if missing and not allow_missing_images:
        preview = "\n".join(str(path) for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} generated images:\n{preview}{suffix}"
        )
    if missing:
        print(f"Warning: skipping {len(missing)} missing generated images.")
    return pending


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add DeQA scores to a grouped benchmark JSON using an isolated legacy environment."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default=os.getenv("DEQA_MODEL_PATH", "zhiyuanyou/DeQA-Score-Mix3"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--attention-implementation", default="flash_attention_2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 means all samples.",
    )
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    input_json = args.input_json.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()

    if not input_json.is_file():
        parser.error(f"Input JSON not found: {input_json}")
    if not image_dir.is_dir():
        parser.error(f"Generated image directory not found: {image_dir}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples cannot be negative")

    source_json = output_json if args.resume and output_json.is_file() else input_json
    payload = _load_grouped_payload(source_json)
    pending = _collect_pending_samples(
        payload,
        image_dir,
        resume=args.resume,
        allow_missing_images=args.allow_missing_images,
    )
    if args.max_samples:
        pending = pending[: args.max_samples]

    if not pending:
        if output_json != source_json:
            atomic_write_json(output_json, payload)
        print(f"No DeQA samples require scoring. Output: {output_json}")
        return

    from PIL import Image
    from tqdm import tqdm

    model = _build_model(args.model_path, args.device, args.attention_implementation)
    progress = tqdm(total=len(pending), desc="Scoring DeQA")

    for start in range(0, len(pending), args.batch_size):
        batch_entries = pending[start : start + args.batch_size]
        batch_images = []
        for _, image_path in batch_entries:
            with Image.open(image_path) as image:
                batch_images.append(image.convert("RGB"))

        batch_scores = model.score(batch_images)
        if len(batch_scores) != len(batch_entries):
            raise ValueError(
                f"Unexpected DeQA output length: got {len(batch_scores)}, expected {len(batch_entries)}"
            )

        for (sample, _), score in zip(batch_entries, batch_scores):
            sample["deqa_score"] = _to_python_scalar(score)

        atomic_write_json(output_json, payload)
        progress.update(len(batch_entries))

    progress.close()
    print(f"Saved DeQA scores to: {output_json}")


if __name__ == "__main__":
    main()
