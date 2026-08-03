#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2].resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench.evaluation.config import (  # noqa: E402
    EvaluationConfig,
    EvaluationPaths,
    parse_stages,
)
from bench.evaluation.pipeline import run_evaluation  # noqa: E402
from bench.evaluation.process import StageExecutionError, resolve_executable  # noqa: E402


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete PhyEdit benchmark evaluation through one entry point while allowing DeQA to use a "
            "separate Python environment."
        )
    )
    parser.add_argument("--benchmark-metadata", type=Path, required=True)
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing ####_seedN.png files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metric JSON and artifacts. Defaults to --image-dir.",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages: core,motion,deqa,phys-vlm. Default: all.",
    )
    parser.add_argument("--main-python", default=sys.executable)
    parser.add_argument(
        "--deqa-python",
        default="",
        help="Python executable for the legacy DeQA environment. Defaults to --main-python.",
    )
    parser.add_argument(
        "--deqa-model-path",
        default=os.getenv("DEQA_MODEL_PATH", ""),
        help="Local DeQA model path or Hugging Face model id.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples-per-bench-item", type=int, default=8)
    parser.add_argument("--motion-alpha", type=float, default=1.0)
    parser.add_argument("--motion-beta", type=float, default=0.8)
    parser.add_argument("--motion-static-motion-ratio", type=float, default=0.02)
    parser.add_argument("--deqa-batch-size", type=int, default=64)
    parser.add_argument("--vlm-grounding-workers", type=int, default=1)
    parser.add_argument("--depth-model-path", default="")
    parser.add_argument("--sam-model-path", default="")
    parser.add_argument("--dino-model-path", default="")
    parser.add_argument(
        "--vlm-base-url",
        default="",
        help="Optional OpenAI-compatible VLM endpoint. API key stays in VLLM_API_KEY.",
    )
    parser.add_argument("--vlm-model-name", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed core/motion outputs and resume DeQA/Phys-VLM outputs when possible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write a planned manifest only.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Evaluation manifest path. Defaults to <output-dir>/evaluation_manifest.json.",
    )
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    try:
        stages = parse_stages(args.stages)
        main_python = resolve_executable(args.main_python)
        deqa_python = resolve_executable(args.deqa_python or main_python)
        paths = EvaluationPaths.create(
            project_root=PROJECT_ROOT,
            benchmark_metadata=args.benchmark_metadata,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
        )
        config = EvaluationConfig(
            stages=stages,
            main_python=main_python,
            deqa_python=deqa_python,
            device=args.device,
            max_samples_per_bench_item=args.max_samples_per_bench_item,
            motion_alpha=args.motion_alpha,
            motion_beta=args.motion_beta,
            motion_static_motion_ratio=args.motion_static_motion_ratio,
            deqa_batch_size=args.deqa_batch_size,
            vlm_grounding_workers=args.vlm_grounding_workers,
            depth_model_path=args.depth_model_path,
            sam_model_path=args.sam_model_path,
            dino_model_path=args.dino_model_path,
            deqa_model_path=args.deqa_model_path,
            vlm_base_url=args.vlm_base_url,
            vlm_model_name=args.vlm_model_name,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        config.validate()
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    try:
        run_evaluation(paths, config)
    except StageExecutionError as error:
        raise SystemExit(1) from error
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if not paths.manifest_path.exists():
            parser.error(str(error))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
