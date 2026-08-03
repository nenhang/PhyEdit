from __future__ import annotations

from pathlib import Path

from .config import EvaluationConfig, EvaluationPaths


def core_command(paths: EvaluationPaths, config: EvaluationConfig) -> list[str]:
    return [
        config.main_python,
        str(paths.bench_dir / "vllm_depth_metrics.py"),
        "--benchmark-metadata",
        str(paths.benchmark_metadata),
        "--output-image-dir",
        str(paths.image_dir),
        "--output-result-json",
        str(paths.core_result),
        "--process-artifact-dir",
        str(paths.artifact_dir),
        "--debug-dir",
        str(paths.debug_dir),
        "--max-samples-per-bench-item",
        str(config.max_samples_per_bench_item),
        "--device",
        config.device,
    ]


def motion_command(
    paths: EvaluationPaths,
    config: EvaluationConfig,
    input_result: Path,
) -> list[str]:
    return [
        config.main_python,
        str(paths.bench_dir / "backfill_motion_penalty.py"),
        "--benchmark-metadata-path",
        str(paths.benchmark_metadata),
        "--existing-result-json",
        str(input_result),
        "--output-result-json",
        str(paths.motion_result),
        "--output-image-dir",
        str(paths.image_dir),
        "--artifact-root-dir",
        str(paths.artifact_dir),
        "--device",
        config.device,
        "--motion-alpha",
        str(config.motion_alpha),
        "--motion-beta",
        str(config.motion_beta),
        "--motion-static-motion-ratio",
        str(config.motion_static_motion_ratio),
    ]


def deqa_command(
    paths: EvaluationPaths,
    config: EvaluationConfig,
    input_result: Path,
) -> list[str]:
    command = [
        config.deqa_python,
        str(paths.evaluation_dir / "score_deqa.py"),
        "--input-json",
        str(input_result),
        "--output-json",
        str(paths.deqa_result),
        "--image-dir",
        str(paths.image_dir),
        "--model-path",
        config.deqa_model_path,
        "--device",
        config.device,
        "--batch-size",
        str(config.deqa_batch_size),
    ]
    if config.resume:
        command.append("--resume")
    return command


def phys_vlm_command(
    paths: EvaluationPaths,
    config: EvaluationConfig,
    input_result: Path,
) -> list[str]:
    command = [
        config.main_python,
        str(paths.bench_dir / "backfill_vlm_logic_consistency.py"),
        "--existing-result-json",
        str(input_result),
        "--output-result-json",
        str(paths.final_result),
        "--output-image-dir",
        str(paths.image_dir),
        "--artifact-root-dir",
        str(paths.artifact_dir),
        "--benchmark-metadata-path",
        str(paths.benchmark_metadata),
    ]
    if config.resume and input_result == paths.final_result:
        command.append("--repair-failed-only")
    return command
