from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import EvaluationConfig, EvaluationPaths
from .io import (
    atomic_write_json,
    git_revision,
    is_valid_grouped_result,
    summarize_inputs,
    utc_now,
)
from .process import StageExecutionError, run_stage
from .stage_commands import (
    core_command,
    deqa_command,
    motion_command,
    phys_vlm_command,
)


def _build_environment(config: EvaluationConfig) -> dict[str, str]:
    environment = dict(os.environ)
    environment["VLM_GROUNDING_MAX_WORKERS"] = str(config.vlm_grounding_workers)
    model_environment = {
        "DA_MODEL_PATH": config.depth_model_path,
        "SAM_MODEL_PATH": config.sam_model_path,
        "DINO_MODEL_PATH": config.dino_model_path,
        "VLLM_BASE_URL": config.vlm_base_url,
        "VLLM_MODEL_NAME": config.vlm_model_name,
    }
    for key, value in model_environment.items():
        if value:
            environment[key] = value
    return environment


def _resolve_existing_input(candidates: list[Path], stage: str, dry_run: bool) -> Path:
    for candidate in candidates:
        if is_valid_grouped_result(candidate):
            return candidate
    if dry_run:
        return candidates[0]
    options = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Stage '{stage}' requires one valid input JSON from: {options}"
    )


def _new_manifest(
    paths: EvaluationPaths,
    config: EvaluationConfig,
    input_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "planned" if config.dry_run else "running",
        "git_revision": git_revision(paths.project_root),
        "project_root": str(paths.project_root),
        "benchmark_metadata": str(paths.benchmark_metadata),
        "image_dir": str(paths.image_dir),
        "output_dir": str(paths.output_dir),
        "device": config.device,
        "main_python": config.main_python,
        "deqa_python": config.deqa_python,
        "stages": list(config.stages),
        "resume": config.resume,
        "input_summary": input_summary,
        "model_overrides": config.model_overrides(),
        "outputs": paths.manifest_outputs(),
        "stage_runs": [],
    }


def _record_reuse(manifest: dict[str, Any], stage: str, output: Path) -> None:
    print(f"[resume] Reusing {stage} result: {output}")
    manifest["stage_runs"].append(
        {
            "stage": stage,
            "status": "reused",
            "output": str(output),
            "finished_at": utc_now(),
        }
    )


def run_evaluation(paths: EvaluationPaths, config: EvaluationConfig) -> dict[str, Any]:
    config.validate()
    if not paths.benchmark_metadata.is_file():
        raise FileNotFoundError(
            f"Benchmark metadata not found: {paths.benchmark_metadata}"
        )
    if not paths.image_dir.is_dir():
        raise FileNotFoundError(
            f"Generated image directory not found: {paths.image_dir}"
        )

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    input_summary = summarize_inputs(
        paths.benchmark_metadata,
        paths.image_dir,
        config.max_samples_per_bench_item,
    )

    print("===== Input Summary =====")
    for key, value in input_summary.items():
        if key != "missing_bench_indices":
            print(f"{key}: {value}")
    if input_summary["missing_bench_indices"]:
        print(f"missing_bench_indices: {input_summary['missing_bench_indices']}")

    environment = _build_environment(config)
    manifest = _new_manifest(paths, config, input_summary)
    atomic_write_json(paths.manifest_path, manifest)

    try:
        current_result: Path | None = None
        for stage in config.stages:
            if stage == "core":
                if config.resume and is_valid_grouped_result(paths.core_result):
                    _record_reuse(manifest, stage, paths.core_result)
                    current_result = paths.core_result
                    atomic_write_json(paths.manifest_path, manifest)
                    continue
                command = core_command(paths, config)
                current_result = paths.core_result

            elif stage == "motion":
                if config.resume and is_valid_grouped_result(paths.motion_result):
                    _record_reuse(manifest, stage, paths.motion_result)
                    current_result = paths.motion_result
                    atomic_write_json(paths.manifest_path, manifest)
                    continue
                motion_input = current_result or _resolve_existing_input(
                    [paths.core_result], stage, config.dry_run
                )
                command = motion_command(paths, config, motion_input)
                current_result = paths.motion_result

            elif stage == "deqa":
                deqa_input = current_result or _resolve_existing_input(
                    [paths.motion_result, paths.core_result], stage, config.dry_run
                )
                command = deqa_command(paths, config, deqa_input)
                current_result = paths.deqa_result

            elif stage == "phys-vlm":
                phys_input = current_result or _resolve_existing_input(
                    [paths.deqa_result, paths.motion_result, paths.core_result],
                    stage,
                    config.dry_run,
                )
                if config.resume and is_valid_grouped_result(paths.final_result):
                    phys_input = paths.final_result
                command = phys_vlm_command(paths, config, phys_input)
                current_result = paths.final_result

            else:
                raise ValueError(f"Unsupported stage: {stage}")

            record = run_stage(
                stage=stage,
                command=command,
                environment=environment,
                project_root=paths.project_root,
                dry_run=config.dry_run,
            )
            manifest["stage_runs"].append(record)
            atomic_write_json(paths.manifest_path, manifest)

    except StageExecutionError as error:
        manifest["stage_runs"].append(error.record)
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["error"] = str(error)
        atomic_write_json(paths.manifest_path, manifest)
        raise
    except (OSError, ValueError, json.JSONDecodeError) as error:
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["error"] = str(error)
        atomic_write_json(paths.manifest_path, manifest)
        raise

    manifest["status"] = "planned" if config.dry_run else "completed"
    manifest["finished_at"] = utc_now()
    atomic_write_json(paths.manifest_path, manifest)

    print("\n===== Evaluation Outputs =====")
    for name, path in manifest["outputs"].items():
        print(f"{name}: {path}")
    print(f"manifest: {paths.manifest_path}")
    return manifest
