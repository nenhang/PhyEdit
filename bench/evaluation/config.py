from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_STAGES = ("core", "motion", "deqa", "phys-vlm")
VALID_STAGES = frozenset(DEFAULT_STAGES)


def parse_stages(raw: str) -> tuple[str, ...]:
    requested = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not requested or requested == ["all"]:
        return DEFAULT_STAGES

    unknown = sorted(set(requested) - VALID_STAGES)
    if unknown:
        raise ValueError(f"Unknown evaluation stages: {', '.join(unknown)}")

    requested_set = set(requested)
    return tuple(stage for stage in DEFAULT_STAGES if stage in requested_set)


@dataclass(frozen=True)
class EvaluationPaths:
    project_root: Path
    benchmark_metadata: Path
    image_dir: Path
    output_dir: Path
    manifest_path: Path

    @classmethod
    def create(
        cls,
        *,
        project_root: Path,
        benchmark_metadata: Path,
        image_dir: Path,
        output_dir: Path | None,
        manifest_path: Path | None,
    ) -> EvaluationPaths:
        resolved_image_dir = image_dir.expanduser().resolve()
        resolved_output_dir = (output_dir or resolved_image_dir).expanduser().resolve()
        resolved_manifest = (
            (manifest_path or resolved_output_dir / "evaluation_manifest.json")
            .expanduser()
            .resolve()
        )
        return cls(
            project_root=project_root.expanduser().resolve(),
            benchmark_metadata=benchmark_metadata.expanduser().resolve(),
            image_dir=resolved_image_dir,
            output_dir=resolved_output_dir,
            manifest_path=resolved_manifest,
        )

    @property
    def bench_dir(self) -> Path:
        return self.project_root / "bench"

    @property
    def evaluation_dir(self) -> Path:
        return self.bench_dir / "evaluation"

    @property
    def artifact_dir(self) -> Path:
        return self.output_dir / "process_artifacts"

    @property
    def debug_dir(self) -> Path:
        return self.output_dir / "debug"

    @property
    def core_result(self) -> Path:
        return self.output_dir / "bench_score_new.json"

    @property
    def motion_result(self) -> Path:
        return self.output_dir / "bench_score_motion.json"

    @property
    def deqa_result(self) -> Path:
        return self.output_dir / "bench_score_penalty_deqa.json"

    @property
    def final_result(self) -> Path:
        return self.output_dir / "bench_score_vlm_logic.json"

    def manifest_outputs(self) -> dict[str, str]:
        return {
            "core": str(self.core_result),
            "motion": str(self.motion_result),
            "deqa": str(self.deqa_result),
            "final": str(self.final_result),
            "process_artifacts": str(self.artifact_dir),
            "debug": str(self.debug_dir),
        }


@dataclass(frozen=True)
class EvaluationConfig:
    stages: tuple[str, ...]
    main_python: str
    deqa_python: str
    device: str = "cuda:0"
    max_samples_per_bench_item: int = 8
    motion_alpha: float = 1.0
    motion_beta: float = 0.8
    motion_static_motion_ratio: float = 0.02
    deqa_batch_size: int = 64
    vlm_grounding_workers: int = 1
    depth_model_path: str = ""
    sam_model_path: str = ""
    dino_model_path: str = ""
    deqa_model_path: str = ""
    vlm_base_url: str = ""
    vlm_model_name: str = ""
    resume: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        if self.max_samples_per_bench_item <= 0:
            raise ValueError("--max-samples-per-bench-item must be positive")
        if self.deqa_batch_size <= 0:
            raise ValueError("--deqa-batch-size must be positive")
        if self.vlm_grounding_workers <= 0:
            raise ValueError("--vlm-grounding-workers must be positive")
        if "deqa" in self.stages and not self.deqa_model_path:
            raise ValueError(
                "The DeQA stage requires --deqa-model-path or DEQA_MODEL_PATH"
            )

    def model_overrides(self) -> dict[str, str | None]:
        return {
            "depth_model_path": self.depth_model_path or None,
            "sam_model_path": self.sam_model_path or None,
            "dino_model_path": self.dino_model_path or None,
            "vlm_base_url": self.vlm_base_url or None,
            "vlm_model_name": self.vlm_model_name or None,
            "deqa_model_path": self.deqa_model_path or None,
        }
