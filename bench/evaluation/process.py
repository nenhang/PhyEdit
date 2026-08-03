from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .io import utc_now


class StageExecutionError(RuntimeError):
    def __init__(self, record: dict[str, Any], cause: BaseException):
        super().__init__(str(cause))
        self.record = record
        self.__cause__ = cause


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Python executable not found: {candidate}")
        return str(candidate.resolve())

    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"Executable not found on PATH: {value}")
    return resolved


def run_stage(
    *,
    stage: str,
    command: list[str],
    environment: dict[str, str],
    project_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    print(f"\n===== Stage: {stage} =====")
    print(shlex.join(command), flush=True)
    record: dict[str, Any] = {
        "stage": stage,
        "command": command,
        "started_at": utc_now(),
        "status": "planned" if dry_run else "running",
    }
    if dry_run:
        record["finished_at"] = utc_now()
        record["elapsed_seconds"] = 0.0
        return record

    start = time.monotonic()
    try:
        subprocess.run(command, cwd=project_root, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        record["status"] = "failed"
        record["returncode"] = getattr(error, "returncode", None)
        record["finished_at"] = utc_now()
        record["elapsed_seconds"] = round(time.monotonic() - start, 3)
        raise StageExecutionError(record, error) from error

    record["status"] = "completed"
    record["returncode"] = 0
    record["finished_at"] = utc_now()
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return record
