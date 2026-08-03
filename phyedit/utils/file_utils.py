import json
from pathlib import Path

import yaml


def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "", postfix: str = ".safetensors") -> tuple[str, int] | None:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists() or not ckpt_path.is_dir():
        return None

    latest_checkpoint = None
    latest_step = -1
    latest_mtime = -1.0

    for pth_file in ckpt_path.glob(f"*/ckpt/{prefix}*{postfix}"):
        try:
            step_str = pth_file.stem[len(prefix) :]
            filename_step = int(step_str) if step_str.isdigit() else -1
        except ValueError:
            filename_step = -1

        step = filename_step
        metadata_path = pth_file.with_name(f"{pth_file.name}.json")
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    state = json.load(f)
                step = int(state.get("optimizer_steps", state.get("total_steps", filename_step)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                step = filename_step

        mtime = pth_file.stat().st_mtime
        if (step, mtime) > (latest_step, latest_mtime):
            latest_step = step
            latest_mtime = mtime
            latest_checkpoint = pth_file

    if latest_checkpoint is not None:
        return str(latest_checkpoint), latest_step

    return None
