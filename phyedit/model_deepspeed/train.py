import argparse
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(PROJECT_ROOT))
if os.path.exists(dotenv_path := PROJECT_ROOT / ".env"):
    load_dotenv(dotenv_path=dotenv_path)

from phyedit.data.dataset import ImagePairDataLoader, qwen_edit_base_area_from_config  # noqa: E402
from phyedit.data.subset_utils import load_metadata  # noqa: E402
from phyedit.model_deepspeed.callbacks import TrainingCallbackDeepSpeed, is_rank_zero  # noqa: E402
from phyedit.model_deepspeed.model import TrainableModel  # noqa: E402
from phyedit.utils.file_utils import get_config  # noqa: E402


def process_is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def init_wandb(wandb_config, run_name, wandb_api_key=None):
    try:
        if wandb_api_key is None:
            wandb_api_key = os.getenv("WANDB_API_KEY", None)
        assert wandb_api_key is not None, "Please provide a valid WANDB_API_KEY"
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def log_crash(exc: BaseException, crash_dir: str | os.PathLike | None = None):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(tb, flush=True)

    crash_path = None
    if crash_dir is not None:
        crash_dir = Path(crash_dir)
        crash_dir.mkdir(parents=True, exist_ok=True)
        crash_path = crash_dir / f"crash_rank{os.environ.get('RANK', '0')}.log"
        crash_path.write_text(tb, encoding="utf-8")
        print(f"[train_deepspeed] Crash traceback saved to {crash_path}", flush=True)

    if process_is_rank_zero() and wandb.run is not None:
        try:
            wandb.run.summary["crash_traceback_tail"] = tb[-8000:]
            if crash_path is not None:
                artifact = wandb.Artifact(f"{wandb.run.name}-crash", type="crash-log")
                artifact.add_file(str(crash_path))
                wandb.log_artifact(artifact)
        except Exception as wandb_exc:
            print(f"[train_deepspeed] Failed to upload crash log to WanDB: {wandb_exc}", flush=True)


def save_config(config, run_name):
    save_path = config["save_path"]
    model_name = config["model_name"]
    os.makedirs(f"{save_path}/{model_name}/{run_name}", exist_ok=True)
    with open(f"{save_path}/{model_name}/{run_name}/config.yaml", "w") as f:
        yaml.dump(config, f)


def parse_env_bool(name: str, default: bool | None = None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_env_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def apply_env_overrides(training_config: dict):
    int_overrides = {
        "TRAIN_MAX_STEPS": "max_steps",
        "TRAIN_MAX_EPOCHS": "max_epochs",
        "TRAIN_DATALOADER_WORKERS": "dataloader_workers",
        "TRAIN_SAVE_INTERVAL": "save_interval",
        "TRAIN_SAMPLE_INTERVAL": "sample_interval",
        "TRAIN_PRINT_EVERY_N_STEPS": "print_every_n_steps",
        "TRAIN_SEED": "seed",
        "TRAIN_ACCUMULATE_GRAD_BATCHES": "accumulate_grad_batches",
        "TRAIN_IMAGE_BASE_SIZE": "image_base_size",
        "TRAIN_IMAGE_BASE_AREA": "image_base_area",
        "TRAIN_IMAGE_TRAIN_BASE_SIZE": "train_image_base_size",
        "TRAIN_IMAGE_TRAIN_BASE_AREA": "train_image_base_area",
        "TRAIN_IMAGE_SAMPLE_BASE_SIZE": "sample_image_base_size",
        "TRAIN_IMAGE_SAMPLE_BASE_AREA": "sample_image_base_area",
        "TRAIN_IMAGE_TRAIN_LONGER_SIDE": "train_longer_side",
        "TRAIN_IMAGE_SAMPLE_LONGER_SIDE": "sample_longer_side",
        "TRAIN_IMAGE_SAMPLE_HEIGHT": "sample_height",
        "TRAIN_IMAGE_SAMPLE_WIDTH": "sample_width",
    }
    for env_name, config_key in int_overrides.items():
        if env_name in os.environ:
            training_config[config_key] = int(os.environ[env_name])

    float_overrides = {
        "TRAIN_LEGACY_RESUME_LR": "legacy_resume_lr",
        "TRAIN_OPTIMIZER_LR": "optimizer_lr_override",
        "TRAIN_DEPTH_LOSS_LAMBDA": "depth_loss_lambda",
    }
    for env_name, config_key in float_overrides.items():
        parsed = parse_env_float(env_name)
        if parsed is not None:
            training_config[config_key] = parsed

    bool_overrides = {
        "TRAIN_DEPTH_SUPERVISE": "depth_supervise",
        "TRAIN_DATALOADER_PIN_MEMORY": "dataloader_pin_memory",
        "TRAIN_NEED_MOVED_IMAGE": "need_moved_image",
        "TRAIN_AUTO_RESUME": "auto_resume",
        "TRAIN_RESUME_PRESERVE_EFFECTIVE_BATCH": "resume_preserve_effective_batch",
    }
    for env_name, config_key in bool_overrides.items():
        parsed = parse_env_bool(env_name)
        if parsed is not None:
            training_config[config_key] = parsed

    if "TRAIN_DATALOADER_MULTIPROCESSING_CONTEXT" in os.environ:
        value = os.environ["TRAIN_DATALOADER_MULTIPROCESSING_CONTEXT"].strip()
        training_config["dataloader_multiprocessing_context"] = value or None

    if "TRAIN_CHECKPOINT_PATH" in os.environ:
        value = os.environ["TRAIN_CHECKPOINT_PATH"].strip()
        training_config["checkpoint_path"] = value or None

    if "TRAIN_RESUME_WORLD_SIZE_POLICY" in os.environ:
        training_config["resume_world_size_policy"] = os.environ[
            "TRAIN_RESUME_WORLD_SIZE_POLICY"
        ].strip()

    if parse_env_bool("TRAIN_DISABLE_WANDB", False):
        training_config.pop("wandb", None)


def init_distributed_if_needed(local_rank_arg: int | None = None):
    if local_rank_arg is not None and local_rank_arg >= 0:
        os.environ["LOCAL_RANK"] = str(local_rank_arg)

    # VS Code/debugpy starts a single Python process directly, without the
    # environment that torchrun/deepspeed launcher normally provides.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        deepspeed.init_distributed()

    return local_rank


def seed_process(base_seed: int) -> int:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(
        os.environ.get("RANK", "0")
    )
    process_seed = int(base_seed) + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
    return process_seed


def build_deepspeed_config(training_config: dict, dtype: str) -> dict:
    ds_user_cfg = training_config.get("deepspeed", {})
    zero_stage = int(ds_user_cfg.get("zero_stage", 2))

    overlap_comm = bool(ds_user_cfg.get("overlap_comm", True))
    contiguous_gradients = bool(ds_user_cfg.get("contiguous_gradients", True))
    reduce_scatter = bool(ds_user_cfg.get("reduce_scatter", True))

    offload_optimizer_device = str(
        ds_user_cfg.get("offload_optimizer_device", "none")
    ).lower()

    offload_param_device = str(
        ds_user_cfg.get("offload_param_device", "none")
    ).lower()

    pin_memory = bool(ds_user_cfg.get("pin_memory", True))

    zero_optimization = {
        "stage": zero_stage,
        "overlap_comm": overlap_comm,
        "contiguous_gradients": contiguous_gradients,
        "reduce_scatter": reduce_scatter,
    }

    if offload_optimizer_device in {"cpu", "nvme"}:
        zero_optimization["offload_optimizer"] = {
            "device": offload_optimizer_device,
            "pin_memory": pin_memory,
        }

    if offload_param_device in {"cpu", "nvme"}:
        zero_optimization["offload_param"] = {
            "device": offload_param_device,
            "pin_memory": pin_memory,
        }

    optimizer_cfg = training_config["optimizer"]

    ds_config = {
        "train_micro_batch_size_per_gpu": int(training_config["batch_size"]),
        "gradient_accumulation_steps": int(
            training_config["accumulate_grad_batches"]
        ),
        "gradient_clipping": float(
            training_config.get("gradient_clip_val", 0.5)
        ),
        "steps_per_print": int(
            training_config.get("print_every_n_steps", 50)
        ),
        "wall_clock_breakdown": False,

        # Let DeepSpeed select the optimizer implementation for the offload mode.
        "optimizer": {
            "type": optimizer_cfg.get("type", "AdamW"),
            "params": dict(optimizer_cfg.get("params", {})),
        },

        "zero_optimization": zero_optimization,
        "bf16": {"enabled": dtype == "bfloat16"},
        "fp16": {"enabled": dtype == "float16"},
    }

    return ds_config


def format_param_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.3f}B ({num_params:,})"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.3f}M ({num_params:,})"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.3f}K ({num_params:,})"
    return f"{num_params:,}"


def print_model_summary(model):
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

    frozen_params = total_params - trainable_params
    trainable_ratio = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    print("[train_deepspeed] Model summary")
    print(f"  Total params:     {format_param_count(total_params)}")
    print(f"  Trainable params: {format_param_count(trainable_params)}")
    print(f"  Frozen params:    {format_param_count(frozen_params)}")
    print(f"  Trainable ratio:   {trainable_ratio:.4f}%")


def apply_optimizer_overrides(
    training_config: dict,
    model: TrainableModel,
    *,
    restore_deepspeed_optimizer: bool,
):
    optimizer = training_config.setdefault("optimizer", {})
    params = optimizer.setdefault("params", {})
    if "optimizer_lr_override" in training_config:
        params["lr"] = float(training_config["optimizer_lr_override"])

    # A LoRA-only resume has no usable Adam moments. Starting it at the original
    # full learning rate can make the first AdamW updates destructive.
    if model.total_steps > 0 and not restore_deepspeed_optimizer:
        fallback_lr = training_config.get("legacy_resume_lr")
        if fallback_lr is not None:
            old_lr = params.get("lr")
            params["lr"] = float(fallback_lr)
            if is_rank_zero():
                print(
                    "[train_deepspeed] DeepSpeed optimizer state is not being restored; "
                    f"using legacy resume lr={params['lr']:.3g} (config lr was {old_lr!r})"
                )


def _resolve_deepspeed_resume_spec(
    model: TrainableModel, training_config: dict
) -> dict | None:
    configured_spec = training_config.get("resume_deepspeed_checkpoint")
    if configured_spec is False:
        return None
    if isinstance(configured_spec, str) and configured_spec.strip():
        return {
            "directory": os.path.expanduser(configured_spec),
            "tag": None,
        }
    if isinstance(configured_spec, dict):
        return dict(configured_spec)
    return model.deepspeed_resume_spec()


def _infer_deepspeed_checkpoint_world_size(resume_spec: dict | None) -> int | None:
    if not resume_spec:
        return None

    checkpoint_dir = Path(os.path.expanduser(str(resume_spec.get("directory", ""))))
    if not checkpoint_dir.is_dir():
        return None

    tag = resume_spec.get("tag")
    if not tag:
        latest_path = checkpoint_dir / "latest"
        if latest_path.is_file():
            tag = latest_path.read_text(encoding="utf-8").strip()
    tag_dir = checkpoint_dir / str(tag) if tag else checkpoint_dir

    ranks = set()
    for path in tag_dir.glob("*zero_pp_rank_*_optim_states.pt"):
        match = re.search(r"zero_pp_rank_(\d+)", path.name)
        if match:
            ranks.add(int(match.group(1)))
    return len(ranks) or None


def build_resume_plan(
    model: TrainableModel,
    training_config: dict,
    *,
    current_world_size: int | None = None,
) -> dict:
    current_world_size = int(
        current_world_size
        if current_world_size is not None
        else (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1)
    )
    checkpoint_state = dict(model.loaded_checkpoint_state or {})
    resume_spec = _resolve_deepspeed_resume_spec(model, training_config)
    has_checkpoint = bool(model.loaded_checkpoint_path) or model.total_steps > 0

    saved_world_size = checkpoint_state.get("world_size")
    if saved_world_size is None:
        saved_world_size = training_config.get("resume_world_size")
    if saved_world_size is None:
        saved_world_size = _infer_deepspeed_checkpoint_world_size(resume_spec)
    saved_world_size = int(saved_world_size) if saved_world_size is not None else None

    policy = str(training_config.get("resume_world_size_policy", "error")).strip().lower()
    if policy not in {"error", "lora_only"}:
        raise ValueError(
            "resume_world_size_policy must be 'error' or 'lora_only', "
            f"got {policy!r}"
        )

    mode = "fresh"
    if has_checkpoint:
        mode = "exact" if resume_spec is not None else "lora_only"

    world_size_changed = bool(
        has_checkpoint
        and saved_world_size is not None
        and saved_world_size != current_world_size
    )
    if world_size_changed:
        if policy == "error":
            raise RuntimeError(
                "Checkpoint was created with "
                f"world_size={saved_world_size}, but the current run has "
                f"world_size={current_world_size}. Set "
                "resume_world_size_policy: lora_only to reset optimizer/RNG state "
                "and resume weights plus a converted data cursor."
            )
        mode = "lora_only"

    resume_epoch = int(
        checkpoint_state.get("epoch", training_config.get("resume_epoch", 0) or 0)
    )
    saved_next_batch_idx = int(
        checkpoint_state.get(
            "next_batch_idx", training_config.get("resume_batch_idx", 0) or 0
        )
    )
    global_batch_cursor = checkpoint_state.get("global_batch_cursor")
    if global_batch_cursor is None and saved_world_size is not None:
        global_batch_cursor = saved_next_batch_idx * saved_world_size

    resume_batch_idx = saved_next_batch_idx
    replayed_global_batches = 0
    if world_size_changed and global_batch_cursor is not None:
        global_batch_cursor = int(global_batch_cursor)
        resume_batch_idx = global_batch_cursor // current_world_size
        replayed_global_batches = global_batch_cursor % current_world_size

    original_accumulate = int(training_config.get("accumulate_grad_batches", 1))
    adjusted_accumulate = original_accumulate
    preserve_effective_batch = bool(
        training_config.get("resume_preserve_effective_batch", False)
    )
    if world_size_changed and preserve_effective_batch:
        saved_config = checkpoint_state.get("config", {})
        if not isinstance(saved_config, dict):
            saved_config = {}
        saved_batch_size = int(
            saved_config.get("batch_size", training_config.get("batch_size", 1))
        )
        saved_accumulate = int(
            checkpoint_state.get(
                "accumulate_grad_batches",
                saved_config.get("accumulate_grad_batches", original_accumulate),
            )
        )
        current_batch_size = int(training_config.get("batch_size", 1))
        saved_effective_batch = saved_batch_size * saved_accumulate * saved_world_size
        current_batch_factor = current_batch_size * current_world_size
        if saved_effective_batch % current_batch_factor != 0:
            raise RuntimeError(
                "Cannot preserve effective batch exactly when changing world size: "
                f"saved_effective_batch={saved_effective_batch}, "
                f"current_batch_size={current_batch_size}, "
                f"current_world_size={current_world_size}. Set "
                "resume_preserve_effective_batch: false and choose "
                "accumulate_grad_batches explicitly."
            )
        adjusted_accumulate = saved_effective_batch // current_batch_factor
        if adjusted_accumulate <= 0:
            raise RuntimeError("Computed accumulate_grad_batches must be positive")
        training_config["accumulate_grad_batches"] = adjusted_accumulate
        model.accumulate_grad_batches = adjusted_accumulate

    restore_deepspeed_optimizer = mode == "exact"
    return {
        "mode": mode,
        "resume_spec": resume_spec,
        "saved_world_size": saved_world_size,
        "current_world_size": current_world_size,
        "world_size_changed": world_size_changed,
        "resume_epoch": resume_epoch,
        "saved_next_batch_idx": saved_next_batch_idx,
        "resume_batch_idx": resume_batch_idx,
        "global_batch_cursor": global_batch_cursor,
        "replayed_global_batches": replayed_global_batches,
        "original_accumulate_grad_batches": original_accumulate,
        "accumulate_grad_batches": adjusted_accumulate,
        "restore_deepspeed_optimizer": restore_deepspeed_optimizer,
    }


def load_deepspeed_resume(
    engine,
    model: TrainableModel,
    training_config: dict,
    resume_plan: dict,
) -> dict | None:
    if resume_plan.get("mode") != "exact":
        return None

    resume_spec = resume_plan.get("resume_spec")

    if not resume_spec:
        return None

    checkpoint_dir = Path(os.path.expanduser(str(resume_spec["directory"])))
    tag = resume_spec.get("tag")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"DeepSpeed resume directory does not exist: {checkpoint_dir}. "
            "Use a LoRA-only fallback or restore the matching training checkpoint."
        )

    if is_rank_zero():
        print(
            f"[train_deepspeed] Loading DeepSpeed training state: "
            f"dir={checkpoint_dir}, tag={tag or 'latest'}"
        )
    loaded_path, client_state = engine.load_checkpoint(
        str(checkpoint_dir),
        tag=tag,
        load_module_strict=False,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )
    if loaded_path is None:
        raise RuntimeError(f"DeepSpeed checkpoint load failed: {checkpoint_dir} tag={tag!r}")

    client_state = client_state or {}
    model.total_steps = int(client_state.get("optimizer_steps", model.total_steps))
    model.batch_steps = int(client_state.get("batch_steps", model.batch_steps))
    return client_state


def restore_resume_rng_state(client_state: dict | None):
    if not client_state:
        return
    states = client_state.get("rng_states")
    if not states:
        return

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank >= len(states) or states[rank] is None:
        raise RuntimeError(f"Missing RNG state for rank {rank} in DeepSpeed checkpoint")

    state = states[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])


def main(config_path: str = "./configs/train_deepspeed.yaml", local_rank_arg: int | None = None):
    local_rank = init_distributed_if_needed(local_rank_arg=local_rank_arg)

    config = get_config(config_path)
    training_config = config["train"]
    apply_env_overrides(training_config)
    process_seed = seed_process(int(training_config.get("seed", 42)))
    if is_rank_zero():
        print(f"[train_deepspeed] base_seed={training_config.get('seed', 42)}, rank0_seed={process_seed}")
    run_name = os.environ.get("RUN_NAME") or f"{training_config['model_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(training_config["save_path"], training_config["model_name"], run_name)
    train_longer_side = training_config.get("train_longer_side")
    train_base_area = (
        None
        if train_longer_side is not None
        else qwen_edit_base_area_from_config(
            training_config,
            area_key="train_image_base_area",
            size_key="train_image_base_size",
        )
    )

    dataset_config = training_config.get("dataset", {})
    metadata_path = dataset_config.get("metadata_path") or os.getenv("REALMANIP_METADATA_PATH")
    train_metadata = load_metadata(metadata_path)

    train_loader = ImagePairDataLoader(
        metadata=train_metadata,
        batch_size=training_config["batch_size"],
        num_workers=training_config["dataloader_workers"],
        base_area=train_base_area,
        longer_side=train_longer_side,
        shuffle=True,
        need_moved_image=training_config.get("need_moved_image", False),
        text_drop_rate=training_config.get("text_drop_rate", 0.0),
        multiprocessing_context=training_config.get("dataloader_multiprocessing_context"),
        pin_memory=training_config.get("dataloader_pin_memory", False),
    )

    if is_rank_zero():
        dataset_size = len(getattr(train_loader, "dataset", []))
        print(f"Dataset length: {dataset_size}")
        print(
            "[train_deepspeed] dataloader "
            f"workers={training_config['dataloader_workers']}, "
            f"multiprocessing_context={training_config.get('dataloader_multiprocessing_context') or 'default'}, "
            f"pin_memory={training_config.get('dataloader_pin_memory', False)}, "
            f"base_area={train_base_area or 'disabled'}, "
            f"longer_side={train_longer_side or 'disabled'}"
        )

    callback = TrainingCallbackDeepSpeed(run_name=run_name, training_config=training_config)
    callback.sync_run_name()
    run_name = callback.run_name
    run_dir = os.path.join(training_config["save_path"], training_config["model_name"], run_name)

    trainable_model = TrainableModel(
        pipe_id=config["pretrained_path"],
        save_dir=os.path.join(training_config["save_path"], training_config["model_name"]),
        training_config=training_config,
        dtype=getattr(torch, config["dtype"]),
        train_method=training_config.get("train_method", "sft"),
        process_visualize=training_config.get("process_visualize", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )
    trainable_model.setup(stage="fit")

    if is_rank_zero():
        print_model_summary(trainable_model)

    current_world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    resume_plan = build_resume_plan(
        trainable_model,
        training_config,
        current_world_size=current_world_size,
    )
    if is_rank_zero() and trainable_model.total_steps > 0:
        if resume_plan["mode"] == "exact":
            print(
                "[train_deepspeed] Resume mode=exact: restoring LoRA, optimizer, "
                "scheduler, RNG, and data cursor."
            )
        else:
            print(
                "[train_deepspeed] Resume mode=lora_only: restoring LoRA and counters; "
                "optimizer, scheduler, and per-rank RNG state will be reset."
            )
            if resume_plan["world_size_changed"]:
                print(
                    "[train_deepspeed] World-size conversion: "
                    f"{resume_plan['saved_world_size']} -> {resume_plan['current_world_size']}, "
                    f"epoch={resume_plan['resume_epoch']}, "
                    f"local_batch={resume_plan['saved_next_batch_idx']} -> "
                    f"{resume_plan['resume_batch_idx']}, "
                    f"replayed_global_batches={resume_plan['replayed_global_batches']}, "
                    "accumulate_grad_batches="
                    f"{resume_plan['original_accumulate_grad_batches']} -> "
                    f"{resume_plan['accumulate_grad_batches']}"
                )

    resume_mismatches = trainable_model.resume_config_mismatches(training_config)
    if resume_plan["world_size_changed"] and training_config.get(
        "resume_preserve_effective_batch", False
    ):
        resume_mismatches = [
            mismatch
            for mismatch in resume_mismatches
            if not mismatch.startswith("accumulate_grad_batches:")
        ]
    if resume_mismatches:
        message = "[train_deepspeed] Resume config mismatch:\n  " + "\n  ".join(resume_mismatches)
        if training_config.get("strict_resume_config", False):
            raise RuntimeError(message)
        if is_rank_zero():
            print(f"WARNING: {message}")

    apply_optimizer_overrides(
        training_config,
        trainable_model,
        restore_deepspeed_optimizer=resume_plan["restore_deepspeed_optimizer"],
    )

    # optimizer = trainable_model.configure_optimizers()
    ds_config = build_deepspeed_config(training_config, config["dtype"])
    trainable_params = trainable_model.get_optimize_parameters()

    if is_rank_zero():
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        effective_batch = (
            ds_config["train_micro_batch_size_per_gpu"]
            * ds_config["gradient_accumulation_steps"]
            * world_size
        )
        optimizer_lr = ds_config["optimizer"]["params"].get("lr")
        print(
            f"[train_deepspeed] local_rank={local_rank}, zero_stage={ds_config['zero_optimization']['stage']}, "
            f"grad_accum={ds_config['gradient_accumulation_steps']}, effective_batch={effective_batch}, "
            f"lr={optimizer_lr}"
        )

    engine, _, _, _ = deepspeed.initialize(
        model=trainable_model,
        model_parameters=trainable_params,
        config=ds_config,
        dist_init_required=None,
    )

    resume_state = load_deepspeed_resume(
        engine,
        trainable_model,
        training_config,
        resume_plan,
    )
    if resume_state is not None:
        saved_world_size = resume_state.get("world_size")
        current_world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if saved_world_size is not None and int(saved_world_size) != current_world_size:
            raise RuntimeError(
                "DeepSpeed checkpoint was created with "
                f"world_size={saved_world_size}, but the current run has world_size={current_world_size}. "
                "Resume with the same number of GPUs."
            )
        resume_epoch = int(resume_state.get("epoch", 0))
        resume_batch_idx = int(resume_state.get("next_batch_idx", 0))
        if is_rank_zero():
            print(
                f"[train_deepspeed] Resumed counters: optimizer_steps={trainable_model.total_steps}, "
                f"batch_steps={trainable_model.batch_steps}, epoch={resume_epoch}, "
                f"next_batch={resume_batch_idx}"
            )
    else:
        resume_epoch = int(resume_plan["resume_epoch"])
        resume_batch_idx = int(resume_plan["resume_batch_idx"])
        if trainable_model.total_steps > 0 and is_rank_zero():
            print(
                "[train_deepspeed] LoRA-only resume: optimizer state is not being restored; "
                "using the checkpoint data cursor "
                f"epoch={resume_epoch}, batch={resume_batch_idx}."
            )

    if is_rank_zero():
        wandb_config = training_config.get("wandb", None)
        if wandb_config is not None:
            init_wandb(wandb_config, run_name, os.getenv("WANDB_API_KEY", None))
        save_config(training_config, run_name)

    max_steps = int(training_config.get("max_steps", -1))
    max_epochs = int(training_config.get("max_epochs", -1))
    try:
        callback.on_train_start(engine.module)
        restore_resume_rng_state(resume_state)
        epoch = resume_epoch
        first_epoch = True
        should_stop = False
        while True:
            engine.train()
            start_batch_idx = resume_batch_idx if first_epoch else 0
            callback.on_train_epoch_start(train_loader, epoch, start_batch_idx=start_batch_idx)
            next_batch_idx = start_batch_idx

            for local_batch_idx, batch in enumerate(train_loader):
                batch_idx = start_batch_idx + local_batch_idx
                loss = engine.module.training_step(batch, batch_idx)
                engine.backward(loss)
                callback.collect_gradient_stats(engine.module, engine)
                optimizer_step_happened = (
                    engine.is_gradient_accumulation_boundary()
                    if hasattr(engine, "is_gradient_accumulation_boundary")
                    else True
                )
                engine.step()

                loss_to_log = float(loss.detach().item())
                callback.on_train_batch_end(
                    engine.module,
                    engine,
                    loss_to_log,
                    batch_idx,
                    epoch,
                    optimizer_step_happened=optimizer_step_happened,
                )
                next_batch_idx = batch_idx + 1

                if max_steps > 0 and engine.module.total_steps >= max_steps:
                    should_stop = True
                    break

            completed_epoch = not should_stop
            callback.on_train_epoch_end(
                engine.module,
                engine,
                epoch,
                next_epoch=epoch + 1 if completed_epoch else epoch,
                next_batch_idx=0 if completed_epoch else next_batch_idx,
            )

            if should_stop:
                break
            epoch += 1
            first_epoch = False
            resume_batch_idx = 0
            if max_epochs > 0 and epoch >= max_epochs:
                break
            if max_steps < 0 and max_epochs < 0:
                # Keep parity with previous behavior: if both unset, keep training.
                # This guard is only to avoid accidental infinite loops when dataset is empty.
                if len(train_loader) == 0:
                    break

        if is_rank_zero():
            print("[train_deepspeed] Training completed")
            if wandb.run is not None:
                wandb.finish(exit_code=0)
    except BaseException as exc:
        log_crash(exc, crash_dir=run_dir)
        if process_is_rank_zero() and wandb.run is not None:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_deepspeed.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=-1)
    args = parser.parse_args()
    main(config_path=args.config, local_rank_arg=args.local_rank)
