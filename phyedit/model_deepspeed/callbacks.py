import hashlib
import json
import os
import random
import shutil
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import wandb
from deepspeed.utils import safe_get_full_grad
from PIL import Image

from ..data.dataset import qwen_edit_base_area_from_config, resolve_qwen_edit_size
from ..data.subset_utils import load_benchmark_metadata
from ..utils.image_process import annotate_image_with_coordinates
from .generate import generate, render_moved_image_previews

SAMPLE_PREVIEW_CACHE_VERSION = 1


def is_rank_zero() -> bool:
    return (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    )


@contextmanager
def eval_mode(model):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        yield
    if was_training:
        model.train()


class TrainingCallbackDeepSpeed:
    def __init__(self, run_name, sample_data_loader=None, training_config=None):
        self.run_name = run_name
        self.training_config = training_config or {}
        self.sample_data_loader = sample_data_loader

        self.print_every_n_steps = self.training_config.get("print_every_n_steps", 10)
        self.save_interval = self.training_config.get("save_interval", 1000)
        self.sample_interval = int(
            self.training_config.get("sample_interval", 1000) or 0
        )
        self.save_deepspeed_state = bool(
            self.training_config.get("save_deepspeed_state", False)
        )
        self.keep_all_checkpoints = bool(
            self.training_config.get("keep_all_checkpoints", False)
        )
        self.deepspeed_state_keep_last = max(
            int(self.training_config.get("deepspeed_state_keep_last", 1)), 1
        )
        self.deepspeed_exclude_frozen_parameters = bool(
            self.training_config.get("deepspeed_exclude_frozen_parameters", True)
        )
        self.benchmark_metadata_path = self.training_config.get(
            "benchmark_metadata_path"
        )
        self.offload_depth_model_during_sampling = self.training_config.get(
            "offload_depth_model_during_sampling", False
        )
        self.save_path = os.path.join(
            self.training_config.get("save_path", "./output"),
            self.training_config.get("model_name", "main"),
        )

        self.num_samples_per_item = self.training_config.get("num_samples_per_item", 4)
        default_sample_seeds = [
            42 + sample_idx * 12345 for sample_idx in range(self.num_samples_per_item)
        ]
        self.sample_seeds = self.training_config.get(
            "sample_seeds", default_sample_seeds
        )
        self.sample_longer_side = self.training_config.get("sample_longer_side")
        self.sample_base_area = (
            None
            if self.sample_longer_side is not None
            else qwen_edit_base_area_from_config(
                self.training_config,
                area_key="sample_image_base_area",
                size_key="sample_image_base_size",
            )
        )
        self.sample_height = self.training_config.get("sample_height")
        self.sample_width = self.training_config.get("sample_width")
        self.sample_preview_cache_enabled = bool(
            self.training_config.get("sample_preview_cache", True)
        )
        configured_preview_cache_dir = self.training_config.get(
            "sample_preview_cache_dir"
        )
        self.sample_preview_cache_dir = os.path.abspath(
            os.path.expanduser(
                str(
                    configured_preview_cache_dir
                    or os.path.join(self.save_path, "sample_preview_cache")
                )
            )
        )

        self.wandb_config = self.training_config.get("wandb", None)
        self.use_wandb = wandb is not None and self.wandb_config is not None
        self.last_gradient_stats = {"mean": 0.0, "max": 0.0, "count": 0}
        self.initial_sample_generated = False
        self.last_training_state_key: tuple[int, int, int, int] | None = None

        if self.sample_data_loader is None:
            if self.sample_enabled():
                if not self.benchmark_metadata_path:
                    raise ValueError(
                        "sample_interval > 0 requires train.benchmark_metadata_path"
                    )
                self.sample_data = load_benchmark_metadata(
                    self.benchmark_metadata_path
                )
            else:
                self.sample_data = []
        else:
            self.sample_data = self.sample_data_loader.dataset.data

    def sample_enabled(self) -> bool:
        return self.sample_interval > 0

    def should_sample_on_step(self, total_steps: int) -> bool:
        return (
            self.sample_enabled()
            and total_steps > 0
            and total_steps % self.sample_interval == 0
        )

    def generate_sharded_sample(
        self, model, file_name_prefix: str, message: str | None = None
    ):
        rank0 = is_rank_zero()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if rank0 and message:
            print(message)
        torch.cuda.empty_cache()
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        try:
            self.generate_sample(
                model,
                f"{self.save_path}/{self.run_name}/samples",
                file_name_prefix,
                rank=rank,
                world_size=world_size,
                is_rank0=rank0,
            )
        finally:
            torch.cuda.empty_cache()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    def sync_run_name(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        synced_run_name = [self.run_name if is_rank_zero() else None]
        dist.broadcast_object_list(synced_run_name, src=0)
        self.run_name = synced_run_name[0]

    def _barrier(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _sample_preview_asset_signature(path_value) -> dict | None:
        if not path_value:
            return None
        path = os.path.abspath(os.path.expanduser(str(path_value)))
        signature = {"path": path}
        try:
            stat = os.stat(path)
        except OSError:
            return signature
        signature.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return signature

    def _sample_preview_cache_path(self, item: dict, width: int, height: int) -> str:
        payload = {
            "version": SAMPLE_PREVIEW_CACHE_VERSION,
            "render_order": "resize_lanczos_then_transform",
            "width": int(width),
            "height": int(height),
            "source": self._sample_preview_asset_signature(item.get("f1_path")),
            "masks": [
                self._sample_preview_asset_signature(path)
                for path in item.get("f1_mask_path", [])
            ],
            "depth": self._sample_preview_asset_signature(item.get("f1_depth_path")),
            "intrinsics": self._sample_preview_asset_signature(
                item.get("f1_intrinsic_path")
            ),
            "extrinsics": self._sample_preview_asset_signature(
                item.get("f1_extrinsic_path")
            ),
            "target_coords": item.get("f2_coords"),
        }
        digest = hashlib.sha1(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:16]
        bench_index = item.get("bench_index")
        label = f"bench_{int(bench_index):04d}" if bench_index is not None else "sample"
        filename = f"{label}_{int(width)}x{int(height)}_{digest}.png"
        return os.path.join(self.sample_preview_cache_dir, filename)

    @staticmethod
    def _load_sample_preview(
        path: str, expected_size: tuple[int, int] | None = None
    ) -> Image.Image:
        with Image.open(path) as image:
            image.load()
            preview = image.convert("RGB")
        if expected_size is not None and preview.size != expected_size:
            raise ValueError(
                f"Cached preview has size {preview.size}, expected {expected_size}: {path}"
            )
        return preview

    @staticmethod
    def _atomic_save_sample_preview(preview: Image.Image, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        try:
            preview.save(temporary, format="PNG")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _get_sample_preview(
        self,
        *,
        item: dict,
        src_image: Image.Image,
        mask_images: list[Image.Image],
        width: int,
        height: int,
        device: torch.device | str,
        is_rank0: bool,
    ) -> Image.Image:
        metadata_preview_path = item.get("moved_image_path")
        if metadata_preview_path and os.path.isfile(str(metadata_preview_path)):
            if is_rank0:
                print(f"[sample] Reusing metadata preview: {metadata_preview_path}")
            return self._load_sample_preview(str(metadata_preview_path))

        expected_size = (int(width), int(height))
        if not self.sample_preview_cache_enabled:
            return render_moved_image_previews(
                src_images=[src_image],
                mask_images=[mask_images],
                depth_images=[item["f1_depth_path"]],
                intrinsics=[item["f1_intrinsic_path"]],
                extrinsics=[item["f1_extrinsic_path"]],
                target_obj_coords=[item["f2_coords"]],
                device=device,
            )[0]

        cache_path = self._sample_preview_cache_path(item, width, height)
        status = [None]
        if is_rank0:
            try:
                try:
                    self._load_sample_preview(cache_path, expected_size=expected_size)
                    cache_status = "reused"
                except (OSError, ValueError):
                    preview = render_moved_image_previews(
                        src_images=[src_image],
                        mask_images=[mask_images],
                        depth_images=[item["f1_depth_path"]],
                        intrinsics=[item["f1_intrinsic_path"]],
                        extrinsics=[item["f1_extrinsic_path"]],
                        target_obj_coords=[item["f2_coords"]],
                        device=device,
                    )[0]
                    if preview.size != expected_size:
                        raise ValueError(
                            f"Rendered preview has size {preview.size}, expected {expected_size}"
                        )
                    self._atomic_save_sample_preview(preview, cache_path)
                    cache_status = "created"
                status[0] = {"ok": True, "status": cache_status, "path": cache_path}
            except Exception as error:
                status[0] = {"ok": False, "error": repr(error), "path": cache_path}

        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(status, src=0)
        elif not is_rank0:
            raise RuntimeError(
                "Sample preview cache requires rank 0 in non-distributed mode"
            )

        result = status[0]
        if not isinstance(result, dict) or not result.get("ok"):
            error = (
                result.get("error")
                if isinstance(result, dict)
                else "missing rank-0 cache status"
            )
            raise RuntimeError(f"Failed to prepare sample preview cache: {error}")
        if is_rank0:
            action = "Reusing" if result["status"] == "reused" else "Created"
            print(f"[sample] {action} preview cache: {result['path']}")
        return self._load_sample_preview(result["path"], expected_size=expected_size)

    def _config_signature(self) -> dict:
        optimizer = self.training_config.get("optimizer", {})
        optimizer_params = (
            optimizer.get("params", {}) if isinstance(optimizer, dict) else {}
        )
        return {
            "batch_size": self.training_config.get("batch_size"),
            "accumulate_grad_batches": self.training_config.get(
                "accumulate_grad_batches"
            ),
            "depth_loss_lambda": self.training_config.get("depth_loss_lambda"),
            "image_base_area": self.training_config.get("image_base_area"),
            "need_moved_image": self.training_config.get("need_moved_image"),
            "optimizer_type": optimizer.get("type")
            if isinstance(optimizer, dict)
            else None,
            "optimizer_lr": optimizer_params.get("lr"),
        }

    def _all_rank_rng_states(self) -> list[dict]:
        local_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        if not (dist.is_available() and dist.is_initialized()):
            return [local_state]

        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local_state)
        return states

    def _prune_deepspeed_states(self, checkpoint_dir: str, current_tag: str):
        if (
            self.keep_all_checkpoints
            or not is_rank_zero()
            or not os.path.isdir(checkpoint_dir)
        ):
            return

        tags = sorted(
            entry
            for entry in os.listdir(checkpoint_dir)
            if entry.startswith("step_")
            and os.path.isdir(os.path.join(checkpoint_dir, entry))
        )
        keep_tags = set(tags[-self.deepspeed_state_keep_last :])
        keep_tags.add(current_tag)
        for tag in tags:
            if tag not in keep_tags:
                shutil.rmtree(os.path.join(checkpoint_dir, tag))

    def save_training_checkpoint(
        self,
        model,
        engine,
        *,
        epoch: int,
        next_batch_idx: int,
        remove_old_lora: bool,
    ):
        deepspeed_checkpoint = None
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        global_batch_cursor = int(next_batch_idx) * int(world_size)
        if self.save_deepspeed_state:
            checkpoint_dir = os.path.join(self.save_path, self.run_name, "deepspeed")
            checkpoint_tag = f"step_{model.total_steps:05d}"
            client_state = {
                "optimizer_steps": int(model.total_steps),
                "batch_steps": int(model.batch_steps),
                "epoch": int(epoch),
                "next_batch_idx": int(next_batch_idx),
                "world_size": world_size,
                "global_batch_cursor": global_batch_cursor,
                "config": self._config_signature(),
                "rng_states": self._all_rank_rng_states(),
            }
            engine.save_checkpoint(
                checkpoint_dir,
                tag=checkpoint_tag,
                client_state=client_state,
                save_latest=True,
                exclude_frozen_parameters=self.deepspeed_exclude_frozen_parameters,
            )
            self._barrier()

            lora_dir = os.path.join(self.save_path, self.run_name, "ckpt")
            deepspeed_checkpoint = {
                "directory": os.path.relpath(checkpoint_dir, lora_dir),
                "tag": checkpoint_tag,
            }

        if is_rank_zero():
            model.save_optimize_parameters(
                run_name=self.run_name,
                total_steps=model.total_steps,
                remove_old=remove_old_lora,
                epoch=epoch,
                next_batch_idx=next_batch_idx,
                deepspeed_checkpoint=deepspeed_checkpoint,
                config_signature=self._config_signature(),
                world_size=world_size,
                global_batch_cursor=global_batch_cursor,
            )
        self._barrier()
        if self.save_deepspeed_state:
            self._prune_deepspeed_states(checkpoint_dir, checkpoint_tag)
            self._barrier()
        self.last_training_state_key = (
            int(model.total_steps),
            int(model.batch_steps),
            int(epoch),
            int(next_batch_idx),
        )

    def collect_gradient_stats(self, model, engine):
        if (
            hasattr(engine, "is_gradient_accumulation_boundary")
            and not engine.is_gradient_accumulation_boundary()
        ):
            self.last_gradient_stats = {"mean": 0.0, "max": 0.0, "count": 0}
            return self.last_gradient_stats

        grad_sum = 0.0
        grad_max = 0.0
        grad_count = 0

        for _, param in model.named_parameters():
            if not param.requires_grad:
                continue
            grad_tensor = safe_get_full_grad(param)
            if grad_tensor is None:
                continue

            grad_norm = grad_tensor.detach().float().norm(2).item()
            grad_sum += grad_norm
            grad_max = max(grad_max, grad_norm)
            grad_count += 1

        grad_mean = grad_sum / grad_count if grad_count > 0 else 0.0
        self.last_gradient_stats = {
            "mean": grad_mean,
            "max": grad_max,
            "count": grad_count,
        }
        return self.last_gradient_stats

    def on_train_epoch_start(self, train_loader, epoch: int, start_batch_idx: int = 0):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch, start_batch=start_batch_idx)
            if is_rank_zero():
                print(
                    f"Epoch {epoch} - Set epoch for batch sampler to {epoch}, "
                    f"resume_batch={start_batch_idx}"
                )

    def on_train_start(self, model):
        if self.initial_sample_generated or not self.sample_enabled():
            return
        self.initial_sample_generated = True
        total_steps = int(getattr(model, "total_steps", 0))
        self.generate_sharded_sample(
            model,
            f"{total_steps:05d}",
            f"Steps: {total_steps} - Generating initial sharded samples before training",
        )

    def on_train_batch_end(
        self,
        model,
        engine,
        loss,
        batch_idx: int,
        epoch: int,
        optimizer_step_happened: bool = True,
    ):
        rank0 = is_rank_zero()
        grad_stats = self.last_gradient_stats or {"mean": 0.0, "max": 0.0, "count": 0}
        has_grad_stats = grad_stats["count"] > 0

        model.batch_steps += 1
        if optimizer_step_happened:
            model.total_steps += 1
        log_metrics = model.pop_log_metrics() if optimizer_step_happened else {}
        loss_to_log = float(log_metrics.pop("loss", loss))

        if optimizer_step_happened and self.use_wandb and rank0 and wandb.run:
            report_dict = {
                "batch": batch_idx,
                "steps": model.total_steps,
                "batch_steps": model.batch_steps,
                "epoch": epoch,
                "loss": loss_to_log,
            }
            if has_grad_stats:
                report_dict["gradient_size"] = grad_stats["mean"]
                report_dict["max_gradient_size"] = grad_stats["max"]
            report_dict.update(log_metrics)
            wandb.log(report_dict, step=model.total_steps)

        if (
            optimizer_step_happened
            and rank0
            and model.total_steps % self.print_every_n_steps == 0
        ):
            if has_grad_stats:
                print(
                    f"Epoch: {epoch}, Steps: {model.total_steps}, Batch steps: {model.batch_steps}, Batch: {batch_idx}, Loss: {model.log_loss:.4f}, Gradient size: {grad_stats['mean']:.4f}, Max gradient size: {grad_stats['max']:.4f}"
                )
            else:
                print(
                    f"Epoch: {epoch}, Steps: {model.total_steps}, Batch steps: {model.batch_steps}, Batch: {batch_idx}, Loss: {model.log_loss:.4f}"
                )

        if optimizer_step_happened and model.total_steps % self.save_interval == 0:
            if rank0:
                state_label = (
                    "LoRA and DeepSpeed state"
                    if self.save_deepspeed_state
                    else "LoRA weights"
                )
                print(
                    f"Epoch: {epoch}, Steps: {model.total_steps}, Batch steps: {model.batch_steps} "
                    f"- Saving {state_label}"
                )
            self.save_training_checkpoint(
                model,
                engine,
                epoch=epoch,
                next_batch_idx=batch_idx + 1,
                remove_old_lora=not self.keep_all_checkpoints,
            )

        if optimizer_step_happened and self.should_sample_on_step(model.total_steps):
            self.generate_sharded_sample(
                model,
                f"{model.total_steps:05d}",
                f"Epoch: {epoch}, Steps: {model.total_steps} - Generating sharded samples",
            )

    def on_train_epoch_end(
        self,
        model,
        engine,
        epoch: int,
        *,
        next_epoch: int,
        next_batch_idx: int,
    ):
        rank0 = is_rank_zero()
        checkpoint_key = (
            int(model.total_steps),
            int(model.batch_steps),
            int(next_epoch),
            int(next_batch_idx),
        )
        if rank0 and self.last_training_state_key != checkpoint_key:
            print(
                f"Epoch: {epoch}, Steps: {model.total_steps}, Batch steps: {model.batch_steps} "
                "- Saving training state at epoch end"
            )
        if self.last_training_state_key != checkpoint_key:
            self.save_training_checkpoint(
                model,
                engine,
                epoch=next_epoch,
                next_batch_idx=next_batch_idx,
                remove_old_lora=False,
            )

        self.generate_sharded_sample(
            model,
            f"epoch_{epoch}",
            f"Epoch: {epoch} - Generating sharded samples at epoch end",
        )

    @contextmanager
    def depth_model_offloaded_for_sampling(self, model, is_rank0=True):
        depth_model = getattr(model, "depth_model", None)
        if (not self.offload_depth_model_during_sampling) or depth_model is None:
            yield
            return

        try:
            original_device = next(depth_model.parameters()).device
        except StopIteration:
            yield
            return

        if original_device.type == "cpu":
            yield
            return

        was_training = depth_model.training
        if is_rank0:
            print(
                f"[sample] Temporarily offloading depth_model to CPU from {original_device}"
            )
        depth_model.to("cpu")
        torch.cuda.empty_cache()

        try:
            yield
        finally:
            torch.cuda.empty_cache()
            if is_rank0:
                print(f"[sample] Moving depth_model back to {original_device}")
            depth_model.to(original_device)
            if was_training:
                depth_model.train()
            else:
                depth_model.eval()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_sample(
        self, model, save_path, file_name_prefix, rank=0, world_size=1, is_rank0=True
    ):
        with (
            self.depth_model_offloaded_for_sampling(model, is_rank0=is_rank0),
            eval_mode(model.transformer),
        ):
            selected_indices = self.training_config.get(
                "sample_bench_indices", [0, 12, 115, 124]
            )
            if "bench_index" in self.sample_data[0]:
                sample_data = [
                    self.sample_data[idx]
                    for idx in range(len(self.sample_data))
                    if self.sample_data[idx].get("bench_index", -1) in selected_indices
                ]
            else:
                sample_data = [
                    self.sample_data[idx]
                    for idx in selected_indices
                    if idx < len(self.sample_data)
                ]

            os.makedirs(save_path, exist_ok=True)
            local_seeds = self.sample_seeds[rank::world_size]

            for item in sample_data:
                src_image = Image.open(item["f1_path"]).convert("RGB")
                gt_image = Image.open(item["f2_path"]).convert("RGB")

                new_w, new_h = resolve_qwen_edit_size(
                    src_image.size,
                    base_area=self.sample_base_area,
                    longer_side=self.sample_longer_side,
                    height=self.sample_height,
                    width=self.sample_width,
                )

                src_image = src_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                gt_image = gt_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

                mask_images = []
                for mask_path in item["f1_mask_path"]:
                    mask_image = Image.open(mask_path).convert("L")
                    mask_image = mask_image.resize(
                        (new_w, new_h), Image.Resampling.LANCZOS
                    )
                    mask_images.append(mask_image)

                sample_subpath = os.path.join(
                    save_path, f"sample_{sample_data.index(item)}"
                )
                os.makedirs(sample_subpath, exist_ok=True)
                if is_rank0 and not os.path.exists(
                    os.path.join(sample_subpath, "condition.png")
                ):
                    annotate_image_with_coordinates(
                        src_image,
                        coordinates=item["f1_coords"],
                        depth_range=item["f1_depth_range"],
                    ).save(os.path.join(sample_subpath, "condition.png"))

                if is_rank0 and not os.path.exists(
                    os.path.join(sample_subpath, "gt.png")
                ):
                    annotate_image_with_coordinates(
                        gt_image,
                        coordinates=item["f2_coords"],
                        depth_range=item["f2_depth_range"],
                    ).save(os.path.join(sample_subpath, "gt.png"))

                preview_image = self._get_sample_preview(
                    item=item,
                    src_image=src_image,
                    mask_images=mask_images,
                    width=new_w,
                    height=new_h,
                    device=model.device,
                    is_rank0=is_rank0,
                )
                preview_display_path = os.path.join(sample_subpath, "preview.png")
                if is_rank0 and not os.path.exists(preview_display_path):
                    preview_image.save(preview_display_path, format="PNG")

                for seed in local_seeds:
                    res_image = generate(
                        pipeline=model.pipe,
                        src_image=[src_image],
                        mask_image=[mask_images],
                        depth_image=[item["f1_depth_path"]],
                        intrinsics=[item["f1_intrinsic_path"]],
                        extrinsics=[item["f1_extrinsic_path"]],
                        src_obj_coords=[item["f1_coords"]],
                        target_obj_coords=[item["f2_coords"]],
                        image_depth_range=[item["f1_depth_range"]],
                        objects=[item["object_name"]],
                        moved_image_input=[preview_image],
                        height=new_h,
                        width=new_w,
                        seed=seed,
                    )[0]

                    seed_subpath = os.path.join(sample_subpath, f"seed_{seed}")
                    os.makedirs(seed_subpath, exist_ok=True)
                    res_image.save(
                        os.path.join(seed_subpath, f"{file_name_prefix}_result.png")
                    )
