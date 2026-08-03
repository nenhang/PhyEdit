import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth
from diffusers import QwenImageEditPlusPipeline
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import Qwen2_5_VLForConditionalGeneration

from ..pipeline.image_processor import decode_images, encode_images, prepare_latents, preprocess_condition_images
from ..pipeline.text_encoder import encode_prompt
from ..utils.da3_api import preprocess_images_for_da3_model
from ..utils.file_utils import find_latest_checkpoint, get_config
from ..utils.image_process import annotate_image_with_coordinates, visualize_mask_weight_map


def _is_rank_zero() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def load_pipeline_from_config(
    config_path: str,
    device: str | torch.device = "cuda",
    checkpoint_path: str | os.PathLike | None = None,
    pretrained_model_path: str | os.PathLike | None = None,
):
    config = get_config(config_path)
    training_config = dict(config["train"])
    if checkpoint_path is not None:
        training_config["checkpoint_path"] = str(checkpoint_path)
    model = TrainableModel(
        pipe_id=str(pretrained_model_path or config["pretrained_path"]),
        save_dir=os.path.join(training_config["save_path"], training_config["model_name"]),
        training_config=training_config,
        dtype=getattr(torch, config["dtype"]),
    )
    model.setup(stage="predict")
    model.to(device)
    return model.pipe


def preprocess_mask_to_weight_map(f1_masks_list, f2_masks_list, latent_shape, expansion_px=12, bg_importance=0.1):
    bsz = len(f1_masks_list)
    h_lat, w_lat = latent_shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_weights = []
    for i in range(bsz):
        current_masks = []
        for m_pil in f1_masks_list[i] + f2_masks_list[i]:
            m_tensor = to_tensor(m_pil).to(device)
            m_lat = F.interpolate(m_tensor.unsqueeze(0), size=(h_lat, w_lat), mode="nearest")
            current_masks.append(m_lat)

        if not current_masks:
            batch_weights.append(torch.full((1, h_lat, w_lat), bg_importance, device=device))
            continue

        union_mask = torch.max(torch.cat(current_masks, dim=0), dim=0, keepdim=True)[0]
        k = 2 * expansion_px + 1
        dilated = F.max_pool2d(union_mask, kernel_size=k, stride=1, padding=expansion_px)
        smoothed = F.avg_pool2d(dilated, kernel_size=k, stride=1, padding=expansion_px)
        final_weight = smoothed * (1.0 - bg_importance) + bg_importance
        batch_weights.append(final_weight)

    return torch.cat(batch_weights, dim=0)


class TrainableModel(nn.Module):
    def __init__(
        self,
        pipe_id: str,
        training_config: dict | None = None,
        save_dir: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        process_visualize: bool = False,
        gradient_checkpointing: bool = False,
        train_method: str = "sft",
        debug_dir: str | None = "./debug",
    ):
        super().__init__()
        training_config = training_config or {}
        self.pipe_id = pipe_id
        self.lora_config = training_config["lora_config"]
        self.dpo_config = training_config["dpo_config"]
        self.optimizer_config = training_config["optimizer"]
        self.gradient_checkpointing = gradient_checkpointing
        self.process_visualize = process_visualize
        self.ckpt_dir = save_dir
        if train_method not in ["sft", "dpo"]:
            raise ValueError("train_method must be 'sft' or 'dpo'")
        self.train_method = train_method
        self.target_dtype = dtype
        self.total_steps = 0
        self.batch_steps = 0
        self.log_loss = 0.0
        self.last_t = 0.0
        self._pending_log_metrics = {}
        self._pending_log_counts = {}
        self.debug_dir = os.path.join(debug_dir, time.strftime("%Y%m%d-%H%M%S")) if debug_dir else None
        if self.debug_dir and _is_rank_zero():
            os.makedirs(self.debug_dir, exist_ok=True)

        self.depth_supervise = training_config.get("depth_supervise", True)
        self.depth_model_id = training_config.get("depth_model_id") or os.getenv(
            "DEPTH_MODEL_ID"
        )
        self.silog_loss_lambda = training_config.get("silog_loss_lambda", 0.5)
        self.depth_loss_lambda = training_config.get("depth_loss_lambda", 0.1)
        self.need_moved_image = training_config.get("need_moved_image", False)
        self.accumulate_grad_batches = int(training_config.get("accumulate_grad_batches", 1))
        self.checkpoint_name_prefix = training_config.get("checkpoint_name_prefix", "transformer_")
        self.checkpoint_path = training_config.get("checkpoint_path")
        self.auto_resume = bool(training_config.get("auto_resume", True))
        self.loaded_checkpoint_path: str | None = None
        self.loaded_checkpoint_state: dict = {}

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, stage: str):
        self.pipe: QwenImageEditPlusPipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.pipe_id, torch_dtype=self.target_dtype
        )
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.transformer.requires_grad_(False)
        self.text_encoder: Qwen2_5_VLForConditionalGeneration = self.pipe.text_encoder
        self.vae: AutoencoderKLQwenImage = self.pipe.vae
        self.transformer: QwenImageTransformer2DModel = self.pipe.transformer

        self.prepare_model(self.lora_config, stage=stage)

        if stage == "fit":
            if self.gradient_checkpointing:
                self.transformer.enable_gradient_checkpointing()
            self.transformer.train()

            self.depth_model = None
            if self.depth_supervise:
                if not self.depth_model_id:
                    raise ValueError(
                        "depth_supervise=true requires train.depth_model_id or DEPTH_MODEL_ID"
                    )
                self.depth_model = DepthAnything3.from_pretrained(self.depth_model_id).to(self.target_dtype)
                if hasattr(self.depth_model.model.da3, "gs_head") and self.depth_model.model.da3.gs_head is not None:
                    self.depth_model.model.da3.gs_head = None
                if (
                    hasattr(self.depth_model.model.da3, "gs_adapter")
                    and self.depth_model.model.da3.gs_adapter is not None
                ):
                    self.depth_model.model.da3.gs_adapter = None
                self.depth_model.requires_grad_(False).eval()
        else:
            self.transformer.eval()

    def _checkpoint_step_from_path(self, ckpt_path: str) -> int:
        stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        if stem.startswith(self.checkpoint_name_prefix):
            step_text = stem[len(self.checkpoint_name_prefix) :]
        else:
            step_text = stem
        return int(step_text) if step_text.isdigit() else 0

    def _find_lora_checkpoint(self) -> tuple[str, str] | None:
        if self.checkpoint_path:
            ckpt_path = os.path.abspath(os.path.expanduser(str(self.checkpoint_path)))
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Explicit checkpoint_path does not exist: {ckpt_path}")
            return ckpt_path, "explicit"

        if self.ckpt_dir is None or not self.auto_resume:
            return None

        latest_checkpoint = find_latest_checkpoint(self.ckpt_dir, prefix=self.checkpoint_name_prefix)
        if latest_checkpoint is None:
            return None
        ckpt_path, _ = latest_checkpoint
        return ckpt_path, "latest"

    def _load_lora_weights(self, ckpt_path: str, adapter_name: str = "default"):
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict=os.path.dirname(ckpt_path),
            adapter_name=adapter_name,
            weight_name=os.path.basename(ckpt_path),
        )

    def prepare_model(self, lora_config: dict, adapter_name: str = "default", stage: str = "fit"):
        resolved_checkpoint = self._find_lora_checkpoint()
        if resolved_checkpoint is None:
            if _is_rank_zero():
                if stage == "fit":
                    print("No checkpoint found, training from scratch.")
                else:
                    print("No checkpoint found; using a fresh LoRA adapter.")
            self.transformer.add_adapter(adapter_config=LoraConfig(**lora_config), adapter_name=adapter_name)
            return

        ckpt_path, source = resolved_checkpoint
        self.loaded_checkpoint_path = ckpt_path
        if _is_rank_zero():
            print(f"Loading {source} checkpoint: {ckpt_path}")

        checkpoint_steps = self._checkpoint_step_from_path(ckpt_path)
        checkpoint_state_path = f"{ckpt_path}.json"
        if os.path.exists(checkpoint_state_path):
            with open(checkpoint_state_path, "r", encoding="utf-8") as f:
                checkpoint_state = json.load(f)
            self.loaded_checkpoint_state = checkpoint_state
            self.total_steps = int(checkpoint_state.get("optimizer_steps", checkpoint_steps))
            self.batch_steps = int(checkpoint_state.get("batch_steps", self.total_steps * self.accumulate_grad_batches))
        else:
            self.batch_steps = checkpoint_steps
            self.total_steps = checkpoint_steps // max(1, self.accumulate_grad_batches)
            if stage == "fit" and _is_rank_zero():
                print(
                    "Checkpoint metadata not found; interpreted checkpoint step as legacy batch steps "
                    "and estimated update steps from current "
                    f"accumulate_grad_batches={self.accumulate_grad_batches}."
                )

        self._load_lora_weights(ckpt_path, adapter_name=adapter_name)

    def resume_config_mismatches(self, training_config: dict) -> list[str]:
        """Return structural resume differences recorded by newer checkpoints."""
        if not self.loaded_checkpoint_state:
            return []

        saved_config = dict(self.loaded_checkpoint_state.get("config", {}))
        if "accumulate_grad_batches" not in saved_config:
            saved_accumulate = self.loaded_checkpoint_state.get("accumulate_grad_batches")
            if saved_accumulate is not None:
                saved_config["accumulate_grad_batches"] = saved_accumulate

        current_optimizer = training_config.get("optimizer", {})
        current_params = current_optimizer.get("params", {}) if isinstance(current_optimizer, dict) else {}
        current_config = {
            "batch_size": training_config.get("batch_size"),
            "accumulate_grad_batches": training_config.get("accumulate_grad_batches"),
            "depth_loss_lambda": training_config.get("depth_loss_lambda"),
            "image_base_area": training_config.get("image_base_area"),
            "need_moved_image": training_config.get("need_moved_image"),
            "optimizer_type": current_optimizer.get("type") if isinstance(current_optimizer, dict) else None,
            "optimizer_lr": current_params.get("lr"),
        }

        mismatches = []
        for key, saved_value in saved_config.items():
            if key not in current_config or saved_value is None:
                continue
            current_value = current_config[key]
            if isinstance(saved_value, float) or isinstance(current_value, float):
                try:
                    equal = abs(float(saved_value) - float(current_value)) <= 1e-12
                except (TypeError, ValueError):
                    equal = saved_value == current_value
            else:
                equal = saved_value == current_value
            if not equal:
                mismatches.append(f"{key}: checkpoint={saved_value!r}, current={current_value!r}")
        return mismatches

    def deepspeed_resume_spec(self) -> dict | None:
        """Resolve the native DeepSpeed state location stored next to a LoRA checkpoint."""
        state = self.loaded_checkpoint_state.get("deepspeed_checkpoint")
        if not state or not self.loaded_checkpoint_path:
            return None

        if isinstance(state, str):
            state = {"directory": state}
        if not isinstance(state, dict):
            return None

        directory = state.get("directory")
        if not directory:
            return None
        directory = Path(os.path.expanduser(str(directory)))
        if not directory.is_absolute():
            directory = Path(self.loaded_checkpoint_path).parent / directory
        return {
            "directory": str(directory.resolve()),
            "tag": state.get("tag"),
        }

    def get_optimize_parameters(self):
        trainable_params = []
        for _, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def accumulate_log_metric(self, name: str, value: float):
        self._pending_log_metrics[name] = self._pending_log_metrics.get(name, 0.0) + float(value)
        self._pending_log_counts[name] = self._pending_log_counts.get(name, 0) + 1

    def pop_log_metrics(self) -> dict[str, float]:
        metrics = {
            name: value / max(1, self._pending_log_counts.get(name, 1))
            for name, value in self._pending_log_metrics.items()
        }
        self._pending_log_metrics = {}
        self._pending_log_counts = {}
        return metrics

    def save_optimize_parameters(
        self,
        run_name: str,
        total_steps: int,
        adapter_name: str = "default",
        remove_old: bool = True,
        batch_steps: int | None = None,
        epoch: int | None = None,
        next_batch_idx: int | None = None,
        deepspeed_checkpoint: dict | None = None,
        config_signature: dict | None = None,
        world_size: int | None = None,
        global_batch_cursor: int | None = None,
    ):
        if not _is_rank_zero():
            return
        if self.ckpt_dir is None:
            return

        save_dir = os.path.join(self.ckpt_dir, run_name, "ckpt")
        os.makedirs(save_dir, exist_ok=True)

        weight_name = f"{self.checkpoint_name_prefix}{total_steps:05d}.safetensors"
        metadata_name = f"{weight_name}.json"
        temp_suffix = f"{os.getpid()}.{time.time_ns()}"
        temp_weight_name = f".{Path(weight_name).stem}.{temp_suffix}.tmp.safetensors"
        temp_weight_path = os.path.join(save_dir, temp_weight_name)
        temp_metadata_path = os.path.join(save_dir, f".{metadata_name}.{temp_suffix}.tmp")
        weight_path = os.path.join(save_dir, weight_name)
        metadata_path = os.path.join(save_dir, metadata_name)
        transformer_lora_layers = get_peft_model_state_dict(self.transformer, adapter_name=adapter_name)

        checkpoint_state = {
            "weight_name": weight_name,
            "batch_steps": int(self.batch_steps if batch_steps is None else batch_steps),
            "optimizer_steps": int(total_steps),
            "total_steps": int(total_steps),
            "accumulate_grad_batches": self.accumulate_grad_batches,
        }
        if epoch is not None:
            checkpoint_state["epoch"] = int(epoch)
        if next_batch_idx is not None:
            checkpoint_state["next_batch_idx"] = int(next_batch_idx)
        if deepspeed_checkpoint is not None:
            checkpoint_state["deepspeed_checkpoint"] = dict(deepspeed_checkpoint)
        if world_size is not None:
            checkpoint_state["world_size"] = int(world_size)
        if global_batch_cursor is not None:
            checkpoint_state["global_batch_cursor"] = int(global_batch_cursor)
        if config_signature:
            checkpoint_state["config"] = dict(config_signature)

        try:
            QwenImageEditPlusPipeline.save_lora_weights(
                save_directory=save_dir,
                transformer_lora_layers=transformer_lora_layers,
                safe_serialization=True,
                weight_name=temp_weight_name,
            )
            with open(temp_metadata_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Publish metadata first. Until the weight rename succeeds, latest
            # checkpoint discovery still sees the previous complete weights.
            os.replace(temp_metadata_path, metadata_path)
            os.replace(temp_weight_path, weight_path)
        finally:
            for temp_path in (temp_weight_path, temp_metadata_path):
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if remove_old:
            keep_files = {weight_name, metadata_name}
            for file in os.listdir(save_dir):
                if file in keep_files:
                    continue
                if file.endswith(".safetensors") or file.endswith(".safetensors.json"):
                    os.remove(os.path.join(save_dir, file))

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = step_loss.item() if self.total_steps <= 1 else self.log_loss * 0.95 + step_loss.item() * 0.05
        return step_loss

    def step(self, batch):
        with torch.no_grad():
            raw_target_images = batch["f2_images"]
            raw_src_images = batch["f1_images"]
            prompts = batch["prompts"]
            moved_images = batch["moved_images"]

            target_images = self.pipe.image_processor.preprocess(raw_target_images)
            batch_size, _, height, width = target_images.shape
            target_latents = encode_images(self.pipe, target_images)

            source_images = (
                [[raw_src_images[i], moved_images[i]] for i in range(len(raw_src_images))]
                if self.need_moved_image
                else [[raw_src_images[i]] for i in range(len(raw_src_images))]
            )
            vae_images, vae_image_sizes, condition_images, _ = preprocess_condition_images(
                self.pipe,
                source_images,
                vae_image_size=width * height,
                condition_image_size=384 * 384,
            )

            prompt_embeds, prompt_embeds_mask = encode_prompt(
                self=self.pipe,
                image=condition_images,
                prompt=prompts,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )

            image_latents_list = []
            for j in range(len(vae_images[0])):
                vae_images_batch = torch.cat([vae_images[i][j] for i in range(len(vae_images))], dim=0)
                vae_image_latents = encode_images(self.pipe, vae_images_batch)
                image_latents_list.append(vae_image_latents)
            image_latents = torch.cat(image_latents_list, dim=1)

            num_channels_latents = self.transformer.config.in_channels // 4
            latents = prepare_latents(
                self=self.pipe,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=self.target_dtype,
                device=self.device,
                generator=None,
            )

            t = torch.sigmoid(torch.randn((batch_size,), device=self.device))
            x_1 = latents
            x_0 = target_latents
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.target_dtype)

            guidance = torch.ones_like(t).to(self.device) if self.pipe.transformer.config.guidance_embeds else None
            # txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
            img_shapes = [
                [
                    (1, height // self.pipe.vae_scale_factor // 2, width // self.pipe.vae_scale_factor // 2),
                    *[
                        (1, vae_height // self.pipe.vae_scale_factor // 2, vae_width // self.pipe.vae_scale_factor // 2)
                        for vae_width, vae_height in vae_image_sizes[0]
                    ],
                ]
            ] * batch_size

            noise_gt = x_1 - x_0
            latent_model_input = torch.cat([x_t, image_latents], dim=1).to(self.target_dtype)

            f1_masks = batch["f1_masks"]
            f2_masks = batch["f2_masks"]

            mask_weight_map = preprocess_mask_to_weight_map(
                f1_masks_list=f1_masks,
                f2_masks_list=f2_masks,
                latent_shape=(height // self.pipe.vae_scale_factor // 2, width // self.pipe.vae_scale_factor // 2),
                expansion_px=3,
                bg_importance=0.5,
            )
            mask_weight_map_flat = mask_weight_map.view(batch_size, -1, 1).expand(-1, -1, noise_gt.shape[-1])

        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=t,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            # txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.size(1)]

        if self.train_method != "sft":
            raise NotImplementedError("Only 'sft' is currently implemented in model_deepspeed")

        loss_map = F.mse_loss(noise_pred.float(), noise_gt.float(), reduction="none")
        loss = (loss_map * mask_weight_map_flat.float()).mean()
        self.accumulate_log_metric("mse_loss", loss.item())

        res = (x_t - noise_pred * t_).to(self.target_dtype)

        if self.depth_supervise and self.depth_model is not None:
            with torch.no_grad():
                src_images_tensor = self.pipe.image_processor.preprocess(raw_src_images)
                src_images_denormed = (src_images_tensor * 0.5 + 0.5).clamp(0, 1)
                src_images_preprocessed = preprocess_images_for_da3_model(
                    self.depth_model, src_images_denormed, process_res=max(height, width), batch_type="batch"
                )
                target_images_denormed = (target_images * 0.5 + 0.5).clamp(0, 1)
                target_images_preprocessed = preprocess_images_for_da3_model(
                    self.depth_model, target_images_denormed, process_res=max(height, width), batch_type="batch"
                )
                src_target_images_for_depth = torch.cat([src_images_preprocessed, target_images_preprocessed], dim=1)
                with torch.autocast(device_type=self.device.type, dtype=self.target_dtype):
                    depth_prediction = self.depth_model.model(src_target_images_for_depth)
                if "non_sky_mask" not in depth_prediction:
                    raise RuntimeError(
                        "Depth Anything 3 is missing the PhyEdit training patch. "
                        "Run scripts/setup_depth_anything_3.sh before enabling depth supervision."
                    )
                src_depth = depth_prediction["depth"][:, :1].detach()
                target_depth = depth_prediction["depth"][:, 1:2].detach()
                valid_mask = depth_prediction["non_sky_mask"][:, 1:2].detach().clone().bool()
                del depth_prediction
                del src_target_images_for_depth
                del target_images_preprocessed
                del target_images_denormed
                del src_images_tensor
                del src_images_denormed

            decoded_pred_images = decode_images(self.pipe, res, height, width, output_type="pt")
            pred_images_preprocessed = preprocess_images_for_da3_model(
                self.depth_model, decoded_pred_images, process_res=max(height, width), batch_type="batch"
            )
            src_pred_images_for_depth = torch.cat([src_images_preprocessed, pred_images_preprocessed], dim=1)

            with torch.autocast(device_type=self.device.type, dtype=self.target_dtype):
                prediction = self.depth_model.model(src_pred_images_for_depth)
            pred_depth = prediction["depth"][:, 1:2]
            del prediction
            del src_pred_images_for_depth
            del src_images_preprocessed
            del pred_images_preprocessed
            del decoded_pred_images

            eps = 1e-6
            with torch.no_grad():
                depth_h, depth_w = pred_depth.shape[-2:]
                depth_weight_map = preprocess_mask_to_weight_map(
                    f1_masks_list=f1_masks,
                    f2_masks_list=f2_masks,
                    latent_shape=(depth_h, depth_w),
                    expansion_px=48,
                    bg_importance=0.5,
                ).to(device=pred_depth.device)

            depth_diff_log = torch.log(target_depth.float().clamp_min(eps)) - torch.log(
                pred_depth.float().clamp_min(eps)
            )
            weighted_mask = (depth_weight_map * valid_mask).float()
            weighted_count = weighted_mask.sum().clamp_min(eps)

            weighted_mean = (weighted_mask * depth_diff_log).sum() / weighted_count
            weighted_sq_mean = (weighted_mask * depth_diff_log.pow(2)).sum() / weighted_count
            silog_term = (weighted_sq_mean - self.silog_loss_lambda * weighted_mean.pow(2)).clamp_min(0.0)
            depth_loss = torch.sqrt(silog_term + eps)

            loss = loss + self.depth_loss_lambda * depth_loss
            self.accumulate_log_metric("silog_loss", depth_loss.item())

        self.accumulate_log_metric("loss", loss.item())

        if self.process_visualize and self.debug_dir and _is_rank_zero():
            with torch.no_grad():
                debug_image_save_dir = os.path.join(self.debug_dir, "train_debug_images")
                os.makedirs(debug_image_save_dir, exist_ok=True)

                f1_coords = batch["f1_coords"]
                f2_coords = batch["f2_coords"]
                f1_depth_ranges = batch["f1_depth_ranges"]
                f2_depth_ranges = batch["f2_depth_ranges"]

                res = (x_t - noise_pred * t_).to(self.target_dtype)
                src_images = decode_images(self.pipe, image_latents[:, : latents.size(1)], height, width)
                if self.need_moved_image:
                    moved_images = decode_images(self.pipe, image_latents[:, latents.size(1) :], height, width)
                res_images = decode_images(self.pipe, res, height, width)
                gt_images = decode_images(self.pipe, x_0, height, width)
                gt_noised_images = decode_images(self.pipe, x_t, height, width)

                mask_weight_map_np = mask_weight_map.cpu().numpy()
                if self.depth_supervise:
                    target_depth_np = target_depth.float().cpu().numpy()
                    pred_depth_np = pred_depth.float().cpu().numpy()
                    src_depth_np = src_depth.float().cpu().numpy()
                    depth_weight_map_np = depth_weight_map.float().cpu().numpy()
                    depth_effective_weight_map_np = weighted_mask.float().cpu().numpy()
                    valid_mask_np = valid_mask.float().cpu().numpy()

                for i in range(len(res_images)):
                    annotate_image_with_coordinates(src_images[i], f1_coords[i], f1_depth_ranges[i]).save(
                        os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_cond_{i}.png")
                    )
                    annotate_image_with_coordinates(gt_images[i], f2_coords[i], f2_depth_ranges[i]).save(
                        os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_gt_{i}.png")
                    )
                    if self.need_moved_image:
                        annotate_image_with_coordinates(moved_images[i], f2_coords[i], f2_depth_ranges[i]).save(
                            os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_moved_{i}.png")
                        )
                    res_images[i].save(os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_res_{i}.png"))
                    gt_noised_images[i].save(
                        os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_gt_noised_{i}.png")
                    )

                    visualize_mask_weight_map(
                        mask_weight_map_np[i][0],
                        os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_mask_weight_map_{i}.png"),
                    )

                    if self.depth_supervise:
                        target_depth_map = target_depth_np[i][0]
                        pred_depth_map = pred_depth_np[i][0]
                        src_depth_map = src_depth_np[i][0]

                        Image.fromarray(visualize_depth(target_depth_map)).save(
                            os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_target_depth_{i}.png")
                        )
                        Image.fromarray(visualize_depth(pred_depth_map)).save(
                            os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_pred_depth_{i}.png")
                        )
                        Image.fromarray(visualize_depth(src_depth_map)).save(
                            os.path.join(debug_image_save_dir, f"step_{self.total_steps:05d}_src_depth_{i}.png")
                        )

                        depth_weight_map_i = depth_weight_map_np[i][0]
                        depth_effective_weight_map_i = depth_effective_weight_map_np[i][0]

                        visualize_mask_weight_map(
                            depth_weight_map_i,
                            os.path.join(
                                debug_image_save_dir, f"step_{self.total_steps:05d}_depth_loss_weight_map_{i}.png"
                            ),
                        )
                        visualize_mask_weight_map(
                            depth_effective_weight_map_i,
                            os.path.join(
                                debug_image_save_dir,
                                f"step_{self.total_steps:05d}_depth_loss_weight_map_valid_{i}.png",
                            ),
                        )
                        visualize_mask_weight_map(
                            valid_mask_np[i][0],
                            os.path.join(
                                debug_image_save_dir, f"step_{self.total_steps:05d}_depth_loss_valid_mask_{i}.png"
                            ),
                        )

        self.last_t = t.mean().item()
        self.accumulate_log_metric("t", self.last_t)
        return loss
