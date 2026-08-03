# 🏋️ Training

PhyEdit fine-tunes Qwen-Image-Edit with LoRA and optional depth supervision.
The public configuration is [configs/train_deepspeed.yaml](../../configs/train_deepspeed.yaml).

## 🗃️ Data

RealManip-40K is distributed through Hugging Face with automatic gated access:

Request access on the dataset page and authenticate with `hf auth login` before
downloading the training split.

```bash
hf download ruihangxu/RealManip-40K \
  --repo-type dataset \
  --local-dir data/RealManip-40K

python data/RealManip-40K/scripts/extract_shards.py --split train
```

The default configuration reads `data/RealManip-40K/metadata/train.jsonl`.

## ⚙️ Configuration

Review these fields before launching:

- `train.batch_size` and `train.accumulate_grad_batches`
- `train.dataset.metadata_path` and `train.save_path`
- `train.image_base_area`
- `train.depth_supervise` and `train.depth_loss_lambda`
- `train.deepspeed.zero_stage` and offload settings
- `train.auto_resume` and `train.resume_world_size_policy`

The default base area is `589824` (`768^2`). This preserves aspect ratio and
produces `1024 x 576` for 16:9 inputs and `768 x 768` for square inputs.

## ▶️ Launch

```bash
deepspeed --num_gpus 2 \
  phyedit/model_deepspeed/train.py \
  --config configs/train_deepspeed.yaml
```

## ♻️ Checkpoints And Resume

Checkpoints contain LoRA weights and a data cursor. When
`save_deepspeed_state: true`, the latest checkpoint also retains native
DeepSpeed optimizer and RNG state.

Exact DeepSpeed resume requires the original world size. When GPU count changes,
`resume_world_size_policy: lora_only` restores LoRA weights and converts the data
cursor while rebuilding optimizer and per-rank RNG state. Set
`resume_world_size_policy: strict` when an exact native resume is required.

## 🧱 Preview Cache

Training can generate movement previews in the data loader. Precomputing them is
recommended for repeated epochs and is documented in
[scripts/README.md](../../scripts/README.md).
