# 📏 ManipEval Evaluation

ManipEval is exposed through one public entry point while keeping DeQA in a
separate legacy environment. The stages run in this order:

1. `core`: VLM grounding, SAM masks, depth/3D metrics, and DINO features.
2. `motion`: relocation-aware motion penalty used by RA-DINO.
3. `deqa`: perceptual quality scoring in an isolated Python environment.
4. `phys-vlm`: physical-plausibility scoring through an OpenAI-compatible VLM.

## 📊 Release Checkpoint Results

The public checkpoint was further trained at a higher image resolution for the
open-source release, using an aspect-ratio-preserving base area of `589824`
(`768^2`). This release-stage scaling places additional emphasis on depth
accuracy and 3D-aware object manipulation. It is not the exact checkpoint used
for the paper tables, so its metric profile may differ slightly from the
reported paper values.

The following results use the public 200-item ManipEval split with eight seeds
per item. All metrics are linearly normalized to `[0, 100]`.

| DIoU (up) | Mask IoU (up) | AbsRel (down) | delta_1.25 (up) | Chamfer (down) | Centroid (down) | RA-DINO (up) | DeQA (up) | Phys-VLM (up) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 63.06 | 23.17 | 48.14 | 50.99 | 21.48 | 33.21 | 34.08 | 76.49 | 91.66 |

## 📦 Installation

Install the main evaluation dependencies in the PhyEdit environment:

```bash
pip install -r bench/requirements.txt
```

Create a separate environment for DeQA, install a CUDA-compatible PyTorch build,
then install its legacy dependencies with:

```bash
/path/to/deqa-env/bin/python -m pip install -r bench/requirements-deqa.txt
```

DeQA additionally uses `flash-attn`, installed with `--no-build-isolation`
after PyTorch.

## 🖼️ Sampling

Use the inference quick start in the repository [README](../README.md)
to generate the standard `####_seedN.png` files.

## ▶️ Evaluation

Run all metric stages on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_API_KEY=your-key \
python bench/evaluation/evaluate.py \
  --benchmark-metadata data/RealManip-40K/metadata/test.json \
  --image-dir outputs/manipeval \
  --output-dir outputs/manipeval \
  --device cuda:0 \
  --deqa-python /path/to/deqa-env/bin/python \
  --deqa-model-path zhiyuanyou/DeQA-Score-Mix3 \
  --depth-model-path depth-anything/da3nested-giant-large \
  --sam-model-path facebook/sam3 \
  --dino-model-path facebook/dinov3-vitl16-pretrain-lvd1689m \
  --vlm-base-url https://your-openai-compatible-endpoint/v1 \
  --vlm-model-name your-vlm-model
```

## ♻️ Resume Behavior

The main process never imports DeQA. It launches `score_deqa.py` through
`--deqa-python` and exchanges JSON files with that process. Use `--resume` to
reuse completed outputs and retry unfinished API calls, or `--dry-run` to print
the exact subprocess commands without loading models.

The final result is written to `bench_score_vlm_logic.json`. Intermediate files,
grounding caches, debug images, and a stage manifest remain in the same output
directory so interrupted evaluations can continue without recomputing every
stage.

## 🧭 Implementation Map

- `sample.py`: multi-GPU benchmark image generation.
- `vllm_depth_metrics.py`: grounding, masks, depth/3D metrics, and DINO scoring.
- `backfill_motion_penalty.py`: relocation-aware motion penalty for RA-DINO.
- `evaluation/score_deqa.py`: DeQA scoring in the isolated legacy environment.
- `backfill_vlm_logic_consistency.py`: physical-plausibility VLM scoring.
- `evaluation/`: stage orchestration, resume manifests, and subprocess isolation.
- `tools/`, `utils/`, and `vlm/`: shared model wrappers and metric utilities.

The stage scripts are intentionally separate because DeQA requires an
incompatible dependency environment and the other stages support independent
resume and recovery.
