# 🧰 Geometry Utilities

## 📐 Depth Anything 3 Setup

Install the tested Depth Anything 3 revision and apply the PhyEdit compatibility
patch:

```bash
bash scripts/setup_depth_anything_3.sh
```

The patch keeps nested-depth calibration outside autograd, exposes the non-sky
mask used by the loss, and preserves gradients through predicted depth. External
DA3 weights are not included; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## 🧱 Preview Cache

Precompute movement previews at the public `768^2` base area:

```bash
python scripts/precompute_moved_image_cache.py \
  --metadata data/RealManip-40K/metadata/train.jsonl \
  --cache-root cache/moved_images_768 \
  --metadata-out cache/moved_images_768/metadata/train.jsonl \
  --format png \
  --workers 1 \
  --device cuda:0 \
  --transform-base-area 589824
```

Set `train.dataset.metadata_path` to the generated metadata file. For CPU
preprocessing, use `--device cpu` with multiple workers. Omit
`--transform-base-area` to run the transform at each source image's original
resolution.
