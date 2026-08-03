# Repository Guide

This repository contains the public PhyEdit research implementation.

## Python Implementation

- `phyedit/data`: RealManip-40K metadata loading and aspect-ratio batching.
- `phyedit/model_deepspeed`: DeepSpeed training, checkpoint resume, and model wrappers.
- `phyedit/pipeline`: Qwen-Image-Edit pipeline adaptations.
- `phyedit/utils`: geometry, prompt, image, and Depth Anything 3 helpers.
- `bench`: ManipEval metric engines and the unified evaluation entry point.
- `gui`: interactive DA3 + SAM geometry editor.
- `configs`: public training configuration.
- `scripts`: preprocessing helpers.
- `patches`: focused third-party compatibility patches required for training.

Run Python checks with:

```bash
ruff check phyedit bench gui scripts
```

The main and DeQA environments intentionally use different Transformers
versions. Do not merge the DeQA model import into the main evaluation process.

## Release Boundaries

Model weights are distributed separately through Hugging Face. Do not add local
checkpoints, datasets, generated outputs, API keys, `.env` files, or
machine-specific absolute paths to the repository.
