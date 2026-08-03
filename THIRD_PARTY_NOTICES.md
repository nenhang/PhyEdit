# Third-Party Components

PhyEdit integrates with several third-party projects and model weights. Their
licenses apply independently from the PhyEdit source-code license.

| Component | Use in this repository | License / terms |
| --- | --- | --- |
| Hugging Face Diffusers | Qwen-Image-Edit pipeline utilities adapted in `phyedit/pipeline/` | Apache License 2.0 |
| Qwen-Image-Edit-2511 | Base image-editing model, downloaded separately | Terms published with the model repository |
| Depth Anything 3 | 3D preview generation, depth supervision, and ManipEval geometry metrics; installed at the pinned revision with `patches/depth_anything_3_phyedit.patch` | Source-code license and model-specific terms published by the project |
| DA3NESTED-GIANT-LARGE | Default depth model | CC BY-NC 4.0 at the time of this release; non-commercial use only |
| SAM 3 | Object localization masks in ManipEval | Terms published with the model repository |
| DINOv3 | Object feature similarity in ManipEval | Terms published with the model repository |
| DeQA-Score-Mix3 | Image-quality score in ManipEval | Terms published with the model repository |

No third-party model weights are redistributed in this repository. Users are
responsible for reviewing and complying with the current terms of every model
and dataset they download.
