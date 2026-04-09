# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is an **early-stage academic repository** for the PhyEdit paper (arXiv: 2604.07230). As of April 2026, **no implementation code has been released**. The following are still forthcoming per the to-do list:
- Inference and Training Code
- Dataset and Benchmark (RealManip-10K, ManipEval)
- Model Weights
- GUI Support

## Project Overview

**PhyEdit** is a framework for physically-grounded image editing that enables 3D-aware object relocation and manipulation in images. Key concepts:
- **Geometric constraints** ensure physical plausibility (shadows, occlusion, perspective)
- **Coordinate + prompt-based** object manipulation in 3D space
- **RealManip-10K**: Real-world dataset for physically-grounded object manipulation
- **ManipEval**: Benchmark for evaluating geometric accuracy of edits

The paper comes from ReLER Lab, CCAI, Zhejiang University (Ruihang Xu, Dewei Zhou, Xiaolong Shen, Fan Ma, Yi Yang).

## Expected Architecture (when code is released)

Based on the research domain (physically-grounded image editing), the codebase will likely involve:
- **Depth estimation**: 3D lifting of 2D images (the parent directory contains Depth-Anything-V2/V3)
- **Object segmentation**: Isolating objects for manipulation (SAM-based, per the parent directory)
- **Diffusion-based inpainting**: Regenerating edited regions with physical consistency
- **3D geometry**: Coordinate transforms, perspective projection, shadow/occlusion reasoning
- **Training pipeline**: Likely PyTorch-based, possibly building on existing diffusion model frameworks

The parent directory `/root/autodl-tmp/video-dataset-process/` contains related infrastructure including depth estimation models and segmentation tools that PhyEdit may depend on.
