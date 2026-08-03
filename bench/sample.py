import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from bench.utils.benchmark_metadata import load_benchmark_metadata  # noqa: E402
from phyedit.data.dataset import resolve_qwen_edit_size  # noqa: E402
from phyedit.model_deepspeed.generate import generate  # noqa: E402
from phyedit.model_deepspeed.model import load_pipeline_from_config  # noqa: E402

PHYEDIT_RELEASE_BASE_AREA = 589824


def _load_bench_indices(path: str | os.PathLike | None) -> set[int] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_indices = payload.get("bench_indices") if isinstance(payload, dict) else payload
    if not isinstance(raw_indices, list):
        raise ValueError(f"Invalid bench indices file: {path}")
    return {int(idx) for idx in raw_indices}


def _resolve_base_area(base_area: int | None = None, base_size: int | None = None) -> int:
    if base_area is not None:
        return int(base_area)
    if base_size is not None:
        return int(base_size) * int(base_size)
    return PHYEDIT_RELEASE_BASE_AREA


def _item_source_size(item) -> tuple[int, int]:
    if item.get("resolution"):
        return tuple(item["resolution"])
    with Image.open(item["f1_path"]) as image:
        return image.size


def _group_items_by_target_size(batch_items, height, width, base_area, longer_side):
    grouped_items = {}
    for item in batch_items:
        target_size = resolve_qwen_edit_size(
            _item_source_size(item),
            base_area=base_area if longer_side is None else None,
            longer_side=longer_side,
            height=height,
            width=width,
        )
        grouped_items.setdefault(target_size, []).append(item)
    return grouped_items.items()


def sample_bench(
    bench_items,
    output_dir,
    batch_size,
    seeds,
    gpu_id=0,
    config_path: str | os.PathLike | None = None,
    checkpoint_path: str | os.PathLike | None = None,
    pretrained_model_path: str | os.PathLike | None = None,
    height: int | None = None,
    width: int | None = None,
    base_area: int | None = None,
    base_size: int | None = None,
    longer_side: int | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline_from_config(
        str(config_path or PROJECT_ROOT / "configs" / "train_deepspeed.yaml"),
        device=device,
        checkpoint_path=checkpoint_path,
        pretrained_model_path=pretrained_model_path,
    )
    pipe.set_progress_bar_config(disable=True)
    base_area = _resolve_base_area(base_area=base_area, base_size=base_size)

    for seed in seeds:
        for i in tqdm(range(0, len(bench_items), batch_size), desc=f"GPU {gpu_id} Seed {seed}"):
            batch_items = bench_items[i : i + batch_size]
            for (target_w, target_h), target_items in _group_items_by_target_size(
                batch_items,
                height=height,
                width=width,
                base_area=base_area,
                longer_side=longer_side,
            ):
                src_images = [item["f1_path"] for item in target_items]
                mask_images = [item["f1_mask_path"] for item in target_items]
                depth_images = [item["f1_depth_path"] for item in target_items]
                intrinsics = [item["f1_intrinsic_path"] for item in target_items]
                extrinsics = [item["f1_extrinsic_path"] for item in target_items]
                src_obj_coords = [item["f1_coords"] for item in target_items]
                target_obj_coords = [item["f2_coords"] for item in target_items]
                image_depth_ranges = [item["f1_depth_range"] for item in target_items]
                objects = [item["object_name"] for item in target_items]
                local_seeds = [seed] * len(target_items)

                generated_images = generate(
                    pipe,
                    src_images,
                    mask_images,
                    depth_images,
                    intrinsics,
                    extrinsics,
                    src_obj_coords,
                    target_obj_coords,
                    image_depth_ranges,
                    objects,
                    obj_edit_prompt=None,
                    additional_prompt=None,
                    height=target_h,
                    width=target_w,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=local_seeds,
                    device=device,
                )

                for item, gen_img in zip(target_items, generated_images):
                    bench_index = item["bench_index"]
                    gen_img.save(os.path.join(output_dir, f"{bench_index:04d}_seed{seed}.png"))


def parallel_sample_bench(
    bench_items,
    output_dir,
    batch_size,
    seeds,
    config_path: str | os.PathLike | None = None,
    checkpoint_path: str | os.PathLike | None = None,
    pretrained_model_path: str | os.PathLike | None = None,
    height: int | None = None,
    width: int | None = None,
    base_area: int | None = None,
    base_size: int | None = None,
    longer_side: int | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
):
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        print(f"Using single process (num_gpus={num_gpus}).")
        sample_bench(
            bench_items,
            output_dir,
            batch_size,
            seeds,
            gpu_id=0,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            pretrained_model_path=pretrained_model_path,
            height=height,
            width=width,
            base_area=base_area,
            base_size=base_size,
            longer_side=longer_side,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        return

    print(f"Found {num_gpus} GPUs. Starting parallel sampling...")
    torch.multiprocessing.set_start_method("spawn", force=True)

    gpu_items = [[] for _ in range(num_gpus)]
    for idx, item in enumerate(bench_items):
        gpu_items[idx % num_gpus].append(item)

    processes = []
    for gpu_id in range(num_gpus):
        p = torch.multiprocessing.Process(
            target=sample_bench,
            args=(gpu_items[gpu_id], output_dir, batch_size, seeds, gpu_id),
            kwargs={
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "pretrained_model_path": pretrained_model_path,
                "height": height,
                "width": width,
                "base_area": base_area,
                "base_size": base_size,
                "longer_side": longer_side,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed_processes = [
        f"gpu={gpu_id}, exit_code={process.exitcode}"
        for gpu_id, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if failed_processes:
        raise RuntimeError(
            "One or more sampling workers failed: " + "; ".join(failed_processes)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample benchmark images with a PhyEdit LoRA checkpoint.")
    parser.add_argument(
        "--config-path",
        default=str(PROJECT_ROOT / "configs" / "train_deepspeed.yaml"),
        help="Training config used to build the Qwen edit pipeline.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("PHYEDIT_CHECKPOINT_PATH", ""),
        help="LoRA safetensors path. May also be set with PHYEDIT_CHECKPOINT_PATH.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        default=os.environ.get("PHYEDIT_PRETRAINED_MODEL_PATH", ""),
        help=(
            "Qwen-Image-Edit model path or model id. Overrides pretrained_path in the config; "
            "may also be set with PHYEDIT_PRETRAINED_MODEL_PATH."
        ),
    )
    parser.add_argument(
        "--benchmark-metadata",
        required=True,
        help="Benchmark metadata JSON. Relative asset paths are resolved against this file.",
    )
    parser.add_argument(
        "--bench-indices-json",
        default="",
        help="Optional JSON containing bench_indices for sampling a benchmark subset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where generated PNGs are written.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47, 48, 49])
    parser.add_argument("--height", type=int, default=None, help="Explicit output height; must be set with --width.")
    parser.add_argument("--width", type=int, default=None, help="Explicit output width; must be set with --height.")
    parser.add_argument("--base-size", type=int, default=None, help="Optional square base size for dynamic area.")
    parser.add_argument(
        "--base-area",
        type=int,
        default=None,
        help="Dynamic target area. Defaults to 589824 and overrides --base-size when provided.",
    )
    parser.add_argument("--longer-side", type=int, default=None, help="Optional longer-side resize mode.")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    args = parser.parse_args()

    if not args.checkpoint_path:
        parser.error("--checkpoint-path or PHYEDIT_CHECKPOINT_PATH is required")

    bench_items = load_benchmark_metadata(args.benchmark_metadata)
    selected_indices = _load_bench_indices(args.bench_indices_json)
    if selected_indices is not None:
        bench_items = [item for item in bench_items if int(item["bench_index"]) in selected_indices]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Sampling {len(bench_items)} benchmark items into: {output_dir}")
    if args.checkpoint_path:
        print(f"Using explicit checkpoint: {args.checkpoint_path}")
    parallel_sample_bench(
        bench_items,
        output_dir,
        batch_size=args.batch_size,
        seeds=args.seeds,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path or None,
        pretrained_model_path=args.pretrained_model_path or None,
        height=args.height,
        width=args.width,
        base_area=args.base_area,
        base_size=args.base_size,
        longer_side=args.longer_side,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
