from copy import deepcopy

import numpy as np
import torch
from PIL import Image

from ..data.dataset import QWEN_IMAGE_EDIT_BASE_AREA, resolve_qwen_edit_size
from ..pipeline.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline__call__
from ..utils.geometry_utils import translate_objects_3d_batch
from ..utils.image_process import mask_moved_image
from ..utils.text_process import get_edit_prompt, get_edit_prompt_coord_only


@torch.no_grad()
def render_moved_image_previews(
    src_images: list[Image.Image],
    mask_images: list[list[Image.Image]],
    depth_images: list[np.ndarray | str],
    intrinsics: list[np.ndarray | str],
    extrinsics: list[np.ndarray | str],
    target_obj_coords: list[list[list | np.ndarray]],
    device: torch.device | str,
) -> list[Image.Image]:
    """Render the geometric preview consumed as Picture 2 by Qwen Image Edit."""
    moved_images, _, bg_patch_masks, _, obj_masks = translate_objects_3d_batch(
        images=src_images,
        masks=mask_images,
        target_coords=deepcopy(target_obj_coords),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depths=depth_images,
        device=device,
    )
    moved_images = mask_moved_image(
        images_torch=moved_images,
        obj_masks=obj_masks,
        bg_patch_masks=bg_patch_masks,
    )

    previews = []
    for moved_image in moved_images:
        moved_image_np = (moved_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(
            np.uint8
        )
        previews.append(Image.fromarray(moved_image_np))
    return previews


@torch.no_grad()
def generate(
    pipeline,
    src_image: list[Image.Image | str],
    mask_image: list[list[Image.Image | str]],
    depth_image: list[np.ndarray | str],
    intrinsics: list[np.ndarray | str],
    extrinsics: list[np.ndarray | str],
    src_obj_coords: list[list[list | np.ndarray]],
    target_obj_coords: list[list[list | np.ndarray]],
    image_depth_range: list,
    objects: list[list[str]],
    obj_edit_prompt: list[list[str | None]] | None = None,
    additional_prompt: list[str] | None = None,
    prompt_override: list[str] | None = None,
    prompt_xy_coord_range: str = "neg1_1",
    need_moved_image: bool = True,
    moved_image_input: list[Image.Image | str] | None = None,
    height: int | None = None,
    width: int | None = None,
    base_area: int | None = QWEN_IMAGE_EDIT_BASE_AREA,
    longer_side: int | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int | list | None = None,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
):
    if not src_image:
        raise ValueError("src_image must not be empty")

    target_size = None
    for i in range(len(src_image)):
        if isinstance(src_image[i], str):
            src_image[i] = Image.open(src_image[i]).convert("RGB")
        else:
            src_image[i] = src_image[i].convert("RGB")

        resolved_size = resolve_qwen_edit_size(
            src_image[i].size,
            base_area=base_area if longer_side is None else None,
            longer_side=longer_side,
            height=height,
            width=width,
        )
        if target_size is None:
            target_size = resolved_size
        elif resolved_size != target_size:
            raise ValueError(
                "All images in one generate() batch must resolve to the same size. "
                "Pass explicit height/width or group samples by aspect ratio."
            )

    width, height = target_size
    for i in range(len(src_image)):
        src_image[i] = src_image[i].resize((width, height), resample=Image.BICUBIC)

    for i in range(len(mask_image)):
        for j in range(len(mask_image[i])):
            if isinstance(mask_image[i][j], str):
                mask_image[i][j] = Image.open(mask_image[i][j]).convert("L")
            else:
                mask_image[i][j] = mask_image[i][j].convert("L")
            mask_image[i][j] = mask_image[i][j].resize(
                (width, height), resample=Image.NEAREST
            )

    normalized_coordinates = []
    for i in range(len(src_obj_coords)):
        depth_range_ = [float(image_depth_range[i][0]), float(image_depth_range[i][1])]
        single_image_coords = []
        for j in range(len(src_obj_coords[i])):
            src_coord = deepcopy(src_obj_coords[i][j])
            tgt_coord = deepcopy(target_obj_coords[i][j])
            src_coord[2] = (src_coord[2] - depth_range_[0]) / (
                depth_range_[1] - depth_range_[0]
            )
            tgt_coord[2] = (tgt_coord[2] - depth_range_[0]) / (
                depth_range_[1] - depth_range_[0]
            )
            single_image_coords.append([src_coord, tgt_coord])
        normalized_coordinates.append(single_image_coords)

    prompts = []
    if prompt_override is not None:
        prompts = deepcopy(prompt_override)
    else:
        for i in range(len(objects)):
            prompts.append(
                get_edit_prompt(
                    object_name=objects[i],
                    coordinates=normalized_coordinates[i],
                    object_edit_prompt=obj_edit_prompt[i]
                    if obj_edit_prompt is not None
                    else None,
                    additional_prompt=additional_prompt[i]
                    if additional_prompt is not None
                    else None,
                    xy_coord_range=prompt_xy_coord_range,
                )
                if need_moved_image
                else get_edit_prompt_coord_only(
                    object_name=objects[i],
                    coordinates=normalized_coordinates[i],
                    object_edit_prompt=obj_edit_prompt[i]
                    if obj_edit_prompt is not None
                    else None,
                    additional_prompt=additional_prompt[i]
                    if additional_prompt is not None
                    else None,
                    xy_coord_range=prompt_xy_coord_range,
                )
            )

    if isinstance(seed, int):
        generator = torch.Generator(device).manual_seed(seed)
    elif isinstance(seed, list):
        generator = [torch.Generator(device).manual_seed(s) for s in seed]
    else:
        generator = None

    input_images = []
    if need_moved_image:
        if moved_image_input is None:
            moved_previews = render_moved_image_previews(
                src_images=src_image,
                mask_images=mask_image,
                depth_images=depth_image,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                target_obj_coords=target_obj_coords,
                device=device,
            )
            for i, moved_preview in enumerate(moved_previews):
                input_images.append([src_image[i], moved_preview])
        else:
            if len(moved_image_input) != len(src_image):
                raise ValueError("moved_image_input length must equal src_image length")
            for i in range(len(moved_image_input)):
                mv = moved_image_input[i]
                if isinstance(mv, str):
                    mv = Image.open(mv).convert("RGB")
                else:
                    mv = mv.convert("RGB")
                mv = mv.resize((width, height), resample=Image.BICUBIC)
                input_images.append([src_image[i], mv])
    else:
        for i in range(len(src_image)):
            input_images.append([src_image[i]])

    image = QwenImageEditPlusPipeline__call__(
        self=pipeline,
        prompt=prompts,
        image=input_images,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images

    return image
