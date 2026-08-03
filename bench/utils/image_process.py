from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_mask_overlay_debug(
    generated_images, generated_obj_masks, generated_obj_bboxes=None, output_dir=None, alpha=0.45
):
    if output_dir is None:
        raise ValueError("output_dir must be provided.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    color_palette = np.array(
        [
            [255, 80, 80],
            [80, 255, 120],
            [80, 160, 255],
            [255, 215, 80],
            [220, 120, 255],
            [80, 240, 240],
        ],
        dtype=np.float32,
    )

    if generated_obj_bboxes is None:
        generated_obj_bboxes = [None for _ in generated_images]

    for image_idx, (image, mask_list, bbox_list) in enumerate(
        zip(generated_images, generated_obj_masks, generated_obj_bboxes)
    ):
        base_image = np.array(image.convert("RGB"), dtype=np.float32)
        overlay_image = base_image.copy()

        if bbox_list is None:
            pairs = [(idx, obj_mask, None) for idx, obj_mask in enumerate(mask_list)]
        else:
            pairs = [(idx, obj_mask, obj_bbox) for idx, (obj_mask, obj_bbox) in enumerate(zip(mask_list, bbox_list))]

        for obj_idx, obj_mask, obj_bbox in pairs:
            if obj_mask is None:
                continue

            h, w = overlay_image.shape[:2]
            if torch.is_tensor(obj_mask):
                mask_tensor = obj_mask.detach().float().cpu()
            else:
                mask_tensor = torch.from_numpy(np.asarray(obj_mask)).float()

            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.shape != (h, w):
                mask_tensor = (
                    torch.nn.functional.interpolate(
                        mask_tensor.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            mask_binary = (mask_tensor > 0.5).numpy()
            if not np.any(mask_binary):
                continue

            color = color_palette[obj_idx % len(color_palette)]
            overlay_image[mask_binary] = overlay_image[mask_binary] * (1 - alpha) + color * alpha

            if obj_bbox is not None and len(obj_bbox) == 4:
                overlay_pil_tmp = Image.fromarray(np.clip(overlay_image, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(overlay_pil_tmp)
                draw.rectangle(obj_bbox, outline=tuple(color.astype(np.uint8).tolist()), width=2)
                overlay_image = np.array(overlay_pil_tmp, dtype=np.float32)

        overlay_pil = Image.fromarray(np.clip(overlay_image, 0, 255).astype(np.uint8))
        overlay_pil.save(output_path / f"{image_idx:04d}_mask_overlay.png")


def bbox_to_binary_mask(image_size, bbox, device):
    width, height = image_size
    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
    if bbox is None or len(bbox) != 4:
        return mask

    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return mask
    mask[y1:y2, x1:x2] = 1.0
    return mask


def get_square_crop_bbox(bbox, image_size):
    width, height = image_size
    if bbox is None or len(bbox) != 4:
        return [0, 0, width, height]

    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))

    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    side = max(bbox_w, bbox_h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    square_x1 = cx - side / 2.0
    square_y1 = cy - side / 2.0
    square_x2 = cx + side / 2.0
    square_y2 = cy + side / 2.0

    if square_x1 < 0:
        shift = -square_x1
        square_x1 += shift
        square_x2 += shift
    if square_y1 < 0:
        shift = -square_y1
        square_y1 += shift
        square_y2 += shift
    if square_x2 > width:
        shift = square_x2 - width
        square_x1 -= shift
        square_x2 -= shift
    if square_y2 > height:
        shift = square_y2 - height
        square_y1 -= shift
        square_y2 -= shift

    square_x1 = max(0.0, square_x1)
    square_y1 = max(0.0, square_y1)
    square_x2 = min(float(width), square_x2)
    square_y2 = min(float(height), square_y2)

    square_x1 = int(np.floor(square_x1))
    square_y1 = int(np.floor(square_y1))
    square_x2 = int(np.ceil(square_x2))
    square_y2 = int(np.ceil(square_y2))

    square_x1 = max(0, min(width - 1, square_x1))
    square_y1 = max(0, min(height - 1, square_y1))
    square_x2 = max(square_x1 + 1, min(width, square_x2))
    square_y2 = max(square_y1 + 1, min(height, square_y2))

    return [square_x1, square_y1, square_x2, square_y2]


def draw_src_target_boxes(
    image: Image.Image,
    orig_bboxes: list[list[float | int]] | None,
    target_bboxes: list[list[float | int]] | None,
) -> Image.Image:
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    w, h = annotated_image.size
    line_width = max(2, round(min(w, h) * 0.006))

    if orig_bboxes is not None:
        for box in orig_bboxes:
            if box is None or len(box) != 4:
                continue
            draw.rectangle(box, outline="red", width=line_width)

    if target_bboxes is not None:
        for box in target_bboxes:
            if box is None or len(box) != 4:
                continue
            draw.rectangle(box, outline="green", width=line_width)

    return annotated_image


def draw_detection_hint(
    image: Image.Image,
    target_bbox: list[float | int] | None,
) -> Image.Image:
    annotated_image = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated_image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    w, h = annotated_image.size
    line_width = max(2, round(min(w, h) * 0.006))

    if target_bbox is not None and len(target_bbox) == 4:
        tx1, ty1, tx2, ty2 = target_bbox
        draw_overlay.rectangle([tx1, ty1, tx2, ty2], fill=(0, 255, 0, 50), outline=(0, 255, 0, 180), width=line_width)

    merged = Image.alpha_composite(annotated_image, overlay)
    return merged.convert("RGB")


def draw_bbox(
    image: Image.Image,
    orig_bbox: list[float | int] | None,
) -> Image.Image:
    annotated_image = image.copy()
    if orig_bbox is None or len(orig_bbox) != 4:
        return annotated_image

    draw = ImageDraw.Draw(annotated_image)
    w, h = annotated_image.size
    line_width = max(2, round(min(w, h) * 0.006))
    draw.rectangle(orig_bbox, outline="red", width=line_width)
    return annotated_image
