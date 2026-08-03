import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw, ImageFont


def annotate_image_with_coordinates(image, coordinates, depth_range=None):
    image_clone = image.copy()
    draw = ImageDraw.Draw(image_clone)
    font = ImageFont.load_default(size=16)

    image_w, image_h = image_clone.size
    if len(coordinates) == 3 and all(isinstance(c, (int, float)) for c in coordinates):
        coordinates = [coordinates]
    for idx, coord in enumerate(coordinates):
        x, y, z = coord[0], coord[1], coord[2]
        label = f"({x:.2f}, {y:.2f}, {z:.2f})"
        start_x = int((x + 1) / 2 * image_w)
        start_y = int((y + 1) / 2 * image_h)
        radius = 3
        draw.ellipse(
            [(start_x - radius, start_y - radius), (start_x + radius, start_y + radius)],
            fill="red",
        )
        draw.text((start_x + 5, start_y - 5), label, fill="red", font=font)

    if depth_range is not None:
        depth_label = f"Depth Range: {depth_range[0]:.2f} to {depth_range[1]:.2f}"
        draw.text((10, 10), depth_label, fill="blue", font=font)

    return image_clone


def visualize_depth_gray(depth: np.ndarray, percentiles=(0, 100), reverse_gray=False):
    vmin, vmax = np.percentile(depth, percentiles)
    depth_clipped = np.clip(depth, vmin, vmax)
    depth_normalized = (depth_clipped - vmin) / (vmax - vmin + 1e-8)
    if reverse_gray:
        depth_normalized = 1.0 - depth_normalized
    depth_gray = (depth_normalized * 255).astype(np.uint8)
    return depth_gray


def visualize_mask_weight_map(mask_weight_map, save_path):
    plt.imshow(mask_weight_map, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


def add_mask(
    image: np.ndarray,
    mask: np.ndarray,
    mask_color: tuple = (255, 0, 0),
    bg_color: tuple | None = None,
    mask_alpha: float = 0.5,
):
    """Overlay a binary mask on a NumPy image."""
    image = image.copy()
    mask = mask.astype(bool)

    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[mask] = mask_color

    blended_image = image.copy()
    blended_image[mask] = (
        ((1 - mask_alpha) * image[mask] + mask_alpha * color_mask[mask]).astype(np.uint8)
        if bg_color is None
        else ((1 - mask_alpha) * np.array(bg_color) + mask_alpha * np.array(mask_color)).astype(np.uint8)
    )

    return blended_image


def mask_moved_image(
    images_torch,
    obj_masks,
    bg_patch_masks,
    bg_patch_color=[1, 1, 1],
    dim_factor=None,
    bg_color=None,
    desaturate=False,
):
    B, C, H, W = images_torch.shape
    highlighted_images = images_torch.clone()

    for i in range(B):
        obj_mask = obj_masks[i]
        bg_patch_mask = bg_patch_masks[i]
        if not obj_mask.any() and not bg_patch_mask.any():
            continue

        img = images_torch[i]

        # Prepare the background/reference region.
        if bg_color:
            bg_processed = torch.zeros_like(img)
            for c in range(3):
                bg_processed[c][~obj_mask] = bg_color[c]
        else:
            if desaturate:
                bg = (img[0] * 0.299 + img[1] * 0.587 + img[2] * 0.114).unsqueeze(0).repeat(3, 1, 1)
            else:
                bg = img.clone()
            if dim_factor is not None:
                bg_processed = bg * dim_factor
            else:
                bg_processed = bg.clone()

        # Composite the moved object and source-region patch.
        res = bg_processed.clone()
        for c in range(3):
            res[c][bg_patch_mask] = bg_patch_color[c]
        for c in range(3):
            res[c][obj_mask] = img[c][obj_mask]

        highlighted_images[i] = res

    return highlighted_images
