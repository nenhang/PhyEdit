# Adapted from Hugging Face Diffusers' Qwen-Image-Edit image processing code.
# The upstream source and these modifications are licensed under Apache-2.0.

import math

import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import CONDITION_IMAGE_SIZE
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image


# Qwen image latents require spatial dimensions aligned to this multiple.
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 16) * 16
    height = round(height / 16) * 16

    return width, height


def get_image_size(image):
    if isinstance(image, list):
        image = image[0]
    if isinstance(image, torch.Tensor):
        _, _, h, w = image.shape
        return w, h
    elif isinstance(image, Image.Image):
        return image.size
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def preprocess_condition_images(self, image, vae_image_size=None, condition_image_size=None):
    # Each batch item is a list of conditioning images.
    assert isinstance(image, list), "Input image must be a list of PIL.Images."
    assert isinstance(image[0], list), "Each item in the image list must be a list of PIL.Images."
    if condition_image_size is None:
        condition_image_size = CONDITION_IMAGE_SIZE
    if vae_image_size is None:
        input_image_width, input_image_height = get_image_size(image[0][0])
        vae_image_size = input_image_width * input_image_height

    # Every item must expose the same conditioning-image layout for tensor stacking.
    batch_image_sizes = [get_image_size(img) for img in image[0]]
    for image_list in image[1:]:
        current_sizes = [get_image_size(img) for img in image_list]
        assert current_sizes == batch_image_sizes, (
            "All images in the batch must have the same dimensions to be processed together."
        )

    condition_image_sizes = [[] for _ in range(len(image))]
    condition_images = [[] for _ in range(len(image))]
    vae_image_sizes = [[] for _ in range(len(image))]
    vae_images = [[] for _ in range(len(image))]
    for i, image_list in enumerate(image):
        for img in image_list:
            image_width, image_height = img.size
            condition_width, condition_height = calculate_dimensions(condition_image_size, image_width / image_height)
            vae_width, vae_height = calculate_dimensions(vae_image_size, image_width / image_height)
            condition_image_sizes[i].append((condition_width, condition_height))
            vae_image_sizes[i].append((vae_width, vae_height))
            condition_images[i].append(self.image_processor.resize(img, condition_height, condition_width))
            vae_images[i].append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

    return vae_images, vae_image_sizes, condition_images, condition_image_sizes


def prepare_latents(
    self,
    batch_size,
    num_channels_latents,
    height,
    width,
    dtype,
    device,
    generator=None,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (self.vae_scale_factor * 2))
    width = 2 * (int(width) // (self.vae_scale_factor * 2))
    shape = (batch_size, 1, num_channels_latents, height, width)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

    return latents


def encode_images(self, images):
    if not isinstance(images, torch.Tensor):
        images = self.image_processor.preprocess(images).unsqueeze(2)
    else:
        if len(images.shape) == 4:  # BCHW
            images = images.unsqueeze(2)  # to BCFHW
    images = images.to(self.device).to(self.dtype)
    if hasattr(self, "transformer") and self.transformer is not None:
        num_channels_latents = self.transformer.config.in_channels // 4
    else:
        num_channels_latents = 64 // 4
    image_latents = self._encode_vae_image(image=images, generator=None)
    image_latent_height, image_latent_width = image_latents.shape[3:]
    images_latents = self._pack_latents(
        image_latents,
        batch_size=images.shape[0],
        num_channels_latents=num_channels_latents,
        height=image_latent_height,
        width=image_latent_width,
    )
    return images_latents


def decode_images(
    self, latents: torch.Tensor, height: int, width: int, output_type: str = "pil", max_batch_size: int | None = None
):
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = latents.to(self.vae.dtype)
    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    if max_batch_size is None:
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
    else:
        images = []
        for i in range(0, latents.shape[0], max_batch_size):
            batch_latents = latents[i : i + max_batch_size]
            batch_images = self.vae.decode(batch_latents, return_dict=False)[0][:, :, 0]
            images.append(batch_images)
        image = torch.cat(images, dim=0)
    image = self.image_processor.postprocess(image, output_type=output_type)
    return image
