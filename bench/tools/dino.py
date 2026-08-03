import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2.functional import InterpolationMode, resize
from transformers import AutoModel, AutoProcessor

from bench.utils.image_process import get_square_crop_bbox


class DINOModel:
    def __init__(
        self,
        model_path=None,
        device: torch.device | str = torch.device("cuda"),
    ):
        if model_path is None:
            model_path = os.getenv("DINO_MODEL_PATH", "facebook/dinov3-vitl16-pretrain-lvd1689m")
        print(f"Loading DINO model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device=device)
        self.model.eval()
        self.device = device

    def extract_masked_features(
        self,
        images: list[Image.Image],
        masks: list[list[Image.Image | np.ndarray]],
        bboxes: list[list[list[int]]],
        batch_size: int | None = None,
    ) -> list[torch.Tensor]:
        """Extract DINO features from masked object regions."""
        # crop images and masks according to bboxes
        cropped_images = []
        cropped_masks = []

        image_indices_in_batch = [[] for _ in range(len(images))]
        for i, (img, single_image_masks_list, single_image_bboxes) in enumerate(zip(images, masks, bboxes)):
            # Flatten all object crops into one model batch and remember ownership.
            for j, (mask, bbox) in enumerate(zip(single_image_masks_list, single_image_bboxes)):
                square_bbox = get_square_crop_bbox(bbox=bbox, image_size=img.size)
                x1, y1, x2, y2 = square_bbox
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_images.append(cropped_img)

                if isinstance(mask, Image.Image):
                    cropped_mask = mask.crop((x1, y1, x2, y2))
                    cropped_mask = np.array(cropped_mask)
                else:
                    cropped_mask = mask[y1:y2, x1:x2]
                cropped_masks.append(cropped_mask)
                image_indices_in_batch[i].append(len(cropped_images) - 1)

        obj_images = cropped_images
        obj_masks = cropped_masks

        if len(obj_images) == 0:
            return [
                torch.empty((0, self.model.config.hidden_size * 2), device=self.device) for _ in image_indices_in_batch
            ]

        if batch_size is None or batch_size <= 0:
            batch_size = len(obj_images)

        all_patch_feats_list = []
        all_cls_feats_list = []
        for start_idx in range(0, len(obj_images), batch_size):
            image_chunk = obj_images[start_idx : start_idx + batch_size]
            inputs = self.processor(images=image_chunk, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            chunk_cls_feats = outputs.last_hidden_state[:, 0, :]
            chunk_patch_feats = outputs.last_hidden_state[:, 1 + self.model.config.num_register_tokens :, :]
            all_cls_feats_list.append(chunk_cls_feats)
            all_patch_feats_list.append(chunk_patch_feats)

        all_cls_feats = torch.cat(all_cls_feats_list, dim=0)
        all_patch_feats = torch.cat(all_patch_feats_list, dim=0)

        # convert masks to tensor
        mask_tensors = []
        for mask in obj_masks:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask_tensor = mask.to(self.device, dtype=torch.float32)  # [H, W], 0/1
            mask_tensor = resize(
                mask_tensor.unsqueeze(0),
                size=(self.model.config.image_size, self.model.config.image_size),
                interpolation=InterpolationMode.BILINEAR,
            )
            mask_tensor = torch.nn.functional.avg_pool2d(
                mask_tensor, kernel_size=self.model.config.patch_size, stride=self.model.config.patch_size
            ).flatten()

            mask_tensors.append(mask_tensor)
        mask_tensors = torch.stack(mask_tensors, dim=0)

        obj_features = []
        for j in range(all_patch_feats.shape[0]):
            feat = all_patch_feats[j]  # [196, C]
            weight = mask_tensors[j].unsqueeze(-1)  # [196, 1]
            cls_feat = all_cls_feats[j]  # [C]

            # Mask-weighted pooling suppresses background inside the crop.
            sum_feat = (feat * weight).sum(0)
            masked_feat = sum_feat / (weight.sum() + 1e-6)
            masked_feat = torch.nn.functional.normalize(masked_feat, dim=-1)
            cls_feat = torch.nn.functional.normalize(cls_feat, dim=-1)

            joint_feat = torch.cat([masked_feat, cls_feat], dim=-1)
            obj_features.append(joint_feat.unsqueeze(0))

        obj_features = torch.cat(obj_features, dim=0)

        # Regroup object features by source image.
        grouped_obj_features = []
        for indices in image_indices_in_batch:
            grouped_obj_features.append(obj_features[indices])

        return grouped_obj_features
