import os

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


class SAMModel:
    def __init__(
        self,
        model_path=None,
        device: torch.device | str = torch.device("cuda"),
    ):
        if model_path is None:
            model_path = os.getenv("SAM_MODEL_PATH", "facebook/sam3")
        print(f"Loading SAM model from {model_path}...")
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path).to(device=device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def get_masks_by_bboxes(
        self,
        images: list[Image.Image | str],
        texts: list[str],
        bboxes: list[list[int]],
        batch_size: int | None = None,
    ):
        if batch_size is None or batch_size <= 0:
            batch_size = len(images)

        if len(images) == 0:
            return []

        all_segment_info = []
        for start_idx in range(0, len(images), batch_size):
            image_chunk = images[start_idx : start_idx + batch_size]
            text_chunk = texts[start_idx : start_idx + batch_size]
            bbox_chunk = bboxes[start_idx : start_idx + batch_size]
            inputs = self.processor(
                images=image_chunk, text=text_chunk, input_boxes=bbox_chunk, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs, multimask_output=False)
            chunk_masks = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=inputs.get("original_sizes").tolist()
            )
            all_segment_info.extend(chunk_masks)

        return all_segment_info
