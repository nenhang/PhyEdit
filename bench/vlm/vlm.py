import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from openai import OpenAI, omit
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

from bench.utils.image_process import (
    draw_bbox,
    draw_detection_hint,
    draw_src_target_boxes,
)

SCORING_PROMPT_TEMPLATE = """
### Task Description

Here are three images:
- Image 1: Original image with RED bounding box indicating the object(s) to be edited.
- Image 2: Edited image by applying the moving operation on the object(s) in Image 1.
- Image 3: Annotated Image 2 with RED bounding box indicating the source location(s) of the object(s) and GREEN bounding box indicating the target location(s).
The moving operation was: "{edit_prompt}".
Please evaluate these images based on the following criteria.

### Evaluation Criteria (Score: 1, 2, 3, 4, 5)

1. Aesthetic_Logic_Consistency (Weight: Physical Realism + Semantic Harmony)
- Evaluate if the overall image looks natural and logically sound after editing.
- Are the proportions, perspective, and lighting of the edited object consistent with the rest of the scene?
- 1: Severe artifacts, broken structures, or blurry textures that make the image look manipulated and unrealistic.
- 3: Mostly harmonious, but minor issues like inconsistent lighting directions or slight perspective mismatch.
- 5: Completely natural; the edit is indistinguishable from a real photo in terms of physical logic and harmony.

2. Source_Inpainting (Weight: Cleanup Quality)
- Inspect the original spot where the object was removed.
- 1: Messy artifacts, holes, or severe warping. Or the object to be moved is still clearly visible and not removed at all.
- 3: Filled, but textures are blurry, repetitive, or have visible seams.
- 5: Seamless; original spot is indistinguishable from its surroundings.

3. Target_Integration (Contextual Awareness & Local Harmony)
- Evaluate how well the object integrates into its **new location**.
- **Relational Adjustment**: Did the model adjust the surrounding context (e.g., local shadows, reflections on nearby surfaces, or alignment with nearby objects) to match the new position?
- 1: Object is severely deformed, broken, or failed to move to the target location as instructed.
- 3: Object is intact and placed correctly but looks like a "flat sticker" or naive copy-paste; lacks realistic contact shadows, local lighting harmony, or matched reflections.
- 5: Perfect integration; the object naturally interacts with its new local environment through realistic shadows, reflections, and seamless edge blending.

4. Background_Preservation (Non-edited Area Fidelity)
- Check if the areas that were completely unrelated to the edit (e.g., distant background, unrelated objects) remain consistent.
- 1: Global color shift, or unrelated objects/regions are significantly altered or distorted.
- 3: Minor changes in distant background or slight texture loss in non-target areas.
- 5: Perfect preservation; every pixel outside the edit/removal area is identical to the original image.

### Note
When you're judging the scores, you can pay less attention to the precision of the move (i.e., whether the object is moved to the exact target location), but be strict on the quality of the edit (i.e., whether the moved object looks natural in the new location).
When judging Source_Inpainting and Target_Integration, use Image 3 as spatial reference for the source and target locations, but evaluate the quality based on Image 2.
Please provide a brief explanation for the scores, especially if there are any low scores (1 or 2) in any criteria, to help understand the main issues in the edit.
Please be strict and objective in your evaluation. A score of 5 should only be given when the edit is flawless. If there are any imperfections, a score of 4 already indicates a very good result.

### Output Format
Respond in the following JSON format:
{{
    "consistency_score": int,
    "inpainting_score": int,
    "integration_score": int,
    "preservation_score": int,
    "explanation": str
}}
"""


SIMPLE_SCORING_PROMPT_TEMPLATE = """
### Task
You are given three images for evaluating one object-moving edit:
- Image 1: Original image with RED box marking the source object(s).
- Image 2: Edited image after moving the object(s).
- Image 3: Annotated Image 2 with RED box for source area and GREEN box for target area.

The edit instruction was: "{edit_prompt}".

Please give ONE overall quality score from 1 to 10 for the edited result in Image 2.

### What to Consider (holistic)
Judge the final quality of the whole edit by combining these factors:
1. Naturalness and realism of the final image.
2. Whether source area is properly cleaned after object removal.
3. Whether object is plausibly integrated at target area (shape, boundary, local lighting/shadow/context).
4. Whether unrelated regions are preserved without unwanted changes.

Use Image 3 only as spatial reference for source/target positions. Evaluate quality based on Image 2.

### Important Priority
Do not over-penalize target-position mismatch or imperfect movement precision. Movement precision is not the primary focus in this score.
Focus more on overall edit quality, visual realism, scene harmony, and semantic correctness.

### Scoring Rubric (1-10)
- 1-2: Very poor. Severe artifacts, obvious failure, or object missing/broken.
- 3-4: Poor. Major quality issues are clearly visible.
- 5-6: Fair. Edit works roughly but has noticeable defects.
- 7-8: Good. Mostly natural with minor flaws.
- 9: Very good. Nearly flawless with only tiny imperfections.
- 10: Perfect. Fully realistic and seamless; no visible defects.

Be strict and objective. Give 10 only when the result is truly flawless.

### Output Format
Respond in JSON only:
{{
    "overall_score": int,
    "explanation": str
}}
"""


LOGIC_CONSISTENCY_PROMPT_TEMPLATE = """
### Task
You are given two edited images of the same result:
- Image 1: Edited image.
- Image 2: Edited image annotated with object bounding boxes.

Please evaluate **Logic_Consistency (Weight: Physical Realism + Semantic Harmony)** from 1 to 5.

### Focus
Judge whether the final edited scene is physically plausible, photo-realistic, and semantically harmonious in a unified way.

Consider these aspects together:
- Lighting direction/intensity/color temperature coherence.
- Shadows, reflections, contact realism, and natural edge blending.
- Perspective, geometry, scale, depth ordering, and material consistency.
- No obvious violations of gravity/support/contact/occlusion.
- Edited content fits scene context, object relations, and global coherence.

Use Image 2 to focus analysis on boxed regions, while still considering whole-image consistency.

### Scoring Rubric (1-5)
- 1: Strongly unrealistic. Severe artifacts or major physical-law violations.
- 2: Clearly unrealistic. Multiple obvious physics/realism problems.
- 3: Partially realistic. Acceptable overall but noticeable physical inconsistencies.
- 4: Mostly realistic. Minor flaws only, no major physical contradiction.
- 5: Highly realistic and physically coherent. Looks like a real photo edit.

Be strict and objective. Give 5 only when quality is truly excellent.

### Output Format
Respond in JSON only:
{{
    "logic_consistency_score": int,
    "explanation": str
}}
"""


class VLMScoringOutput(BaseModel):
    consistency_score: int
    inpainting_score: int
    integration_score: int
    preservation_score: int
    explanation: str = ""


class VLMSimpleScoringOutput(BaseModel):
    overall_score: int
    explanation: str = ""


class VLMLogicConsistencyOutput(BaseModel):
    logic_consistency_score: int
    explanation: str = ""


GROUNDING_PROMPT_TEMPLATE = """
Task: Locate the target object in the 'Edited Image' based on the provided references.

Setting:
Image 2 is the edited image from Image 1 by applying the moving operation on the object "{object_name}". The {object_name} may have been moved to a new location, and may have undergone certain appearance changes. Your task is to find the object in Image 2 and grounding it with precise bounding box.
The two given images are (in order):
- Image 1 (Annotated Original Image): The object "{object_name}" is highlighted with a RED bounding box (source location).
- Image 2 (Annotated Edited Image): GREEN highlighted region indicates the expected target area in edited image.

Instructions:
1. First, use Image 1 to understand what object to track from source location.
2. Search around the GREEN highlighted target area and locate "{object_name}" in Image 2. If the object is not found near the target area, expand search to the whole Image 2.
3. Ground the object in Image 2 with a precise bounding box. Return the bounding box in [xmin, ymin, xmax, ymax] format (relative coordinates scaled to [0, 1000]).
4. Also provide a brief but precise object description for segmentation guidance in Image 2:
     - If object name alone is already uniquely identifiable, use only the plain object name.
     - If ambiguous (e.g. multiple similar objects), add short qualifiers (position, relation, local appearance), but keep it brief.
     - Keep this text concise and practical for segmentation prompts (e.g. "apple on the left", "person wearing a hat", "red car in the back").
5. If the object is completely missing, severely corrupted, or cannot be found, set bbox to [0, 0, 0, 0], and set object_description to the plain object name.

Respond ONLY with JSON:
{{
    "bbox": [xmin, ymin, xmax, ymax],
    "object_description": "brief precise description"
}}
"""


class VLMGroundingOutput(BaseModel):
    bbox: list[int]
    object_description: str


def encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_from_base64(image_str: str) -> Image.Image:
    image_bytes = base64.b64decode(image_str.encode("utf-8"))
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _image_to_data_url(image: Image.Image) -> str:
    return f"data:image/png;base64,{encode_image_to_base64(image)}"


def _extract_json_obj(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text}")
    return json.loads(match.group(0))


def _detect_refusal_or_policy_block(text: str) -> str | None:
    lowered = text.lower()
    refusal_markers = [
        "i can't help",
        "i cannot help",
        "can't assist",
        "cannot assist",
        "cannot comply",
        "can't comply",
        "policy",
        "safety",
        "content policy",
        "not appropriate",
        "cannot provide",
        "inappropriate",
        "refuse",
    ]
    for marker in refusal_markers:
        if marker in lowered:
            return marker
    return None


def _target_size_from_aspect(width: int, height: int) -> tuple[int, int]:
    # Allowed inputs are expected near 16:9, 3:2, 1:1.
    if width <= 0 or height <= 0:
        return (768, 432)
    ratio = width / height
    candidates = [
        ((768, 432), 16 / 9),
        ((768, 512), 3 / 2),
        ((512, 512), 1.0),
    ]
    best_size, _ = min(candidates, key=lambda item: abs(ratio - item[1]))
    return best_size


def normalize_image_for_eval(image: Image.Image) -> Image.Image:
    target_size = _target_size_from_aspect(*image.size)
    if image.size == target_size:
        return image
    return image.resize(target_size, resample=Image.Resampling.BILINEAR)


class VLMModel:
    def __init__(self, base_url=None, model_name=None, api_key=None, timeout=300):
        raw_base_url = (
            base_url or os.getenv("VLLM_BASE_URL") or os.getenv("VLM_SERVER_URL") or "http://127.0.0.1:8000"
        ).rstrip("/")
        if raw_base_url.endswith("/v1"):
            self.base_url = raw_base_url
        else:
            self.base_url = f"{raw_base_url}/v1"

        self.model_name = (
            model_name or os.getenv("VLLM_MODEL_NAME") or os.getenv("VLM_MODEL_PATH", "Qwen/Qwen3-VL-32B-Instruct")
        )
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.timeout = timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        print(f"Using vLLM OpenAI endpoint: {self.base_url} (model={self.model_name})")

    def _chat_completion(
        self,
        messages,
        max_tokens=512,
        temperature=0.0,
        response_format: Any = omit,
        enable_thinking: bool | None = None,
    ):
        if enable_thinking is None:
            enable_thinking = str(os.getenv("VLM_ENABLE_THINKING", "1")).lower() not in {"0", "false", "no", "off"}
        max_attempts = max(int(os.getenv("VLM_REQUEST_MAX_ATTEMPTS", "3")), 1)
        retry_delay = max(float(os.getenv("VLM_REQUEST_RETRY_DELAY_SECONDS", "1")), 0.0)
        structured_requested = response_format is not omit
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            # Some compatible providers occasionally abort schema-constrained generation. Retry
            # without response_format; every prompt already requires JSON text.
            use_structured_output = structured_requested and attempt == 0
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extra_body": {"enable_thinking": enable_thinking},
                }
                if use_structured_output:
                    response = self.client.chat.completions.parse(
                        **request_kwargs,
                        response_format=response_format,
                    )
                else:
                    response = self.client.chat.completions.create(**request_kwargs)

                response_message = response.choices[0].message
                parsed = getattr(response_message, "parsed", None)
                if parsed is not None:
                    return parsed
                content = getattr(response_message, "content", None)
                if content is not None:
                    return content
                raise ValueError(f"No content in VLM response: {response}")
            except Exception as error:
                last_error = error
                if attempt + 1 >= max_attempts:
                    break
                next_mode = "plain-json fallback" if use_structured_output else "plain-json retry"
                print(
                    f"[VLM][RequestRetry] attempt={attempt + 1}/{max_attempts}, "
                    f"next={next_mode}, error={error}"
                )
                if retry_delay > 0:
                    time.sleep(retry_delay * (2**attempt))

        error_msg = (
            f"OpenAI-compatible request failed at {self.base_url} after "
            f"{max_attempts} attempts: {last_error}"
        )
        print(f"[VLM][RequestError] {error_msg}")
        return {
            "_vlm_error": True,
            "error_type": "request_error",
            "error_message": error_msg,
        }

    def score_editing_batch(self, original_images, edited_images, edit_prompts, orig_bboxes, target_bboxes):
        scores = []
        for i in tqdm(range(len(original_images)), desc="Scoring editing batch with vLLM..."):
            try:
                pil_f1 = original_images[i]
                pil_f2 = edited_images[i]
                annotated_ref = draw_src_target_boxes(
                    image=pil_f2,
                    orig_bboxes=orig_bboxes[i],
                    target_bboxes=target_bboxes[i],
                )

                messages = [
                    {"role": "system", "content": "You are an expert image editing auditor."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SCORING_PROMPT_TEMPLATE.format(edit_prompt=edit_prompts[i])},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(pil_f1)}},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(pil_f2)}},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(annotated_ref)}},
                        ],
                    },
                ]

                response = self._chat_completion(
                    messages=messages, max_tokens=512, temperature=0.0, response_format=VLMScoringOutput
                )

                if isinstance(response, dict) and response.get("_vlm_error"):
                    print(f"[VLM][ScoringWarning] sample={i} error={response.get('error_message', 'unknown')}")
                    scores.append(
                        {
                            "consistency_score": None,
                            "inpainting_score": None,
                            "integration_score": None,
                            "preservation_score": None,
                            "explanation": "",
                            "vlm_failed": True,
                            "vlm_error": response.get("error_message", "unknown_error"),
                        }
                    )
                    continue

                if isinstance(response, VLMScoringOutput):
                    score_item = response.model_dump()
                    score_item["vlm_failed"] = False
                    scores.append(score_item)
                elif isinstance(response, str):
                    try:
                        parsed = _extract_json_obj(response)
                        score_item = VLMScoringOutput.model_validate(parsed).model_dump()
                        score_item["vlm_failed"] = False
                        scores.append(score_item)
                    except Exception as parse_error:
                        print(f"[VLM][ScoringWarning] sample={i} parse_error={parse_error}")
                        scores.append(
                            {
                                "consistency_score": None,
                                "inpainting_score": None,
                                "integration_score": None,
                                "preservation_score": None,
                                "explanation": "",
                                "vlm_failed": True,
                                "vlm_error": f"parse_error: {parse_error}",
                            }
                        )
                else:
                    print(f"[VLM][ScoringWarning] sample={i} unexpected_response={type(response)}")
                    scores.append(
                        {
                            "consistency_score": None,
                            "inpainting_score": None,
                            "integration_score": None,
                            "preservation_score": None,
                            "explanation": "",
                            "vlm_failed": True,
                            "vlm_error": f"unexpected_response_type: {type(response)}",
                        }
                    )
            except Exception as error:
                print(f"[VLM][ScoringWarning] sample={i} exception={error}")
                scores.append(
                    {
                        "consistency_score": None,
                        "inpainting_score": None,
                        "integration_score": None,
                        "preservation_score": None,
                        "explanation": "",
                        "vlm_failed": True,
                        "vlm_error": str(error),
                    }
                )

        return scores

    def score_editing_batch_simple(self, original_images, edited_images, edit_prompts, orig_bboxes, target_bboxes):
        scores = []
        for i in tqdm(range(len(original_images)), desc="Scoring editing batch (1-10) with vLLM..."):
            try:
                pil_f1 = original_images[i]
                pil_f2 = edited_images[i]
                annotated_ref = draw_src_target_boxes(
                    image=pil_f2,
                    orig_bboxes=orig_bboxes[i],
                    target_bboxes=target_bboxes[i],
                )

                messages = [
                    {"role": "system", "content": "You are a strict and professional image editing auditor."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": SIMPLE_SCORING_PROMPT_TEMPLATE.format(edit_prompt=edit_prompts[i]),
                            },
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(pil_f1)}},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(pil_f2)}},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(annotated_ref)}},
                        ],
                    },
                ]

                response = self._chat_completion(
                    messages=messages,
                    max_tokens=256,
                    temperature=0.0,
                    response_format=VLMSimpleScoringOutput,
                )

                if isinstance(response, dict) and response.get("_vlm_error"):
                    print(f"[VLM][SimpleScoringWarning] sample={i} error={response.get('error_message', 'unknown')}")
                    scores.append(
                        {
                            "overall_score": None,
                            "explanation": "",
                            "vlm_failed": True,
                            "vlm_error": response.get("error_message", "unknown_error"),
                        }
                    )
                    continue

                if isinstance(response, VLMSimpleScoringOutput):
                    score_item = response.model_dump()
                    score_item["vlm_failed"] = False
                    scores.append(score_item)
                elif isinstance(response, str):
                    try:
                        parsed = _extract_json_obj(response)
                        score_item = VLMSimpleScoringOutput.model_validate(parsed).model_dump()
                        score_item["vlm_failed"] = False
                        scores.append(score_item)
                    except Exception as parse_error:
                        print(f"[VLM][SimpleScoringWarning] sample={i} parse_error={parse_error}")
                        scores.append(
                            {
                                "overall_score": None,
                                "explanation": "",
                                "vlm_failed": True,
                                "vlm_error": f"parse_error: {parse_error}",
                            }
                        )
                else:
                    print(f"[VLM][SimpleScoringWarning] sample={i} unexpected_response={type(response)}")
                    scores.append(
                        {
                            "overall_score": None,
                            "explanation": "",
                            "vlm_failed": True,
                            "vlm_error": f"unexpected_response_type: {type(response)}",
                        }
                    )
            except Exception as error:
                print(f"[VLM][SimpleScoringWarning] sample={i} exception={error}")
                scores.append(
                    {
                        "overall_score": None,
                        "explanation": "",
                        "vlm_failed": True,
                        "vlm_error": str(error),
                    }
                )

        return scores

    def score_logic_consistency_batch(self, edited_images, annotated_images, run_name: str = ""):
        scores = []
        desc = "Scoring logic consistency (1-5) with vLLM..."
        if run_name:
            desc = f"[{run_name}] {desc}"
        for i in tqdm(range(len(edited_images)), desc=desc):
            try:
                edited_image = normalize_image_for_eval(edited_images[i])
                annotated_image = normalize_image_for_eval(annotated_images[i])

                messages = [
                    {
                        "role": "system",
                        "content": "You are a strict expert image auditor specializing in physical realism and consistency.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": LOGIC_CONSISTENCY_PROMPT_TEMPLATE},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(edited_image)}},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(annotated_image)}},
                        ],
                    },
                ]

                logic_max_tokens = int(os.getenv("VLM_LOGIC_MAX_TOKENS", "512"))
                logic_enable_thinking = str(os.getenv("VLM_LOGIC_ENABLE_THINKING", "0")).lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                response = self._chat_completion(
                    messages=messages,
                    max_tokens=logic_max_tokens,
                    temperature=0.0,
                    response_format=VLMLogicConsistencyOutput,
                    enable_thinking=logic_enable_thinking,
                )

                if isinstance(response, dict) and response.get("_vlm_error"):
                    print(f"[VLM][LogicConsistencyWarning] sample={i} error={response.get('error_message', 'unknown')}")
                    scores.append(
                        {
                            "logic_consistency_score": None,
                            "explanation": "",
                            "vlm_failed": True,
                            "vlm_error": response.get("error_message", "unknown_error"),
                        }
                    )
                    continue

                if isinstance(response, VLMLogicConsistencyOutput):
                    score_item = response.model_dump()
                    score_item["vlm_failed"] = False
                    scores.append(score_item)
                elif isinstance(response, str):
                    refusal_marker = _detect_refusal_or_policy_block(response)
                    if refusal_marker is not None:
                        print(
                            f"[VLM][LogicConsistencyWarning] sample={i} policy/refusal_detected marker='{refusal_marker}'"
                        )
                        scores.append(
                            {
                                "logic_consistency_score": None,
                                "explanation": response[:500],
                                "vlm_failed": True,
                                "vlm_error": f"policy_or_refusal_response: {refusal_marker}",
                            }
                        )
                        continue
                    try:
                        parsed = _extract_json_obj(response)
                        score_item = VLMLogicConsistencyOutput.model_validate(parsed).model_dump()
                        score_item["vlm_failed"] = False
                        scores.append(score_item)
                    except Exception as parse_error:
                        print(f"[VLM][LogicConsistencyWarning] sample={i} parse_error={parse_error}")
                        scores.append(
                            {
                                "logic_consistency_score": None,
                                "explanation": "",
                                "vlm_failed": True,
                                "vlm_error": f"parse_error: {parse_error}",
                            }
                        )
                else:
                    print(f"[VLM][LogicConsistencyWarning] sample={i} unexpected_response={type(response)}")
                    scores.append(
                        {
                            "logic_consistency_score": None,
                            "explanation": "",
                            "vlm_failed": True,
                            "vlm_error": f"unexpected_response_type: {type(response)}",
                        }
                    )
            except Exception as error:
                print(f"[VLM][LogicConsistencyWarning] sample={i} exception={error}")
                scores.append(
                    {
                        "logic_consistency_score": None,
                        "explanation": "",
                        "vlm_failed": True,
                        "vlm_error": str(error),
                    }
                )

        return scores

    def detect_object_vllm(
        self,
        original_images,
        edited_images,
        orig_bboxes,
        obj_names,
        target_bboxes=None,
        cache_keys: list[str] | None = None,
        cached_results: dict[str, dict] | None = None,
        on_result: Callable[[str, dict], None] | None = None,
    ):
        if cache_keys is not None and len(cache_keys) != len(original_images):
            raise ValueError("cache_keys must align with grounding inputs")

        def _detect_one(i: int):
            try:
                current_target_bbox = None if target_bboxes is None else target_bboxes[i]
                source_annotated_original = draw_bbox(image=original_images[i], orig_bbox=orig_bboxes[i])
                target_annotated_edited = draw_detection_hint(image=edited_images[i], target_bbox=current_target_bbox)

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert image analyst specializing in object grounding.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": GROUNDING_PROMPT_TEMPLATE.format(object_name=obj_names[i]),
                            },
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(source_annotated_original)}},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(target_annotated_edited)}},
                        ],
                    },
                ]

                response = self._chat_completion(
                    messages=messages,
                    max_tokens=128,
                    temperature=0.0,
                    response_format=VLMGroundingOutput,
                    enable_thinking=False,
                )
                if isinstance(response, dict) and response.get("_vlm_error"):
                    print(f"[VLM][GroundingWarning] sample={i} error={response.get('error_message', 'unknown')}")
                    return {
                        "bbox": [0, 0, 0, 0],
                        "object_description": obj_names[i],
                        "vlm_failed": True,
                        "vlm_error": response.get("error_message", "unknown_error"),
                    }

                if isinstance(response, VLMGroundingOutput):
                    box = [int(v) for v in response.bbox]
                    object_description = response.object_description.strip()
                elif isinstance(response, str):
                    parsed = None
                    try:
                        parsed = _extract_json_obj(response)
                    except Exception:
                        parsed = None

                    if isinstance(parsed, dict) and "bbox" in parsed:
                        box = [int(v) for v in parsed["bbox"]]
                        object_description = str(parsed.get("object_description", "")).strip()
                    else:
                        match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", response)
                        if not match:
                            raise ValueError(f"No bounding box found in response: {response}")
                        box = [int(g) for g in match.groups()]
                        object_description = ""
                else:
                    raise ValueError(f"Unexpected response format: {response}")

                if not object_description:
                    object_description = obj_names[i]

                return {
                    "bbox": box,
                    "object_description": object_description,
                    "vlm_failed": False,
                }
            except Exception as error:
                print(f"[VLM][GroundingWarning] sample={i} exception={error}")
                return {
                    "bbox": [0, 0, 0, 0],
                    "object_description": obj_names[i],
                    "vlm_failed": True,
                    "vlm_error": str(error),
                }

        cached_results = cached_results or {}
        detected_results: list[dict | None] = [None] * len(original_images)
        pending_indices = []
        for i in range(len(original_images)):
            cache_key = cache_keys[i] if cache_keys is not None else None
            cached = cached_results.get(cache_key) if cache_key is not None else None
            if isinstance(cached, dict) and not bool(cached.get("vlm_failed", False)):
                detected_results[i] = dict(cached)
            else:
                pending_indices.append(i)

        reused_count = len(original_images) - len(pending_indices)
        if reused_count:
            print(f"Reused cached VLM grounding results: {reused_count}/{len(original_images)}")

        def _store_result(i: int, result: dict) -> None:
            detected_results[i] = result
            if cache_keys is not None and on_result is not None:
                on_result(cache_keys[i], result)

        def _finalize_results() -> list[dict]:
            missing = [i for i, result in enumerate(detected_results) if result is None]
            if missing:
                raise RuntimeError(f"Missing VLM grounding results at indices: {missing[:10]}")
            return [result for result in detected_results if result is not None]

        max_workers = int(os.getenv("VLM_GROUNDING_MAX_WORKERS", "1"))
        if max_workers <= 1:
            for i in tqdm(pending_indices, desc="Detecting objects with vLLM..."):
                _store_result(i, _detect_one(i))
            return _finalize_results()

        print(f"Detecting objects with vLLM using {max_workers} concurrent requests...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            result_iter = executor.map(_detect_one, pending_indices)
            for i, result in zip(
                pending_indices,
                tqdm(
                    result_iter,
                    total=len(pending_indices),
                    desc="Detecting objects with vLLM...",
                ),
            ):
                _store_result(i, result)
        return _finalize_results()
