"""Convert raw CoT trajectory JSONL files into LLaMA-Factory ShareGPT format.

This script reads JSONL files produced by the CoT generation pipeline, applies
post-processing (flatten images, normalise bounding boxes, merge information
turns, inject the system prompt), and exports a single JSON file compatible
with LLaMA-Factory's multimodal SFT training.

Usage
-----
    python convert_to_llamafactory.py \
        --input-dir  /path/to/cot_jsonl_dir \
        --output-path /path/to/output.json

    python convert_to_llamafactory.py \
        --input-path /path/to/single_file.jsonl \
        --output-path /path/to/output.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from io import BytesIO
from typing import Union

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def process_image(
    image: Union[str, dict, Image.Image],
    max_pixels: int = 2048 * 2048,
    min_pixels: int = 512 * 512,
) -> Image.Image:
    """Resize *image* so its total pixel count stays within [*min_pixels*, *max_pixels*].

    Parameters
    ----------
    image:
        A file path, a dict with a ``bytes`` key, or a PIL Image.
    max_pixels:
        Upper bound on total pixel count.
    min_pixels:
        Lower bound on total pixel count.
    """
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, str):
        image = Image.open(image)

    if image.width * image.height > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))

    if image.width * image.height < min_pixels:
        factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def trans_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    """Convert normalised [0, 1000] bbox to pixel-space coordinates."""
    x1_norm, y1_norm, x2_norm, y2_norm = (v / 1000.0 for v in bbox)
    return [
        int(x1_norm * width),
        int(y1_norm * height),
        int(x2_norm * width),
        int(y2_norm * height),
    ]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """Please strictly follow the steps below to answer the user's question:

1. **Reason First**  
   Upon receiving any new information, you **must** first perform reasoning within <think> and </think> tags to determine whether you already have sufficient information to answer the user's question directly.

2. **Initiate a Search (if needed)**  
   If reasoning indicates that necessary information is missing, initiate an image search using <search>query</search>. The system will return a set of images (indexed starting from 0).

3. **Image Analysis and Selection**  
   - Within the <think> tags, **analyze each image one by one**, focusing on whether it contains **text, charts, tables, labels, or other key visual information** relevant to the query.  
   - Evaluate the completeness and relevance of information in each image, and **select the single image most helpful for answering the query**.  
   - Explicitly specify the index of the chosen image in a <select> tag (e.g., <select>2</select>).

4. **Detailed Image Content Analysis**  
   **After selecting an image, proceed with this two-stage analysis:**
   
   ### 4.1 Initial ROI Identification and Quality Assessment
   Within `<think>` tags:
   - Identify specific regions of interest (ROI) with coordinates `[x1, y1, x2, y2]` relevant to the query
   - For each region, summarize its key semantic content
   - **Assess readability**: Determine if content is clear or needs magnification
   
   ### 4.2 Content Extraction Decision
   Based on quality assessment:
   - **If content is readable**: Extract semantic meaning and output `<information>Answer based on this image</information>`
   - **If content needs magnification**: Request specific regions using `<bbox>[[x1, y1, x2, y2], ...]</bbox>` and pause
     *Magnification is needed only when enlarging the ROI of the selected image might help answer the query more clearly and accurately.*

5. **Magnified Image Processing (If Applicable)**  
   **Only execute if magnification was requested in Step 4:**
   - Upon receiving magnified image(s), analyze within `<think>` tags
   - Extract relevant information from the high-resolution regions
   - Output findings in `<information>Based on magnified image analysis</information>`

6. **Final Answer Determination**  
   Within `<think>` tags, evaluate:
   - Is current information sufficient to answer the original question?
   - What critical information (if any) remains missing?
   
   **Output exactly one:**
   - If insufficient: `<search>precise follow-up query</search>`
   - If sufficient: `<answer>direct final answer</answer>`

7. **Iterative Refinement**  
   Repeat Steps 3-6 as needed until sufficient information is gathered to provide a complete and accurate final answer.

**CRITICAL CLARIFICATION ON MAGNIFICATION:**
- Request magnification ONLY when enlarging the specific region of the selected image might help answer the query more clearly and accurately
- Use `<bbox>` coordinates to specify EXACT regions that need magnification
- After magnification, base your analysis SOLELY on the magnified region

Always adhere strictly to this sequential workflow to ensure methodical reasoning, justified image selection, and precise information extraction.

**User question: {question}**"""


def crop_and_dump(
    image_path: str,
    bbox_list: list[list[int]],
    output_path: str,
) -> str | None:
    """Crop regions from *image_path*, stitch them vertically, and save.

    Parameters
    ----------
    image_path:
        Path to the source image.
    bbox_list:
        Bounding boxes as absolute pixel coords in the *processed* image space.
    output_path:
        Destination path for the stitched crop image.

    Returns
    -------
    str | None
        *output_path* on success, ``None`` on failure.
    """
    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    try:
        image = Image.open(image_path)
        image_zoom = process_image(image, 512 * 28 * 28, 256 * 28 * 28)
    except Exception as exc:
        logger.warning("Cannot open %s: %s", image_path, exc)
        return None

    cropped_images: list[Image.Image] = []
    for bbox in bbox_list:
        # Map from processed-image space back to original image space.
        x1, y1, x2, y2 = bbox
        x1 = int(x1 / image_zoom.width * image.width)
        y1 = int(y1 / image_zoom.height * image.height)
        x2 = int(x2 / image_zoom.width * image.width)
        y2 = int(y2 / image_zoom.height * image.height)

        x1 = max(0, min(x1, image.width))
        y1 = max(0, min(y1, image.height))
        x2 = max(0, min(x2, image.width))
        y2 = max(0, min(y2, image.height))

        if x1 >= x2 or y1 >= y2:
            logger.warning("Invalid bbox %s, skipping.", bbox)
            continue
        cropped_images.append(image.crop((x1, y1, x2, y2)))

    if not cropped_images:
        logger.warning("No valid crops to process.")
        return None

    stitched = _stitch_vertically(cropped_images)

    try:
        stitched.save(output_path)
        return output_path
    except Exception as exc:
        logger.error("Failed to save stitched image: %s", exc)
        return None


def _stitch_vertically(images: list[Image.Image]) -> Image.Image:
    """Stitch a list of PIL images into a single image, top-to-bottom."""
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    mode = images[0].mode if images[0].mode in ("RGB", "RGBA") else "RGB"
    canvas = Image.new(mode, (max_width, total_height))

    y_offset = 0
    for img in images:
        if img.mode != mode:
            img = img.convert(mode)
        canvas.paste(img, (0, y_offset))
        y_offset += img.height

    return canvas


# ---------------------------------------------------------------------------
# Data loading and conversion
# ---------------------------------------------------------------------------

def load_data(json_file_path: str) -> tuple[list[dict], int]:
    """Load a single JSONL trajectory file and convert to ShareGPT format.

    Each line is expected to be a JSON object with ``messages`` and ``images``
    keys, produced by the CoT generation pipeline.

    Returns
    -------
    data : list[dict]
        Converted examples with ``messages`` and ``images`` keys.
    crop_num : int
        Number of examples that contain bounding-box crop requests.
    """
    data: list[dict] = []
    crop_num = 0

    with open(json_file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line in %s", json_file_path)
                continue

            if example is None:
                continue

            messages: list[dict] = example["messages"]

            # Only keep examples that passed the judge.
            if messages[-1]["content"] != "The judge is YES.":
                continue
            del messages[-1]
            
            # Flatten nested image lists into a single list of paths.
            images = _flatten_images(example["images"])

            # Normalise bounding boxes and detect crop usage.
            messages, has_crop = _normalise_bboxes(messages, images)
            if has_crop:
                crop_num += 1

            # Sanity check: every <image> must have a corresponding path.
            total_image_refs = sum(m["content"].count("<image>") for m in messages)
            if total_image_refs != len(images):
                logger.warning(
                    "Image count mismatch (%d refs vs %d paths) – skipping example",
                    total_image_refs,
                    len(images),
                )
                continue

            # Validate image paths.
            for img_path in images:
                if not os.path.exists(img_path):
                    logger.warning("Image not found: %s", img_path)

            # Inject system prompt into the first user message.
            question = messages[0]["content"]
            messages[0]["content"] = SYSTEM_PROMPT.format(question=question)

            data.append({"messages": messages, "images": images})

    return data, crop_num


def _flatten_images(raw_images: list) -> list[str]:
    """Flatten nested image lists into a single list of path strings."""
    flat: list[str] = []
    for item in raw_images:
        if isinstance(item, str):
            flat.append(item)
        else:
            flat.extend(item)
    return flat


_BBOX_PATTERN = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")


def _normalise_bboxes(
    messages: list[dict],
    images: list[str],
) -> tuple[list[dict], bool]:
    """Convert pixel-space bboxes to normalised [0,1000] coords; detect crop requests."""
    image_idx = 0
    has_crop = False

    for msg_idx, message in enumerate(messages):
        content = message["content"]
        image_count = content.count("<image>")
        image_idx += image_count

        if image_count == 0:
            continue

        image_path = images[image_idx - 1]
        matches = list(set(_BBOX_PATTERN.findall(content)))

        if matches:
            try:
                pil_image = process_image(
                    Image.open(image_path), 512 * 28 * 28, 256 * 28 * 28
                )
            except Exception as exc:
                logger.warning("Cannot open %s: %s", image_path, exc)
            else:
                w, h = pil_image.size
                for match in matches:
                    bbox = [int(c) for c in match]
                    content = content.replace(str(bbox), str(trans_bbox(bbox, w, h)))

        if "<bbox>" in content:
            has_crop = True

        messages[msg_idx]["content"] = content

    return messages, has_crop


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CoT trajectory JSONL files to LLaMA-Factory SFT format.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing one or more JSONL trajectory files.",
    )
    group.add_argument(
        "--input-path",
        type=str,
        help="Path to a single JSONL trajectory file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for the converted JSON file.",
    )
    return parser.parse_args()


def _collect_input_files(args: argparse.Namespace) -> list[str]:
    """Resolve input arguments into a list of JSONL file paths."""
    if args.input_dir:
        return sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith((".jsonl", ".json"))
        )
    return [args.input_path]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()

    all_data: list[dict] = []
    all_crop_num = 0

    for json_file_path in _collect_input_files(args):
        logger.info("Processing %s ...", json_file_path)
        data, crop_num = load_data(json_file_path)
        all_data.extend(data)
        all_crop_num += crop_num
        logger.info("  -> %d examples, %d with crops", len(data), crop_num)

    # Ensure output directory exists.
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as fh:
        json.dump(all_data, fh, ensure_ascii=False, indent=4)

    logger.info(
        "Saved %d examples (%d with crops) to %s",
        len(all_data),
        all_crop_num,
        args.output_path,
    )


if __name__ == "__main__":
    main()
