"""Utilities for bounding-box parsing, drawing, cropping, and IoU matching."""

import os
import re

import cv2
from PIL import Image


def extract_bboxes_from_string(text: str):
    """Extract all `[x1, y1, x2, y2]` bounding boxes from a string."""
    pattern = r"\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]"
    matches = re.finditer(pattern, text)

    original_texts = []
    extracted_bboxes = []

    for match in matches:
        original_text = match.group(0)
        coords = match.groups()
        try:
            extracted_bboxes.append([int(coord) for coord in coords])
            original_texts.append(original_text)
        except ValueError:
            continue

    return original_texts, extracted_bboxes


def draw_bboxes_on_image(image_path, bboxes, output_path, color=(0, 0, 255), thickness=2):
    """Draw bounding boxes on an image and save the result."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to: {output_path}")


def convert_bbox(bbox_l: list[list[float]], reverse: bool, image_path: str) -> list[list[float]]:
    """Convert between normalized and pixel-space bounding boxes."""
    results = []
    image = Image.open(image_path)
    width, height = image.size

    for bbox in bbox_l:
        if reverse:
            x1 = int((bbox[0] / width) * 1000.0)
            y1 = int((bbox[1] / height) * 1000.0)
            x2 = int((bbox[2] / width) * 1000.0)
            y2 = int((bbox[3] / height) * 1000.0)
        else:
            x1 = int((bbox[0] / 1000.0) * width)
            y1 = int((bbox[1] / 1000.0) * height)
            x2 = int((bbox[2] / 1000.0) * width)
            y2 = int((bbox[3] / 1000.0) * height)

        results.append([x1, y1, x2, y2])

    return results


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    """Resize an image so its resolution stays within the configured bounds."""
    import math

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def bboxes_image(bbox_list, image_path, save_path):
    """Render indexed bounding boxes on an image and save the visualization."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    all_bboxes = []
    for idx, bbox in enumerate(bbox_list):
        x1, y1, x2, y2 = convert_bbox([bbox], reverse=False, image_path=image_path)[0]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        text = str(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        text_x = max(x1 - text_w - 10, 0)
        text_y = y1 + (y2 - y1) // 2 + text_h // 2

        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        all_bboxes.append(bbox)

    cv2.imwrite(save_path, image)
    return all_bboxes


def crop_and_dump(image_path, bbox_list, output_folder):
    """Crop multiple regions, stitch them vertically, and save the output image."""
    os.makedirs(output_folder, exist_ok=True)

    try:
        image = Image.open(image_path)
    except Exception as exc:
        print(f"Cannot open {image_path}: {exc}")
        return None

    cropped_images = []
    width, height = image.size

    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        x1 = int((x1 / 1000.0) * width)
        y1 = int((y1 / 1000.0) * height)
        x2 = int((x2 / 1000.0) * width)
        y2 = int((y2 / 1000.0) * height)

        x1 = int(max(0, min(x1, image.width - 1)))
        y1 = int(max(0, min(y1, image.height - 1)))
        x2 = int(max(0, min(x2, image.width - 1)))
        y2 = int(max(0, min(y2, image.height - 1)))

        if x1 >= x2 or y1 >= y2:
            print(f"Invalid bbox {bbox}, skipping.")
            continue

        cropped_images.append(image.crop((x1, y1, x2, y2)))

    if not cropped_images:
        print("No valid crops to process.")
        return None

    max_width = max(img.width for img in cropped_images)
    total_height = sum(img.height for img in cropped_images)
    mode = cropped_images[0].mode if cropped_images[0].mode in ("RGB", "RGBA") else "RGB"
    stitched = Image.new(mode, (max_width, total_height))

    y_offset = 0
    for img in cropped_images:
        if img.mode != mode:
            img = img.convert(mode)
        stitched.paste(img, (0, y_offset))
        y_offset += img.height

    stitched = process_image(stitched, 512 * 28 * 28, 256 * 28 * 28)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    bbox_name_str = "_".join(
        [f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}" for x1, y1, x2, y2 in bbox_list]
    )[:100]
    output_path = os.path.join(output_folder, f"{base_name}_{bbox_name_str}.png")

    try:
        stitched.save(output_path)
        return output_path
    except Exception as exc:
        print(f"Failed to save stitched image: {exc}")
        return None


def calculate_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_bboxes_by_iou(bbox_list, layout_bbox_list):
    """Match each bbox to the layout bbox with the highest IoU."""
    if not layout_bbox_list:
        return [(bbox, None, 0.0) for bbox in bbox_list]

    matched_bboxes = []
    for bbox in bbox_list:
        max_iou = -1.0
        best_match = None

        for layout_bbox in layout_bbox_list:
            iou = calculate_iou(bbox, layout_bbox)
            if iou > max_iou:
                max_iou = iou
                best_match = layout_bbox

        if best_match not in matched_bboxes:
            matched_bboxes.append(best_match)

    return matched_bboxes
