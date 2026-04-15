"""Core CoT generation pipeline with search, reranking, and optional cropping."""

import base64
import io
import itertools
import json
import os
import random
import threading

import cv2
import dirtyjson
import requests
from PIL import Image

from mineru_layout import mineru_parse_doc
from llm import Qwen_LLM_Client, Qwen_VLLM_Client
from prompt import *
from utils_bbox import *


def extract_json(response) -> dict:
    """Parse a JSON payload from a model response."""
    response = response.strip().replace("```json", "").replace("```", "")
    response = response.replace("```\n", "").replace("\n```", "")
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            return dirtyjson.loads(response)
        except Exception as exc:
            print(f"JSON parse error: {exc}")
            return {}


def safe_num(value):
    """Return a safe string representation for numeric model outputs."""
    return str(value) if value else "0"


def encode_image(image_path):
    """Encode an image file as base64."""
    if not os.path.exists(image_path):
        return ""

    with Image.open(image_path) as image:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image.format if image.format else "JPEG")
        return base64.b64encode(image_bytes.getvalue()).decode("utf-8")


class CoT_Generator:
    """Generate multi-step CoT traces over search and image evidence."""

    _global_perm_pool = {}
    _global_perm_lock = threading.Lock()

    def __init__(
        self,
        llm_engine_url,
        vllm_engine_url,
        search_engine_url,
        layout_parser_url="http://127.0.0.1:30000",
        max_steps=3,
        image_topk=3,
        crop_folder="crops",
        crop_all=False,
        enable_rerank=False,
        enable_crop=False,
        merge_bbox=False,
    ):
        self.llm = (
            Qwen_LLM_Client(
                base_url=llm_engine_url,
                model_name="Qwen3-VL-235B-A22B-Instruct-FP8",
            )
            if llm_engine_url
            else None
        )
        self.vllm = Qwen_VLLM_Client(
            base_url=vllm_engine_url,
            model_name="Qwen3-VL-235B-A22B-Instruct-FP8",
        )
        self.search_url = search_engine_url
        self.layout_parser_url = layout_parser_url

        self.max_steps = max_steps
        self.topk = image_topk
        self.crop_folder = crop_folder
        self.crop_all = crop_all
        self.merge_bbox = merge_bbox

        self.enable_rerank = enable_rerank
        self.enable_crop = enable_crop

        self.search_prompt = SearchPrompt()
        self.bbox_prompt = BBoxPrompt()
        self.other_prompt = OtherPrompt()

        os.makedirs(self.crop_folder, exist_ok=True)

    def _search_images(self, query):
        """Run the retrieval service for a search query."""
        try:
            response = requests.get(self.search_url, params={"queries": [query]})
            response_json = response.json()
            return [item["image_file"] for item in response_json[0]]
        except Exception as exc:
            print(f"Search failed: {exc}")
            return []

    def _process_rerank(self, query, available_images):
        """Optionally rerank candidate images and return the selected one."""
        if not available_images:
            return "No images found.", None, -1, []

        current_topk = min(len(available_images), self.topk)
        candidate_images = available_images[:current_topk]

        if not self.enable_rerank:
            return "Rerank disabled, selecting the first image.", candidate_images[0], 0, candidate_images

        candidate_images = self.get_balanced_shuffled_images(candidate_images.copy())
        rerank_think_content = self.vllm.call_qwen_rerank(
            user_prompt=self.other_prompt.rerank_think_prompt,
            image_path_list=candidate_images,
            query=query,
            topk=current_topk,
        )
        rerank_json = extract_json(rerank_think_content)

        selected_label = rerank_json.get("rerank", "0")
        selected_index = int(safe_num(selected_label.split(" ")[-1]))
        if selected_index >= len(candidate_images):
            selected_index = 0

        return rerank_json.get("think", ""), candidate_images[selected_index], selected_index, candidate_images

    def _process_image_analysis(self, query, image_path):
        """Analyze a selected image, optionally with a crop-first workflow."""
        if not image_path or not os.path.exists(image_path):
            return [], "Image not found", "Image load failed", None

        if not self.enable_crop:
            user_prompt = (
                self.other_prompt.based_image_answer_prompt
                + f"\nOur Input is:\nsearch question:{query}\nimage:"
            )
            response = self.vllm.call_qwen(user_prompt, image_path)
            response_json = extract_json(response)

            messages = [
                {
                    "role": "assistant",
                    "content": (
                        f'<think>{response_json.get("think", "")}</think>\n'
                        f'<information>{response_json.get("answer", "")}</information>'
                    ),
                }
            ]
            return messages, response_json.get("answer", ""), response_json.get("think", ""), None

        crop_result_path, crop_flag, think_before_crop, think_after_crop, answers = self.crop_or_not(
            search_query=query,
            image_path=image_path,
        )

        messages = []
        if crop_flag == "true":
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>{think_before_crop}</think>\n<bbox>{answers[0]}</bbox>",
                }
            )
            messages.append({"role": "user", "content": "The cropped image is:\n<image>"})
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>{think_after_crop}</think>\n<information>{answers[1]}</information>",
                }
            )
            return messages, answers[1], think_after_crop, crop_result_path

        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{think_before_crop}</think>\n<information>{answers[0]}</information>",
            }
        )
        return messages, answers[0], think_before_crop, None

    def execute_search_step(self, query, repeated_images):
        """Execute one full search → rerank → analyze step."""
        all_images = self._search_images(query)
        valid_images = [image for image in all_images if image not in repeated_images]

        rerank_think, selected_image_path, selected_index, topk_images = self._process_rerank(query, valid_images)

        step_messages = []
        step_images = []

        if self.enable_rerank and topk_images:
            image_slots = "\n".join([f"Image {index}:<image>" for index in range(len(topk_images))])
            image_slots = "The list of retrieved images is as follows:\n" + image_slots
            step_messages.append({"role": "user", "content": image_slots})
            step_images.append(topk_images)
            step_messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>{rerank_think}</think>\n<select>{selected_index}</select>"
                }
            )

        if not selected_image_path:
            return [], [], "No valid image found", "No image", None

        step_messages.append({"role": "user", "content": "The selected image is:\n<image>"})
        step_images.append(selected_image_path)

        analysis_messages, answer_text, think_log, crop_path = self._process_image_analysis(query, selected_image_path)
        step_messages.extend(analysis_messages)
        if crop_path:
            step_images.append(crop_path)

        return step_messages, step_images, answer_text, think_log, selected_image_path

    def cot_generation(self, sample):
        """Generate a complete CoT trajectory for a single sample."""
        step_index = 0
        steps = []
        history_messages = [{"role": "user", "content": sample["query"]}]
        history_images = []
        repeated_images = []

        query = sample["query"]
        reference_answer = sample.get("reference_answer")

        beginning = self.first_search(query)
        current_search_query = beginning["search"]
        history_messages.append(
            {
                "role": "assistant",
                "content": f"<think>{beginning['think']}</think>\n<search>{current_search_query}</search>",
            }
        )

        need_search = True
        while need_search and step_index < self.max_steps:
            new_messages, new_images, step_answer, step_think, selected_image = self.execute_search_step(
                current_search_query,
                repeated_images,
            )

            history_messages.extend(new_messages)
            for image_item in new_images:
                history_images.append(image_item)

            if selected_image:
                repeated_images.append(selected_image)

            step_index += 1
            steps.append(
                {
                    "step_i": step_index,
                    "search_query": current_search_query,
                    "search_information": step_think,
                    "answer": step_answer,
                }
            )

            judge_results = self.search_or_answer(
                query,
                self.search_prompt.search_plan_prompt,
                steps,
                reference_answer,
            )
            judge_type = judge_results.get("judge_or_search")
 
            if judge_type == "search":
                search_message = judge_results["messages"][0]
                history_messages[-1]["content"] += "\n" + search_message.get("content", "")
                try:
                    content = search_message.get("content", "")
                    current_search_query = content.split("<search>")[1].split("</search>")[0]
                except Exception:
                    print("Failed to parse the next search query.")
                    need_search = False
            elif judge_type == "answer":
                history_messages[-1]["content"] += "\n" + judge_results["messages"][0].get("content", "")
                final_answer = judge_results.get("answer")
                judge_value = judge_results.get("judge", "")

                if "True" in judge_value:
                    history_messages.append({"role": "user", "content": "The judge is YES."})
                else:
                    history_messages.append({"role": "user", "content": "The judge is NO."})
                return history_messages, history_images, final_answer

        return history_messages, history_images, "Process ended without a definitive answer."

    def crop_or_not(self, search_query, image_path):
        """Decide whether cropping is needed and return the corresponding evidence."""
        image_name = image_path.split("/")[-1].rsplit(".", 1)[0]
        layout_dir = "/".join(image_path.replace("img", "layout_vlm").split("/")[0:-1])
        json_file_path = os.path.join(layout_dir, image_name, "vlm", f"{image_name}_content_list.json")
        layout_image_path = os.path.join(layout_dir, image_name, "vlm", f"{image_name}_bbox.jpg")

        if not os.path.exists(json_file_path):
            mineru_parse_doc(
                input_image_path=[image_path],
                output_dir=layout_dir,
                backend="vlm-http-client",
                server_url=self.layout_parser_url,
            )

        with open(json_file_path, "r", encoding="utf-8") as file_obj:
            bbox_readdata = json.load(file_obj)
            bbox_list = [item["bbox"] for item in bbox_readdata]

        all_bboxes = self.bboxes_image(bbox_list, image_path, layout_image_path)
        user_prompt = self.bbox_prompt.select_roi_prompt + f"\n#Input\nQuestion:{search_query}\nImage:"
        response = self.vllm.call_qwen(user_prompt=user_prompt, image_path=layout_image_path)
        response_json = extract_json(response)
        indices = response_json["indices"]
        bboxes = [all_bboxes[index] for index in indices if index < len(all_bboxes)]

        if len(bboxes) == 0:
            user_prompt = (
                self.bbox_prompt.based_image_bbox_answer_prompt
                + f"\nOur Input is:\nsearch question:{search_query}\nimage:"
            )
            response = self.vllm.call_qwen(user_prompt, image_path, zoom=False, max_pixels=1275 * 1650)
            response_json = extract_json(response)
            bboxes = response_json["bbox"]
            think_content = response_json["think"]
            answer = response_json["answer"]

            if len(bboxes) == 0:
                return image_path, "false", think_content, "false", [answer]

            layout_bboxes = self.match_bboxes_by_iou(bboxes, all_bboxes)
            if len(layout_bboxes) == 0:
                return image_path, "false", think_content, "false", [answer]
            bboxes = layout_bboxes

        user_prompt = (
            self.bbox_prompt.reselect_roi_prompt_new
            + f"\n#Input\nQuestion:{search_query}\nbbox List:{bboxes}\nImage:"
        )
        response = self.vllm.call_qwen(user_prompt=user_prompt, image_path=image_path, zoom=False, max_pixels=1275 * 1650)
        response_json = extract_json(response)
        bboxes = response_json["bbox"]
        think_content = response_json["think"]

        if len(bboxes) == 0:
            user_prompt = (
                self.bbox_prompt.based_image_bbox_answer_prompt
                + f"\nOur Input is:\nsearch question:{search_query}\nimage:"
            )
            response = self.vllm.call_qwen(user_prompt, image_path, zoom=False, max_pixels=1275 * 1650)
            response_json = extract_json(response)
            think_content = response_json["think"]
            return image_path, "false", think_content, "false", [response_json["answer"]]

        old_bboxes = bboxes
        user_prompt = (
            self.bbox_prompt.judge_crop_prompt_2
            + f"\n#Input\nQuestion:{search_query}\nbbox List:{old_bboxes}\nImage:"
        )
        response = self.vllm.call_qwen(
            user_prompt=user_prompt,
            image_path=image_path,
            zoom=True,
            max_pixels=512 * 28 * 28,
            min_pixels=256 * 28 * 28,
        )
        response_json = extract_json(response)
        bboxes = response_json["bbox"]
        think_content = response_json["think"]

        if response_json.get("crop", "") is False:
            response = self.vllm.call_qwen(user_prompt=user_prompt, image_path=image_path, zoom=False)
            response_json = extract_json(response)
            bboxes = response_json["bbox"]
            think_content = response_json["think"]

            if response_json.get("crop", "") is False:
                return image_path, "false", think_content, "false", [response_json["answer"]]

            print("=" * 48)
            print("Cropping is required on the original image.")
            print(f"think_content: {think_content}")
            print(f"bboxes: {bboxes}")
            print("=" * 48)
            crop_img_path = self.crop_and_dump(image_path, bbox_list=bboxes, output_folder=self.crop_folder)
            user_prompt = self.bbox_prompt.cropped_prompt + f"\n#Input\nQuestion:{search_query}\nCropped Image:"
            messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image;base64," + encode_image(image_path)}}
                    ],
                }
            )
            messages.append({"role": "user", "content": [{"type": "text", "text": "\nCropped Image:"}]})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image;base64," + encode_image(crop_img_path)}}
                    ],
                }
            )
            cropped_response = self.vllm.generation(messages)
            cropped_response_json = extract_json(cropped_response)
            think_content_judge = f"{cropped_response_json['think']}"
            return crop_img_path, "true", think_content, think_content_judge, [bboxes, cropped_response_json["answer"]]

        print("=" * 48)
        print(f"think_content: {think_content}")
        print(f"bboxes: {bboxes}")
        print("=" * 48)
        crop_img_path = self.crop_and_dump(image_path, bbox_list=bboxes, output_folder=self.crop_folder)
        user_prompt = self.bbox_prompt.cropped_prompt + f"\n#Input\nQuestion:{search_query}\nCropped Image:"
        messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image;base64," + encode_image(image_path)}}],
            }
        )
        messages.append({"role": "user", "content": [{"type": "text", "text": "\nCropped Image:"}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image;base64," + encode_image(crop_img_path)}}
                ],
            }
        )
        cropped_response = self.vllm.generation(messages)
        cropped_response_json = extract_json(cropped_response)
        think_content_judge = f"{cropped_response_json['think']}"
        return crop_img_path, "true", think_content, think_content_judge, [bboxes, cropped_response_json["answer"]]

    def first_search(self, query):
        """Generate the first search query for a question."""
        content = f"Question: {query}"
        user_prompt = self.search_prompt.start_search + f"\nOur Input is:\n{content}"
        search_plan = self.vllm.call_qwen(user_prompt=user_prompt)
        return extract_json(search_plan)

    def search_or_answer(self, query, user_prompt, history_steps=None, reference_answer=None) -> dict:
        """Decide whether to continue searching or provide a final answer."""
        if history_steps is None:
            history_steps = []

        judge_results = {
            "judge_or_search": None,
            "messages": [],
            "judge": None,
            "answer": None,
        }

        content = json.dumps({"Question": query, "Steps": history_steps})
        user_prompt = user_prompt + f"\nOur Input is:\n{content}"

        next_step_judge = self.vllm.call_qwen(user_prompt=user_prompt)
        next_step_judge = extract_json(next_step_judge)

        if "answer" in next_step_judge:
            generated_answer = next_step_judge["answer"]
            judge_final = self.llm_judge(query, reference_answer, generated_answer)
            print(f"Judge: {judge_final}")
            judge_results["judge_or_search"] = "answer"
            judge_results["messages"].append(
                {
                    "role": "assistant",
                    "content": f'<think>{next_step_judge["think"]}</think>\n<answer>{next_step_judge["answer"]}</answer>',
                }
            )
            judge_results["judge"] = judge_final
            judge_results["answer"] = generated_answer

        if "search" in next_step_judge:
            judge_results["judge_or_search"] = "search"
            judge_results["messages"].append(
                {
                    "role": "assistant",
                    "content": f'<think>{next_step_judge["think"]}</think>\n<search>{next_step_judge["search"]}</search>',
                }
            )

        return judge_results

    def llm_judge(self, search_query, reference_answer, answer):
        """Judge a generated answer against the reference answer."""
        results = self.vllm.judgement(
            query=search_query,
            reference_answer=reference_answer,
            generated_answer=answer,
        )
        return json.dumps(results)

    @staticmethod
    def bboxes_image(bbox_list, image_path, save_path):
        """Render bounding boxes on an image with index labels."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        all_bboxes = []
        for idx, bbox in enumerate(bbox_list):
            x1, y1, x2, y2 = convert_bbox([bbox], reverse=False, image_path=image_path)[0]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            text = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size

            text_x = max(x1 - text_w - 10, 0)
            text_y = y1 + (y2 - y1) // 2 + text_h // 2

            cv2.putText(image, text, (int(text_x), int(text_y)), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
            all_bboxes.append(bbox)

        cv2.imwrite(save_path, image)
        return all_bboxes

    @staticmethod
    def crop_and_dump(image_path, bbox_list, output_folder):
        """Crop, stitch, resize, and save the selected regions."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

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
        output_filename = f"{base_name}_{bbox_name_str}.png"
        output_path = os.path.join(output_folder, output_filename)

        try:
            stitched.save(output_path)
            return output_path
        except Exception as exc:
            print(f"Failed to save stitched image: {exc}")
            return None

    def get_balanced_shuffled_images(self, images):
        """Return a globally balanced shuffled ordering for images of the same length."""
        num_images = len(images)
        if num_images == 0:
            return []

        with self._global_perm_lock:
            pool = self._global_perm_pool.get(num_images, [])
            if not pool:
                permutations = list(itertools.permutations(range(num_images)))
                random.shuffle(permutations)
                pool = permutations
                self._global_perm_pool[num_images] = pool

            try:
                current_order = pool.pop()
            except IndexError:
                return self.get_balanced_shuffled_images(images)

        return [images[index] for index in current_order]

    def match_bboxes_by_iou(self, bbox_list, layout_bbox_list):
        """Return all layout boxes with positive IoU against the predicted boxes."""
        if not layout_bbox_list:
            return [(bbox, None, 0.0) for bbox in bbox_list]

        matched_bboxes = []
        for bbox in bbox_list:
            for layout_bbox in layout_bbox_list:
                iou = self.calculate_iou(bbox, layout_bbox)
                if iou > 0:
                    matched_bboxes.append(layout_bbox)
        return matched_bboxes

    @staticmethod
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
