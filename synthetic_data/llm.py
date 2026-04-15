"""Lightweight clients for text and multimodal Qwen-compatible endpoints."""

import base64
import io
import math
import random
import time
from io import BytesIO

from PIL import Image


class Qwen_LLM_Client:
    def __init__(self, base_url="http://127.0.0.1:9009/v1", model_name="Qwen2.5-72b-Instruct"):
        self.api_key = "EMPTY"
        self.base_url = base_url
        self.model = model_name
        self.temperature = 0.0
        self.max_retries = 3
        self.timeout = 120

    def get_client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    @staticmethod
    def _retry_wait(attempt):
        return (2 ** attempt) + random.uniform(0, 1)

    def call_qwen(self, user_prompt):
        for attempt in range(self.max_retries):
            try:
                client = self.get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as exc:
                print(f"LLM call attempt {attempt + 1} failed: {exc}")
                if attempt < self.max_retries - 1:
                    wait_time = self._retry_wait(attempt)
                    print(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"All {self.max_retries} attempts failed")
                    return None

    def judgement(self, query, reference_answer, generated_answer) -> str:
        user_prompt = f"""You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

Your response should be formatted as following:
<judge>True or False</judge>

If the generated answer is correct, please set \"judge\" to True. Otherwise, please set \"judge\" to False.

Please note that the generated answer may contain additional information beyond the reference answer.
"""
        return str(self.call_qwen(user_prompt))


class Qwen_VLLM_Client:
    def __init__(self, base_url="http://127.0.0.1:9000/v1", model_name="Qwen3-VL-235B-A22B-Instruct-FP8"):
        self.api_key = "EMPTY"
        self.base_url = base_url
        self.model = model_name
        self.temperature = 0.0
        self.max_retries = 3

    def get_client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def process_image(self, image, max_pixels=64 * 28 * 28, min_pixels=32 * 28 * 28, zoom=False):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, str):
            image = Image.open(image)

        if zoom:
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

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else "JPEG")
        return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    @staticmethod
    def _retry_wait(attempt):
        return (2 ** attempt) + random.uniform(0, 1)

    def call_qwen(
        self,
        user_prompt,
        image_path=None,
        zoom=False,
        max_pixels=64 * 28 * 28,
        min_pixels=32 * 28 * 28,
        temperature=0.0,
        top_p=1.0,
    ):
        for attempt in range(self.max_retries):
            try:
                client = self.get_client()
                content = [{"type": "text", "text": user_prompt}]

                if isinstance(image_path, str):
                    base64_image = self.process_image(
                        image_path,
                        zoom=zoom,
                        max_pixels=max_pixels,
                        min_pixels=min_pixels,
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        }
                    )
                elif isinstance(image_path, list):
                    for img_path in image_path:
                        base64_image = self.process_image(img_path, zoom=zoom)
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            }
                        )

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=16384,
                )
                return response.choices[0].message.content
            except Exception as exc:
                print(f"VLLM call attempt {attempt + 1} failed: {exc}")
                if attempt < self.max_retries - 1:
                    wait_time = self._retry_wait(attempt)
                    print(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"All {self.max_retries} attempts failed")
                    return None

    def judgement(self, query, reference_answer, generated_answer) -> str:
        user_prompt = f"""You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

Your response should be formatted as following:
<judge>True or False</judge>

If the generated answer is correct, please set \"judge\" to True. Otherwise, please set \"judge\" to False.

Please note that the generated answer may contain additional information beyond the reference answer.
"""
        return str(self.call_qwen(user_prompt))

    def judgement_new(self, query, reference_answer, generated_answer) -> str | None:
        user_prompt = f"""You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the user's query
- a reference answer
- a generated answer

Your task is to evaluate the correctness of the generated answer by following these steps:

1. Extract one or more key factual points from the reference answer.
2. Determine whether the generated answer fully covers those key points.

You must strictly follow the format below for your output:
<judge>True</judge>
or
<judge>False</judge>

Important guidelines:
- Only evaluate whether the generated answer contains the key points from the reference answer.
- Additional correct information is allowed.
- Missing or contradictory information should be judged as False.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""
        return self.call_qwen(user_prompt)

    def call_qwen_rerank(self, user_prompt, image_path_list, query=None, topk=3, zoom=False):
        for attempt in range(self.max_retries):
            try:
                if not isinstance(image_path_list, list):
                    raise ValueError("image_path_list must be a list")

                client = self.get_client()
                content = [{"type": "text", "text": user_prompt}]

                if query is not None:
                    content.append({"type": "text", "text": f"The query is {query}"})

                for index, img_path in enumerate(image_path_list[:topk]):
                    base64_image = self.process_image(img_path, zoom=zoom)
                    content.append({"type": "text", "text": f"- Image {index} is"})
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        }
                    )

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=8192,
                )
                return response.choices[0].message.content
            except Exception as exc:
                print(f"VLLM call attempt {attempt + 1} failed: {exc}")
                if attempt < self.max_retries - 1:
                    wait_time = self._retry_wait(attempt)
                    print(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"All {self.max_retries} attempts failed")
                    return None

    def generation(self, messages):
        client = self.get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=8192,
        )
        return response.choices[0].message.content
