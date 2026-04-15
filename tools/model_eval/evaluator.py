"""Accelerated LLM-based answer evaluator built on top of vLLM."""

import random
import re
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_SEED = 42
DEFAULT_JUDGE_TEMPLATE = """You are an expert evaluation system for a question answering chatbot.

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


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducible inference behavior where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LLMGenerator:
    """Run batched answer evaluation with a vLLM-backed model."""

    def __init__(
        self,
        model_name: str,
        seed: int = DEFAULT_SEED,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",
        max_model_len: int = 32768,
        max_num_batched_tokens: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        set_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)

        print("=" * 50)
        print("Loading evaluation model with vLLM...")
        print(f"Model: {model_name}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")
        print(f"Dtype: {dtype}")
        print("=" * 50)

        self.llm = LLM(
            model=model_name,
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        self.sampling_params = SamplingParams(
            best_of=1,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            temperature=0.0,
            n=1,
        )

    @staticmethod
    def build_prompt(sample: Dict[str, Any]) -> str:
        """Build the evaluation prompt for a single sample."""
        return DEFAULT_JUDGE_TEMPLATE.format(
            query=str(sample["query"]),
            reference_answer=str(sample["reference_answer"]),
            generated_answer=str(sample["generated_answer"]),
        )

    def batch_generate(self, prompts: List[Dict[str, Any]]) -> List[str]:
        """Generate raw evaluator responses for a batch of samples."""
        messages_batch = [[{"role": "user", "content": self.build_prompt(prompt)}] for prompt in prompts]
        prompt_texts = self.tokenizer.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]

        outputs = self.llm.generate(prompt_texts, self.sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

    @staticmethod
    def parse_judge(response_text: str) -> float:
        """Parse a `<judge>` tag from the model response."""
        match = re.search(r"<judge>(.*?)</judge>", response_text, re.DOTALL)
        if not match:
            return 0.0
        return 1.0 if "true" in match.group(1).lower() else 0.0

    def eval_func(self, prompts: List[Dict[str, Any]]) -> List[float]:
        """Return binary evaluation scores for a batch of samples."""
        responses = self.batch_generate(prompts)
        return [self.parse_judge(response) for response in responses]


if __name__ == "__main__":
    raise SystemExit("Use `evaluator_api.py` to serve the evaluator, or import `LLMGenerator` from Python.")
