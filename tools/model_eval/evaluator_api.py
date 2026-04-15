"""FastAPI wrapper for the vLLM-based answer evaluator."""

import os
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tqdm import tqdm

from evaluator import LLMGenerator


DEFAULT_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
DEFAULT_TENSOR_PARALLEL_SIZE = int(os.getenv("EVAL_TENSOR_PARALLEL_SIZE", "8"))
DEFAULT_GPU_MEMORY_UTILIZATION = float(os.getenv("EVAL_GPU_MEMORY_UTILIZATION", "0.9"))
DEFAULT_DTYPE = os.getenv("EVAL_DTYPE", "bfloat16")
DEFAULT_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "32"))
DEFAULT_HOST = os.getenv("EVAL_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("EVAL_PORT", "8003"))

app = FastAPI(
    title="Model Evaluator API",
    description="Serve batched correctness evaluation over query, reference answer, and generated answer triples.",
    version="1.0.0",
)

model_eval = None


class EvalSample(BaseModel):
    query: str
    reference_answer: str
    generated_answer: str


class EvalRequest(BaseModel):
    prompts: List[EvalSample]
    bs: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the evaluator model once at startup."""
    global model_eval
    print("Initializing evaluator service...")
    model_eval = LLMGenerator(
        model_name=DEFAULT_MODEL_NAME,
        tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
        dtype=DEFAULT_DTYPE,
    )


@app.post("/eval", response_model=List[float])
async def evaluate(request: EvalRequest) -> List[float]:
    """Evaluate a batch of prediction samples."""
    if model_eval is None:
        raise HTTPException(status_code=503, detail="Evaluator service has not finished initializing.")

    data_eval = [sample.model_dump() for sample in request.prompts]
    eval_results: List[float] = []
    for index in tqdm(range(0, len(data_eval), request.bs), desc="Evaluating batches"):
        eval_results.extend(model_eval.eval_func(data_eval[index : index + request.bs]))
    return eval_results


if __name__ == "__main__":
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
