# Model Evaluation Toolkit

This directory contains a vLLM-based evaluation service for answer correctness scoring.

## Files

- `evaluator.py`: core evaluator implementation using vLLM.
- `evaluator_api.py`: FastAPI service that exposes batched evaluation.

## Environment variables

The API supports these optional environment variables:

- `EVAL_MODEL_NAME`
- `EVAL_TENSOR_PARALLEL_SIZE`
- `EVAL_GPU_MEMORY_UTILIZATION`
- `EVAL_DTYPE`
- `EVAL_BATCH_SIZE`
- `EVAL_HOST`
- `EVAL_PORT`

## Install dependencies

```bash
pip install -r requirements.txt
```

## Example request payload

```json
{
  "bs": 8,
  "prompts": [
    {
      "query": "What is the capital of France?",
      "reference_answer": "Paris",
      "generated_answer": "Paris is the capital of France."
    }
  ]
}
```

## Run the API

```bash
python evaluator_api.py
```
