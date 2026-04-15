"""FastAPI service for the multimodal search engine."""

import os
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from search_engine import SearchEngine


DATASET_DIR = os.getenv("SEARCH_DATASET_DIR", "./corpus")
IMAGE_DIRNAME = os.getenv("SEARCH_IMAGE_DIRNAME", "img")
NODE_DIR_PREFIX = os.getenv("SEARCH_NODE_DIR_PREFIX", "colqwen_ingestion")
EMBED_MODEL_NAME = os.getenv("SEARCH_EMBED_MODEL", "vidore/colqwen2-v1.0")
DEVICE = os.getenv("SEARCH_DEVICE", "cuda:0")
TOP_K = int(os.getenv("SEARCH_TOP_K", "10"))
HOST = os.getenv("SEARCH_HOST", "0.0.0.0")
PORT = int(os.getenv("SEARCH_PORT", "8005"))

app = FastAPI(
    title="Hybrid Search Engine API",
    description="Serve batched image retrieval queries over a prebuilt multimodal index.",
    version="1.0.0",
)

search_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize the search engine when the API starts."""
    global search_service
    print(f"Mounted dataset directory: {DATASET_DIR}")
    print("Initializing SearchEngine...")
    search_service = SearchEngine(
        dataset_dir=DATASET_DIR,
        embed_model_name=EMBED_MODEL_NAME,
        node_dir_prefix=NODE_DIR_PREFIX,
        device_map=DEVICE,
        top_k=TOP_K,
    )


@app.get(
    "/search",
    summary="Perform a search query.",
    description="Run the initialized search engine for a batch of queries.",
    response_model=List[List[Dict[str, Any]]],
)
async def search(queries: List[str] = Query(...)):
    """Run batched retrieval and return resolved image paths."""
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search engine has not finished initializing.")

    results_batch = search_service.batch_search(queries)
    image_dir = os.path.join(DATASET_DIR, IMAGE_DIRNAME)
    return [
        [
            {"idx": idx, "image_file": os.path.join(image_dir, file_name)}
            for idx, file_name in enumerate(query_results)
        ]
        for query_results in results_batch
    ]


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
