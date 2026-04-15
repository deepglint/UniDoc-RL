# Search Engine Toolkit

This directory contains the multimodal retrieval components used by the project.

## Files

- `vl_embedding.py`: embedding wrapper for ColQwen, ColPali, and compatible OpenBMB models.
- `ingestion.py`: builds serialized node files from a dataset image directory.
- `search_engine.py`: loads node files and serves batched retrieval.
- `search_engine_api.py`: FastAPI service that exposes the retrieval endpoint.

## Environment variables for the API

The API server reads the following optional environment variables:

- `SEARCH_DATASET_DIR`
- `SEARCH_IMAGE_DIRNAME`
- `SEARCH_NODE_DIR_PREFIX`
- `SEARCH_EMBED_MODEL`
- `SEARCH_DEVICE`
- `SEARCH_TOP_K`
- `SEARCH_HOST`
- `SEARCH_PORT`

## Install dependencies

```bash
pip install -r requirements.txt
```

The current `requirements.txt` is trimmed to the direct dependencies used in this folder and pinned to versions available in the `vrag` conda environment.

## Example usage

Build node files:

```bash
python ingestion.py --dataset_dir /path/to/dataset
```

Start the API server:

```bash
python search_engine_api.py
```
