# Synthetic Data Generation

This directory contains the data-generation pipeline for multimodal chain-of-thought (CoT) traces.

## Files

- `generate_cot_data.py`: asynchronous CLI entrypoint for dataset generation.
- `cot_generator.py`: core search, rerank, analysis, and crop pipeline.
- `llm.py`: lightweight clients for Qwen-compatible text and multimodal endpoints.
- `utils_bbox.py`: bounding-box utilities used during crop selection and visualization.
- `run_example.sh`: example command for running the SlideVQA pipeline.

## Required services

The pipeline expects the following services to be available:

- a search endpoint
- a layout parsing endpoint
- a text LLM endpoint for answer judging
- a multimodal VLLM endpoint for search planning, reranking, and image analysis

Default local endpoints are defined in the scripts and can be overridden with command-line arguments or environment variables.

## Python dependencies

Install the base dependencies with:

```bash
pip install -r requirements.txt
```

## Example

Run the example script:

```bash
./run_example.sh
```

Or call the CLI directly:

```bash
python generate_cot_data.py --help
```

## Convert to LLaMA-Factory open-source data format

The raw output of `generate_cot_data.py` is a JSONL trajectory file. To use it directly in LLaMA-Factory, convert it into the ShareGPT-style multimodal format used by open-source instruction datasets:

```bash
python convert_to_llamafactory.py \
	--input-path /path/to/cot_output.jsonl \
	--output-path ../LLaMA-Factory/data/unidoc_cot.json
```
