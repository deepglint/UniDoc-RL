"""Asynchronous entrypoint for generating CoT data."""

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import aiofiles

from cot_generator import CoT_Generator


async def append_batch_async(batch, output_path):
    """Append a batch of results to a JSONL file."""
    try:
        async with aiofiles.open(output_path, "a", encoding="utf-8") as file_obj:
            for _, item in batch:
                await file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Failed to write results: {exc}")


def load_dataset_json(json_file_path):
    """Load a dataset JSON file and unwrap the `examples` field when present."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read().strip()
            if not content:
                raise ValueError(f"File is empty: {json_file_path}")

            data = json.loads(content)
            if "examples" in data:
                data = data["examples"]
            return data
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file {json_file_path}: {exc}") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File does not exist: {json_file_path}") from exc


def create_generator(
    llm_engine_url,
    vllm_engine_url,
    search_engine_url,
    layout_parser_url,
    max_steps,
    image_topk,
    crop_folder,
    crop_all,
    merge_bbox,
    enable_rerank,
    enable_crop,
):
    """Create a configured CoT generator instance."""
    return CoT_Generator(
        llm_engine_url=llm_engine_url,
        vllm_engine_url=vllm_engine_url,
        search_engine_url=search_engine_url,
        layout_parser_url=layout_parser_url,
        max_steps=max_steps,
        image_topk=image_topk,
        crop_folder=crop_folder,
        crop_all=crop_all,
        merge_bbox=merge_bbox,
        enable_rerank=enable_rerank,
        enable_crop=enable_crop,
    )


async def main(args):
    """Run asynchronous CoT generation for the full dataset."""
    batch = []
    tasks = []

    dataset = load_dataset_json(args.data_path)
    output_json_path = args.output_path

    if os.path.exists(output_json_path):
        existing_uids = []
        with open(output_json_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_uids.append(json.loads(line)["uid"])
                except Exception:
                    continue

        dataset = [sample for sample in dataset if sample["uid"] not in existing_uids]

    print(f"Total samples: {len(dataset)}")
    indexed_dataset = [(index, sample) for index, sample in enumerate(dataset)]

    semaphore = asyncio.Semaphore(args.num_workers)
    executor = ThreadPoolExecutor(max_workers=args.num_workers)
    loop = asyncio.get_running_loop()

    if not os.path.exists(output_json_path):
        async with aiofiles.open(output_json_path, "w", encoding="utf-8"):
            pass

    def process_single(indexed_sample):
        sample_index, sample = indexed_sample
        generator = create_generator(
            args.llm_engine_url,
            args.vllm_engine_url,
            args.search_engine_url,
            args.layout_parser_url,
            args.max_steps,
            args.image_topk,
            args.crop_folder,
            args.crop_all,
            args.merge_bbox,
            args.enable_rerank,
            args.enable_crop,
        )

        try:
            history_messages, history_images, generated_answer = generator.cot_generation(sample)
            if history_messages is None:
                return sample_index, None

            return sample_index, {
                "uid": sample["uid"],
                "reference_answer": sample.get("reference_answer"),
                "generated_answer": generated_answer,
                "messages": history_messages,
                "images": history_images,
            }
        except Exception as exc:
            print(f"Error processing sample {sample_index} (UID: {sample['uid']}): {exc}")
            import traceback

            traceback.print_exc()
            return sample_index, None

    async def worker(indexed_sample):
        async with semaphore:
            return await loop.run_in_executor(executor, process_single, indexed_sample)

    for indexed_sample in indexed_dataset:
        tasks.append(worker(indexed_sample))

    start_time = time.time()
    completed_count = 0

    for task_future in asyncio.as_completed(tasks):
        result = await task_future
        completed_count += 1
        print(f"Progress: {completed_count}/{len(tasks)}")

        if result is None or result[1] is None:
            continue

        batch.append(result)
        if len(batch) >= args.batch_size:
            await append_batch_async(batch, output_json_path)
            batch = []

    if batch:
        await append_batch_async(batch, output_json_path)

    print(f"Completed in {time.time() - start_time:.2f} seconds.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate CoT data asynchronously.")

    parser.add_argument("--data-path", type=str, required=True, help="Path to the input JSON dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file.")

    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for buffered writes.")

    parser.add_argument(
        "--llm-engine-url",
        type=str,
        default="http://127.0.0.1:9009/v1",
        help="Base URL for the judge LLM service.",
    )
    parser.add_argument(
        "--vllm-engine-url",
        type=str,
        default="http://127.0.0.1:9000/v1",
        help="Base URL for the multimodal VLLM service.",
    )
    parser.add_argument(
        "--search-engine-url",
        type=str,
        default="http://127.0.0.1:8200/search",
        help="Search service endpoint.",
    )
    parser.add_argument(
        "--layout-parser-url",
        type=str,
        default="http://127.0.0.1:30000",
        help="Layout parsing service endpoint.",
    )
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum number of search steps.")
    parser.add_argument("--image-topk", type=int, default=3, help="Number of candidate images to inspect.")
    parser.add_argument(
        "--crop-folder",
        type=str,
        default="./outputs/crops",
        help="Directory used to store cropped images.",
    )
    parser.add_argument("--crop-all", action="store_true", help="Crop every region candidate.")
    parser.add_argument("--merge-bbox", action="store_true", help="Enable bounding-box merging.")
    parser.add_argument("--enable-rerank", action="store_true", help="Enable image reranking.")
    parser.add_argument("--enable-crop", action="store_true", help="Enable region cropping.")

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
