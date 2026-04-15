"""Dataset ingestion utilities for building search-engine node files."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleFileNodeParser
from vl_embedding import VLEmbedding


class DatasetIngestion:
    """Build image node files with multimodal embeddings for retrieval."""

    def __init__(
        self,
        dataset_dir,
        input_prefix="img",
        output_prefix="colqwen_ingestion",
        embed_model_name="vidore/colqwen2-v1.0",
        device="cuda:0",
        workers=5,
    ):
        self.dataset_dir = dataset_dir
        self.input_dir = os.path.join(dataset_dir, input_prefix)
        self.output_dir = os.path.join(dataset_dir, output_prefix)
        self.workers = workers
        self.reader = SimpleDirectoryReader(input_dir=self.input_dir)
        self.pipeline = IngestionPipeline(
            transformations=[SimpleFileNodeParser(), VLEmbedding(model=embed_model_name, mode="image", device=device)]
        )

    def ingest_file(self, input_file, output_file):
        """Ingest a single file and persist the resulting nodes."""
        documents = self.reader.load_file(Path(input_file), self.reader.file_metadata, self.reader.file_extractor)
        nodes = self.pipeline.run(documents=documents, num_workers=1, show_progress=False)
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump([node.to_dict() for node in nodes], json_file, indent=2, ensure_ascii=False)
        return True

    def ingest_dataset(self):
        """Ingest all files under the input directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        files_to_process = []
        for file_name in os.listdir(self.input_dir):
            file_prefix, _ = os.path.splitext(file_name)
            input_file = os.path.join(self.input_dir, file_name)
            output_file = os.path.join(self.output_dir, f"{file_prefix}.node")
            if not os.path.exists(output_file):
                files_to_process.append((input_file, output_file))

        if self.workers == 1:
            for input_file, output_file in tqdm(files_to_process, desc="Ingesting files"):
                self.ingest_file(input_file, output_file)
            return

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.ingest_file, input_file, output_file): (input_file, output_file)
                for input_file, output_file in files_to_process
            }
            for future in tqdm(as_completed(futures), total=len(files_to_process), desc="Ingesting files"):
                future.result()


Ingestion = DatasetIngestion


def parse_args():
    """Parse command-line arguments for dataset ingestion."""
    parser = argparse.ArgumentParser(description="Build node files for the multimodal search engine.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--input_prefix", type=str, default="img", help="Input folder name relative to the dataset directory.")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="colqwen_ingestion",
        help="Output folder name relative to the dataset directory.",
    )
    parser.add_argument(
        "--embed_model_name",
        type=str,
        default="vidore/colqwen2-v1.0",
        help="Embedding model name.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used for embedding generation.")
    parser.add_argument("--workers", type=int, default=5, help="Number of worker threads.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingestion = DatasetIngestion(
        dataset_dir=args.dataset_dir,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        embed_model_name=args.embed_model_name,
        device=args.device,
        workers=args.workers,
    )
    ingestion.ingest_dataset()
