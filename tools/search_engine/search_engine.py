"""Core search engine implementation for image retrieval."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
from tqdm import tqdm

from llama_index.core.schema import ImageNode, TextNode
from vl_embedding import VLEmbedding


LOGGER = logging.getLogger(__name__)


def nodefile2node(input_file):
    """Load llama-index nodes from a serialized `.node` JSON file."""
    nodes = []
    try:
        with open(input_file, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
            if not isinstance(data, list):
                LOGGER.warning("Skipping non-list node file: %s", input_file)
                return []

            for doc in data:
                class_name = doc.get("class_name")
                if class_name == "TextNode" and doc.get("text"):
                    nodes.append(TextNode.from_dict(doc))
                elif class_name == "ImageNode":
                    nodes.append(ImageNode.from_dict(doc))
    except json.JSONDecodeError:
        LOGGER.error("Invalid or empty JSON file: %s", input_file)
    except FileNotFoundError:
        LOGGER.error("Node file not found: %s", input_file)
    except Exception as exc:
        LOGGER.error("Unexpected error while loading %s: %s", input_file, exc)
    return nodes


class SearchEngine:
    """Load precomputed node files and serve top-k retrieval over image embeddings."""

    def __init__(
        self,
        dataset_dir="search_engine/corpus",
        node_dir_prefix="colqwen_ingestion",
        embed_model_name="vidore/colqwen2-v1.0",
        device_map="cuda:0",
        top_k=10,
        max_workers=10,
    ):
        self.dataset_dir = dataset_dir
        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.top_k = top_k
        self.max_workers = max_workers
        self.vector_embed_model = VLEmbedding(model=embed_model_name, mode="image", device=device_map)
        self.nodes = []
        self.embedding_img = []
        self.image_count = 0
        self._load_index()

    def load_nodes(self):
        """Load all serialized node files under the configured node directory."""
        files = os.listdir(self.node_dir)
        parsed_nodes = []

        def parse_file(file_name):
            input_file = os.path.join(self.node_dir, file_name)
            if not input_file.endswith(".node"):
                return []
            return nodefile2node(input_file)

        if self.max_workers == 1:
            for file_name in tqdm(files, desc="Loading node files"):
                parsed_nodes.extend(parse_file(file_name))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(tqdm(executor.map(parse_file, files), total=len(files), desc="Loading node files"))
            for result in results:
                parsed_nodes.extend(result)

        return parsed_nodes

    def _load_index(self):
        """Load nodes and move their embeddings onto the target device."""
        LOGGER.info("Loading nodes from %s", self.node_dir)
        self.nodes = self.load_nodes()
        self.embedding_img = [
            torch.tensor(node.embedding).view(-1, 128).bfloat16() for node in tqdm(self.nodes, desc="Preparing embeddings")
        ]
        self.embedding_img = [
            tensor.to(self.vector_embed_model.embed_model.device)
            for tensor in tqdm(self.embedding_img, desc="Moving embeddings to device")
        ]
        self.image_count = len(self.embedding_img)

    def batch_search(self, queries: List[str]):
        """Search the index for a batch of queries and return ranked image file names."""
        batch_queries = self.vector_embed_model.processor.process_queries(queries).to(
            self.vector_embed_model.embed_model.device
        )
        with torch.no_grad():
            query_embeddings = self.vector_embed_model.embed_model(**batch_queries)

        scores = self.vector_embed_model.processor.score_multi_vector(
            query_embeddings,
            self.embedding_img,
            batch_size=256,
            device=self.vector_embed_model.embed_model.device,
        )
        _, indices = torch.topk(scores, k=min(self.image_count, self.top_k), dim=1)
        return [[self.nodes[idx].metadata["file_name"] for idx in row] for row in indices]


if __name__ == "__main__":
    engine = SearchEngine(dataset_dir="search_engine/corpus", embed_model_name="vidore/colqwen2-v1.0")
    print(engine.batch_search(["o", "a"]))
