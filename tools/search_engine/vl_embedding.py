"""Vision-language embedding wrappers for retrieval and ingestion."""

from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings import MultiModalEmbedding


def weighted_mean_pooling(hidden_states, attention_mask):
    """Apply weighted mean pooling over the sequence dimension."""
    weighted_mask = attention_mask * attention_mask.cumsum(dim=1)
    numerator = torch.sum(hidden_states * weighted_mask.unsqueeze(-1).float(), dim=1)
    denominator = weighted_mask.sum(dim=1, keepdim=True).float()
    return numerator / denominator


class VLEmbedding(MultiModalEmbedding):
    """A multimodal embedding adapter for ColQwen, ColPali, and OpenBMB models."""

    model: str = Field(description="The embedding model name.")
    api_key: Optional[str] = Field(default=None, description="Unused API key placeholder.")
    dimensions: Optional[int] = Field(default=1024, description="Output embedding dimension when applicable.")
    timeout: Optional[float] = Field(default=None, description="Unused timeout placeholder.")
    mode: str = Field(default="text", description="Embedding mode: `text` or `image`.")
    show_progress: bool = Field(default=False, description="Whether to show progress bars.")
    embed_model: Union[ColQwen2, AutoModel, ColPali, None] = Field(default=None)
    processor: Optional[Union[ColQwen2Processor, ColPaliProcessor]] = Field(default=None)
    tokenizer: Optional[AutoTokenizer] = Field(default=None)

    def __init__(
        self,
        model: str = "vidore/colqwen2-v1.0",
        dimensions: Optional[int] = 1024,
        timeout: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        mode: str = "text",
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )
        self.mode = mode
        self._load_model(model=model, device=device)

    def _load_model(self, model: str, device: str) -> None:
        """Load the configured embedding model and processor."""
        if "openbmb" in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self.embed_model = AutoModel.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device,
            ).eval()
        elif "vidore" in model and "qwen" in model:
            self.embed_model = ColQwen2.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map=device,
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained(model)
        elif "vidore" in model and "pali" in model:
            self.embed_model = ColPali.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map=device,
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model)
        else:
            raise ValueError(f"Unsupported embedding model: {model}")

    @classmethod
    def class_name(cls) -> str:
        return "VLEmbedding"

    def embed_img(self, image_paths):
        """Generate image embeddings for one or more image files."""
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if "vidore" in self.model:
            images = [Image.open(path) for path in image_paths]
            batch_images = self.processor.process_images(images).to(self.embed_model.device)
            with torch.no_grad():
                image_embeddings = self.embed_model(**batch_images)
            return image_embeddings

        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = {"text": [""] * len(images), "image": images, "tokenizer": self.tokenizer}
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            pooled = weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
            return F.normalize(pooled, p=2, dim=1).detach().cpu().numpy()

    def embed_text(self, text):
        """Generate text embeddings for one or more query strings."""
        if isinstance(text, str):
            text = [text]

        if "colqwen" in self.model or "colpali" in self.model:
            batch_queries = self.processor.process_queries(text).to(self.embed_model.device)
            with torch.no_grad():
                return self.embed_model(**batch_queries)

        instruction = "Represent this query for retrieving relevant documents: "
        queries = [instruction + query for query in text]
        inputs = {"text": queries, "image": [None] * len(queries), "tokenizer": self.tokenizer}
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            pooled = weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
            return F.normalize(pooled, p=2, dim=1).detach().cpu().tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.embed_text(query)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.embed_text(text)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embed_text(texts)
        if hasattr(embeddings, "view"):
            return embeddings.view(embeddings.size(0), -1).tolist()
        return embeddings

    def _aget_query_embedding(self, query: str) -> List[float]:
        return self.embed_text(query)[0]

    def _aget_text_embedding(self, text: str) -> List[float]:
        return self.embed_text(text)[0]

    def _get_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)

    def _aget_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)

    def __call__(self, nodes, **kwargs):
        """Attach computed embeddings to llama-index nodes."""
        if "vidore" in self.model:
            if self.mode == "image":
                embeddings = self.embed_img([node.metadata["file_path"] for node in nodes])
            else:
                embeddings = self.embed_text([node.text for node in nodes])
            embeddings = embeddings.view(embeddings.size(0), -1).tolist()
        else:
            if self.mode == "image":
                embeddings = self.embed_img([node.metadata["file_path"] for node in nodes]).tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return nodes

    def score(self, image_embeddings, text_embeddings):
        """Compute matching scores between image and text embeddings."""
        if "vidore" in self.model:
            return self.processor.score_multi_vector(image_embeddings, text_embeddings)
        return text_embeddings @ image_embeddings.T


VL_Embedding = VLEmbedding


if __name__ == "__main__":
    embedding_model = VLEmbedding("vidore/colqwen2-v1.0")
    image_embeddings = embedding_model.embed_img("./search_engine/img/sample.jpg")
    text_embeddings = embedding_model.embed_text("Hello, world!")
    score = embedding_model.processor.score_multi_vector(image_embeddings, text_embeddings)
    print(score)
