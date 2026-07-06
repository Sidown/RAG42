from sentence_transformers import SentenceTransformer, util
from src.chunker import Chunk
import numpy as np
import torch
from typing import Any


class SemanticIndexing():
    """
    Semantic indexing class
    """
    INDEX_PATH = "./data/processed/semantic_index.npy"

    def __init__(self) -> None:
        """
        init the model used
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def build(self, chunks: list[Chunk]) -> None:
        """
        Encode all chunks and save the embedding matrix to disk.

        Args:
            chunks: List of Chunk dicts to encode and index.
        """
        chunk_text_list = []
        for chunk in chunks:
            chunk_text_list.append(chunk['text'])
        chunk_embeddings = self.model.encode(
            chunk_text_list,
            show_progress_bar=True,
            batch_size=256
            )
        np.save('./data/processed/semantic_index', chunk_embeddings)

    def search(self, query: str, k: int) -> list[int]:
        """
        Find the top-k most semantically similar chunks for a query.

        Args:
            query: The user query string.
            k: Number of top results to return.

        Returns:
            A list of integer indices into the original chunks list,
            sorted by descending cosine similarity.
        """
        index = np.load(self.INDEX_PATH)
        query_embeddings = self.model.encode(query)
        scores = util.cos_sim(query_embeddings, index)
        sorted_i = torch.argsort(scores[0], descending=True).numpy()
        return sorted_i[:k].tolist()
