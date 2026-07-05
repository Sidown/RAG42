from sentence_transformers import SentenceTransformer, util
from student.chunker import Chunk
import numpy as np
import torch


class SemanticIndexing():

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def build(self, chunks: list[Chunk]) -> None:
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
        index = np.load('./data/processed/semantic_index.npy')
        query_embeddings = self.model.encode(query)
        scores = util.cos_sim(query_embeddings, index)
        idx = torch.argsort(scores[0], descending=True).numpy()
        return idx[:k]

