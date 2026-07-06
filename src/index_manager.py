import json
import os
import bm25s
from src.chunker import Chunk
from src.semantic_embeddings import SemanticIndexing


def save_index(path: str, retriever: bm25s.BM25,
               chunks: list[Chunk]) -> bool:
    """
    Save the BM25 index as a json file
    path: path of the json file
    retriever: BM25 index
    chunks: dict containing all the chunks
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        retriever.save(path)
        chunks_path = os.path.join(os.path.dirname(path),
                                   "chunks.json")
        with open(chunks_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        return True
    except Exception as e:
        print(e)
        return False


def load_index(path: str) -> tuple[bm25s.BM25, dict]:
    """
    Load bm25 index and the json file then return them
    path: path of the file
    """
    chunks_path = os.path.join(os.path.dirname(path),
                               "chunks.json")
    with open(chunks_path, 'r') as f:
        json_data: dict = json.load(f)
    bm25_index = bm25s.BM25.load(path)
    return bm25_index, json_data


def corpus_constructor(chunks: list[Chunk]) -> list[str]:
    """
    Construct corpus and return it
    """
    corpus = []
    for chunk in chunks:
        corpus.append(chunk['text'])
    return corpus


def build_bm25_index(chunks: list[Chunk]) -> bm25s.BM25:
    """
    Index the chunks using BM25
    """
    corpus = corpus_constructor(chunks)
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever


def bm25_search (query: str, retriever: bm25s.BM25,
                 chunks: dict[str, str | int],
                 nb_of_top_match: int) -> list[str | int]:
    """
    Use bm25 to search match for the query and return a list
    of top matched chunk.
    query: User query
    retriever: BM25 retriever
    chunks: dict of chunk
    nb_of_top_match: number of chunk to return
    """
    results = []
    query_tokens = bm25s.tokenize(query)
    results, _ = retriever.retrieve(query_tokens, k=nb_of_top_match)
    for chunk_idx in results[0]:
        results.append(chunks[chunk_idx])
    return results


def rrf_search(query: str, retriever: bm25s.BM25,
               semantic: SemanticIndexing,
               chunks: list[Chunk], k: int) -> list[Chunk]:
    """
    Use BM25 and semantic indexation to do an hybrid search of matched chunks,
    and use Reciprocal Rank Fusion (RRF) algorithm to sort them.
    Return a list of k matched chunks.
    query: User query
    retriever: BM25 retriever
    semantic: SemanticIndexing class
    chunks: list of Chunk class
    k: number of result to return
    """
    candidate_k = k * 3
    scores: dict[int, float] = {}

    query_tokens = bm25s.tokenize(query)
    bm25_results, _ = retriever.retrieve(query_tokens, k=candidate_k)
    
    # RRF formula: score = 1 / (k + rank), k=60 prevents top results from dominating
    for rank, chunk_idx in enumerate(bm25_results[0]):
        scores[chunk_idx] = scores.get(chunk_idx, 0) + 1 / (60 + rank)

    semantic_results = semantic.search(query, candidate_k)
    for rank, chunk_idx in enumerate(semantic_results):
        scores[chunk_idx] = scores.get(chunk_idx, 0) + 1 / (60 + rank)

    sorted_indices = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunks[i] for i in sorted_indices[:k]]
