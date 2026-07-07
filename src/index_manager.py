import json
import os
import bm25s
from src.chunker import Chunk
from src.semantic_embeddings import SemanticIndexing
import re


def _split_identifier(token: str) -> str:
    """
    Insert spaces at snake_case and camelCase boundaries so a
    sub-word becomes matchable on its own by BM25.
    """
    s = token.replace('_', ' ')
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', s)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)
    return s


def expand_identifiers(text: str) -> str:
    """
    Append space-split versions of snake_case/camelCase identifiers
    to a text, so BM25 can also match on their sub-words, not only
    on the identifier verbatim.
    """
    tokens = re.findall(r'\w+', text)
    extra = []
    for tok in tokens:
        if '_' in tok or re.search(r'[a-z][A-Z]', tok):
            split_version = _split_identifier(tok)
            if split_version != tok:
                extra.append(split_version)
    if extra:
        return text + ' ' + ' '.join(extra)
    return text


def save_index(path: str, retriever: bm25s.BM25,
               chunks: list[Chunk]) -> bool:
    """
    Save the BM25 index and chunks to disk.

    Args:
        path: Directory path for the BM25 index.
        retriever: The fitted BM25 retriever object.
        chunks: List of all chunks to save alongside the index.

    Returns:
        True on success, False on error.
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


def load_index(path: str) -> tuple[bm25s.BM25, list[Chunk]]:
    """
    Load the BM25 index and chunks from disk.

    Args:
        path: Directory path of the saved BM25 index.

    Returns:
        A tuple of (BM25 retriever, list of chunks).
    """
    chunks_path = os.path.join(os.path.dirname(path),
                               "chunks.json")
    with open(chunks_path, 'r') as f:
        json_data: list[Chunk] = json.load(f)
    bm25_index = bm25s.BM25.load(path)
    return bm25_index, json_data


def corpus_constructor(chunks: list[Chunk]) -> list[str]:
    """
    Extract text content from chunks to build a searchable corpus.

    Args:
        chunks: List of Chunk dicts.

    Returns:
        A list of text strings, one per chunk.
    """
    corpus = []
    for chunk in chunks:
        corpus.append(expand_identifiers(chunk['text']))
    return corpus


def build_bm25_index(chunks: list[Chunk]) -> bm25s.BM25:
    """
    Build and return a BM25 index from a list of chunks.

    Args:
        chunks: List of Chunk dicts to index.

    Returns:
        A fitted BM25 retriever object.
    """
    corpus = corpus_constructor(chunks)
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever


def bm25_search(query: str, retriever: bm25s.BM25,
                chunks: list[Chunk],
                nb_of_top_match: int) -> list[Chunk]:
    """
    Search the BM25 index and return the top-k most relevant chunks.

    Args:
        query: The user query string.
        retriever: The fitted BM25 retriever.
        chunks: The list of all indexed chunks.
        nb_of_top_match: Number of top results to return.

    Returns:
        A list of the top-k matching chunks.
    """
    matched_chunks = []
    query_tokens = bm25s.tokenize(expand_identifiers(query))
    results, _ = retriever.retrieve(query_tokens, k=nb_of_top_match)
    for chunk_idx in results[0]:
        matched_chunks.append(chunks[chunk_idx])
    return matched_chunks


def rrf_search(query: str, retriever: bm25s.BM25,
               semantic: SemanticIndexing,
               chunks: list[Chunk], k: int) -> list[Chunk]:
    """
    Hybrid search combining BM25 and semantic retrieval via RRF.

    Retrieves candidate_k=k*3 results from each system, then fuses
    their rankings using Reciprocal Rank Fusion (RRF, k=60).

    Args:
        query: The user query string.
        retriever: The fitted BM25 retriever.
        semantic: A SemanticIndexing instance with a loaded index.
        chunks: The list of all indexed chunks.
        k: Number of final results to return.

    Returns:
        A list of the top-k chunks ranked by RRF score.
    """
    candidate_k = k * 3
    bm25_weight = 0.6
    scores: dict[int, float] = {}

    query_tokens = bm25s.tokenize(expand_identifiers(query))
    bm25_results, _ = retriever.retrieve(query_tokens, k=candidate_k)

    # RRF formula: score = 1 / (k + rank),
    # k=60 prevents top results from dominating
    for rank, chunk_idx in enumerate(bm25_results[0]):
        scores[chunk_idx] = (scores.get(chunk_idx, 0)
                             + bm25_weight
                             * (1 / (60 + rank)))

    semantic_results = semantic.search(query, candidate_k)
    for rank, chunk_idx in enumerate(semantic_results):
        scores[chunk_idx] = (scores.get(chunk_idx, 0)
                             + (1 - bm25_weight)
                             * (1 / (60 + rank)))

    sorted_indices = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunks[i] for i in sorted_indices[:k]]
