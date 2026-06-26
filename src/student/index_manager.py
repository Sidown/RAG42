import json
from student.chunker import get_all_chunk
import os
import bm25s


def save_index(path: str, retriever, chunks) -> bool:
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


# querie = "VLLM"
# chunks = get_all_chunk()
# retriever = index_chunks(chunks)
# matched = search_match(querie, retriever, chunks, 1)
# save_index("data/processed/bm25_index", retriever, chunks)


def load_index(path: str):
    chunks_path = os.path.join(os.path.dirname(path),
                               "chunks.json")
    with open(chunks_path, 'r') as f:
        json_data = json.load(f)
    bm25_index = bm25s.BM25.load(path)
    return bm25_index, json_data


def corpus_constructor(chunks: list[dict[str, str | int]]) -> list[str]:
    corpus = []
    for chunk in chunks:
        corpus.append(chunk['text'])
    return corpus


def index_chunks(chunks: list[dict[str, str | int]]) -> bm25s.BM25:
    corpus = corpus_constructor(chunks)
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever
# print(load_index("data/processed/bm25_index"))


def search_match(query: str, retriever: bm25s.BM25,
                 chunks: list[dict[str, str | int]],
                 nb_of_top_match: int):
    matched_chunk = []
    query_tokens = bm25s.tokenize(query)
    results, _ = retriever.retrieve(query_tokens, k=nb_of_top_match)
    for r in results[0]:
        matched_chunk.append(chunks[r])
    return matched_chunk