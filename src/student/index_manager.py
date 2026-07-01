import json
import os
import bm25s


def save_index(path: str, retriever: bm25s.BM25,
               chunks: list[dict[str, str | int]]) -> bool:
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


# querie = "VLLM"
# chunks = get_all_chunk()
# retriever = index_chunks(chunks)
# matched = search_match(querie, retriever, chunks, 1)
# save_index("data/processed/bm25_index", retriever, chunks)


def load_index(path: str) -> list[bm25s.BM25, dict]:
    """
    Load bm25 index and the json file then return them
    path: path of the file
    """
    try:
        chunks_path = os.path.join(os.path.dirname(path),
                                   "chunks.json")
        with open(chunks_path, 'r') as f:
            json_data = json.load(f)
        bm25_index = bm25s.BM25.load(path)
        return bm25_index, json_data
    except Exception:
        pass


def corpus_constructor(chunks: list[dict[str, str | int]]) -> list[str]:
    """
    Construct corpus and return it
    """
    corpus = []
    for chunk in chunks:
        corpus.append(chunk['text'])
    return corpus


def index_chunks(chunks: list[dict[str, str | int]]) -> bm25s.BM25:
    """
    Index the chunks using BM25
    """
    corpus = corpus_constructor(chunks)
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever
# print(load_index("data/processed/bm25_index"))


def search_match(query: str, retriever: bm25s.BM25,
                 chunks: list[dict[str, str | int]],
                 nb_of_top_match: int) -> list[dict[str, str | int]]:
    matched_chunk = []
    query_tokens = bm25s.tokenize(query)
    results, _ = retriever.retrieve(query_tokens, k=nb_of_top_match)
    for r in results[0]:
        matched_chunk.append(chunks[r])
    return matched_chunk
