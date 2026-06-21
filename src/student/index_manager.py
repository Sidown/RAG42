import json
from student.files_loader import get_all_chunk
from student.indexation import index_chunks
from student.search_files import search_match
import os
import bm25s


def save_index(path: str, retriever, chunks) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        retriever.save(path)
        chunks_path = os.path.join(os.path.dirname(path),
                                   "chunks.json")
        with open(chunks_path, 'w') as f:
            json.dump(chunks, f)
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


print(load_index("data/processed/bm25_index"))