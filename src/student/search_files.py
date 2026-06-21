import bm25s
# from student.files_loader import get_all_chunk
# from student.indexation import index_chunks


def search_match(query: str, retriever: bm25s.BM25,
                 chunks: list[dict[str, str | int]],
                 nb_of_top_match: int):
    matched_chunk = []
    query_tokens = bm25s.tokenize(query)
    results, _ = retriever.retrieve(query_tokens, k=nb_of_top_match)
    for r in results[0]:
        matched_chunk.append(chunks[r])
    return matched_chunk


# querie = "VLLM"
# chunks = get_all_chunk()
# retriever = index_chunks(chunks)
# matched = search_match(querie, retriever, chunks, 1)
# for m in matched:
#     print(m)