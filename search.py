import bm25s


def search_match(querie: str, retriever: bm25s.BM25,
                 chunks: list[dict[str, str | int]],
                 nb_of_top_match: int = 5):
    matched_chunk = []
    query_tokens = bm25s.tokenize(querie)
    results, _ = retriever.retrieve(query_tokens, k=nb_of_top_match)
    for r in results[0]:
        matched_chunk.append(chunks[r])
    return matched_chunk