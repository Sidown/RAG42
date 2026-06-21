import bm25s
# from student.files_loader import get_all_chunk


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

# print(index_chunks(get_all_chunk()))
