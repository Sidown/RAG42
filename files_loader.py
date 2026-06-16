import textwrap
import glob


def text_chunker(text: str) -> list[str]:
    chunks = []
    chunks = textwrap.wrap(text, 2000)
    return chunks


def python_code_chunker(text: str) -> list[str]:
    chunks = []
    chunks = text.split('def')
    for chunk in chunks:
        if chunk == '':
            chunks.remove(chunk)
    return chunks


def files_name_loader() -> dict[str, list[str]]:
    repo = './vllm-0.10.1/'
    files_dict = {}
    py_files = glob.glob(repo + '/**/*.py', recursive=True)
    files_dict.update({'py': py_files})
    txt_file =  glob.glob(repo + '/**/*.txt', recursive=True)
    files_dict.update({'txt': txt_file})
    md_file = glob.glob(repo + '/**/*.md', recursive=True)
    files_dict.update({'md': md_file})
    return files_dict


def read_file(file: str) -> str:
    with open(file) as f:
        text = f.read()
    return text


def get_all_chunk() -> list[str]:
    chunks = []
    files = files_name_loader()
    for file in files['py']:
        chunks.append(python_code_chunker(read_file(file)))
    for file in files['txt']:
        chunks.append(text_chunker(read_file(file)))
    for file in files['md']:
        chunks.append(text_chunker(read_file(file)))
    return chunks

chunks = get_all_chunk()


import bm25s
corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a bird eat grains",
    "a bird can be of many colors",
    "the cat eat the bird"
]
print('ici')
corpus_tokens = bm25s.tokenize(corpus)
retriever = bm25s.BM25(corpus=corpus)
retriever.index(corpus_tokens)

query = "what is a bird?"
quer_tokens = bm25s.tokenize(query)
docs, scores = retriever.retrieve(quer_tokens, k=2)
print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")