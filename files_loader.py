import textwrap
import glob
import ast


def text_chunker(text: str) -> list[str]:
    chunks = []
    chunks = textwrap.wrap(text, 2000)
    return chunks


# def python_code_chunker(text: str) -> list[str]:
#     chunks = []
#     chunks = text.split('def')
#     for chunk in chunks:
#         if chunk == '':
#             chunks.remove(chunk)
#     return chunks

def python_code_chunker(text: str) -> dict[int, str]:
    tree = ast.parse(text)
    d = {}
    lines = text.split('\n')
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):     
            first_char = sum(len(lines[i]) + 1
                            for i in range(node.lineno - 1))
            last_char = sum(len(lines[i]) + 1
                            for i in range(node.end_lineno - 1))
            print(f"f: {first_char}, l: {last_char}")
            d.update({first_char:
                      '\n'.join(lines[node.lineno - 1:node.end_lineno])})
    return d
            


def files_name_loader() -> dict[str, list[str]]:
    repo = './test_file/'
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
        chunks.extend(python_code_chunker(read_file(file)))
    for file in files['txt']:
        chunks.extend(text_chunker(read_file(file)))
    for file in files['md']:
        chunks.extend(text_chunker(read_file(file)))
    return chunks

chunks = get_all_chunk()


# import bm25s
# corpus = [
#     "a cat is a feline and likes to purr",
#     "a dog is the human's best friend and loves to play",
#     "a bird is a beautiful animal that can fly",
#     "a bird eat grains",
#     "a bird can be of many colors",
#     "the cat eat the bird"
# ]
# print('ici')
# corpus_tokens = bm25s.tokenize(corpus)
# retriever = bm25s.BM25(corpus=corpus)
# retriever.index(corpus_tokens)

# query = "what is a bird?"
# quer_tokens = bm25s.tokenize(query)
# docs, scores = retriever.retrieve(quer_tokens, k=2)
# print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")