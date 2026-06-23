import textwrap
import glob
import ast
from tqdm import tqdm


def text_chunker(text: str, file_path: str,
                 max_chunk_size: int) -> list[dict[str | int]]:
    chunks = []
    lines = text.split('#')
    last_pos = 0
    for line in lines:
        if line == '':
            continue
        if len(line) > max_chunk_size:
            line_sliced = textwrap.wrap(line, max_chunk_size)
            for s in line_sliced:
                first_char = text.find(s, last_pos)
                last_char = first_char + len(s)
                chunks.append({
                    'file': file_path,
                    'text': s,
                    'first_char_index': first_char,
                    'last_char_index': last_char
                })
                last_pos = last_char
        else:
            first_char = text.find(line, last_pos)
            last_char = first_char + len(line)
            chunks.append({
                'file': file_path,
                'text': line,
                'first_char_index': first_char,
                'last_char_index': last_char
            })
            last_pos = last_char

    return chunks


def python_code_chunker(text: str, file_path: str, max_chunk_size: int) -> list[dict[str, str | int]]:
    chunks = []
    try:
        tree = ast.parse(text)
        lines = text.split('\n')
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):     
                first_char = sum(len(lines[i]) + 1
                                for i in range(node.lineno - 1))
                last_char = sum(len(lines[i]) + 1 
                                for i in range(node.end_lineno - 1)) + len(lines[node.end_lineno - 1])
                chunks.append({
                    'file': file_path,
                    'text': '\n'.join(lines[node.lineno - 1:node.end_lineno]),
                    'first_char_index': first_char,
                    'last_char_index': last_char
                })
        return chunks
    except Exception:
        return []
            


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


def get_all_chunk(max_chunk_size) -> list[dict[str, str | int]]:
    chunks = []
    files = files_name_loader()
    for file in tqdm(files['py'], desc="Indexing .py files"):
        chunks.extend(python_code_chunker(read_file(file), file, max_chunk_size))
    for file in tqdm(files['md'], desc="Indexing .md files"):
        chunks.extend(text_chunker(read_file(file), file, max_chunk_size))
    return chunks

# print(get_all_chunk())