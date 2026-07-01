import textwrap
import glob
import ast
from tqdm import tqdm


def text_chunker(text: str, file_path: str,
                 max_chunk_size: int) -> list[dict[str | int]]:
    """
    Chunk a text with a split on '#' and a wrap if one of the chunks
    is longer than the max_chunk_size.
    text: the text to chunk
    file_path: path of the text
    max_chunk_size: The len max of each chunks
    return a list of dict with file, text, first_char_index,
    last_char_index as keys.
    """
    chunks = []
    lines = text.split('#')
    last_pos = 0
    try:
        for line in lines:
            if line == '':
                continue

            first_char_of_line = text.find(line, last_pos)

            if len(line) > max_chunk_size:
                line_sliced = textwrap.wrap(line, max_chunk_size)
                offset = first_char_of_line
                for s in line_sliced:
                    first_char = offset
                    last_char = first_char + len(s)
                    chunks.append({
                        'file': file_path,
                        'text': s,
                        'first_char_index': first_char,
                        'last_char_index': last_char
                    })
                    offset = last_char
                last_pos = offset

            else:
                first_char = first_char_of_line
                last_char = first_char + len(line)
                chunks.append({
                    'file': file_path,
                    'text': line,
                    'first_char_index': first_char,
                    'last_char_index': last_char
                })
                last_pos = last_char

        return chunks

    except Exception:
        return []


def python_code_chunker(text: str, file_path: str,
                        max_chunk_size: int) -> list[dict[str, str | int]]:
    """
    Chunk a python code on every class and def using ast
    text: the python code to chunk
    file_path: path of the file containing the text
    max_chunk_size: number max of char in a chunk
    return a list of dict containing file, text, first_char_index,
    last_char_index as keys.
    """
    chunks = []
    try:
        tree = ast.parse(text)
        lines = text.split('\n')
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                first_char = sum(len(lines[i]) + 1
                                 for i in range(node.lineno - 1))
                last_char = (sum(len(lines[i]) + 1
                                 for i in range(node.end_lineno - 1))
                             + len(lines[node.end_lineno - 1]))
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
    """
    Charge all the .md and .py file in the 'data/raw/vllm-0.10.1/' folder.
    Return a dict with py and md as keys, and the corresponding list
    of files path for each keys
    """
    repo = 'data/raw/vllm-0.10.1/'
    files_dict = {}
    try:
        py_files = glob.glob(repo + '**/*.py', recursive=True)
        files_dict.update({'py': [f.replace('\\', '/') for f in py_files]})
        md_file = glob.glob(repo + '**/*.md', recursive=True)
        files_dict.update({'md': [f.replace('\\', '/') for f in md_file]})
        return files_dict
    except Exception:
        return {"py": [''],
                "md": ['']}


def read_file(file: str) -> str:
    """
    Read a file and return it's text
    file: path of the file to read
    """
    try:
        with open(file, encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception:
        return ''


def get_all_chunk(max_chunk_size) -> list[dict[str, str | int]]:
    """
    Get all the chunks for py and md files
    max_chunk_size: size max of a chunk
    return a list of dict with file, text, first_char_index, last_char_index
    as keys
    """
    chunks = []
    files = files_name_loader()
    try:
        for file in tqdm(files['py'], desc="Indexing .py files"):
            chunks.extend(python_code_chunker(read_file(file),
                                              file, max_chunk_size))
        for file in tqdm(files['md'], desc="Indexing .md files"):
            chunks.extend(text_chunker(read_file(file), file, max_chunk_size))
        return chunks

    except Exception:
        return []
# print(get_all_chunk(2000))
