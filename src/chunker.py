import textwrap
import glob
import ast
from tqdm import tqdm
from typing import TypedDict


class Chunk(TypedDict):
    file: str
    text: str
    first_char_index: int
    last_char_index: int


def text_chunker(text: str, file_path: str,
                 max_chunk_size: int) -> list[Chunk]:
    """
    Split a markdown text into chunks by '#' sections.

    Args:
        text: The full text content to chunk.
        file_path: Path of the source file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of Chunk dicts with file, text, first_char_index,
        last_char_index keys.
    """
    chunks: list[Chunk] = []
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
                        max_chunk_size: int) -> list[Chunk]:
    """
    Split a Python file into chunks by class and function definitions.

    Uses the ast module to identify FunctionDef and ClassDef nodes
    and extract their character positions in the original file.

    Args:
        text: The full Python source code.
        file_path: Path of the source file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of Chunk dicts with file, text, first_char_index,
        last_char_index keys.
    """
    chunks: list[Chunk] = []
    overlap = 200
    try:
        tree = ast.parse(text)
        lines = text.split('\n')
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.ClassDef))
               and node.lineno is not None
               and node.end_lineno is not None):
                first_char = sum(len(lines[i]) + 1
                                 for i in range(node.lineno - 1))
                last_char = (sum(len(lines[i]) + 1
                             for i in range(node.end_lineno - 1))
                             + len(lines[node.end_lineno - 1]))
                chunks.append({
                    'file': file_path,
                    'text': '\n'.join(
                        lines[node.lineno - 1:node.end_lineno]),
                    'first_char_index': first_char,
                    'last_char_index': last_char
                })

        saved: list[Chunk] = []
        for chunk in chunks:
            if len(chunk['text']) > max_chunk_size:
                texts = textwrap.wrap(chunk['text'], max_chunk_size)
                chunk = {
                    'file': chunk['file'],
                    'text': texts[0],
                    'first_char_index': chunk['first_char_index'],
                    'last_char_index': chunk['first_char_index'] + len(texts[0])
                }
                texts.pop(0)

                offset = 0
                for t in texts:
                    print(f"Taille de texts = {len(texts)}")
                    saved.append({
                        'file': chunk['file'],
                        'text': t,
                        'first_char_index': chunk['last_char_index'] + offset,
                        'last_char_index': chunk['last_char_index'] + offset + len(t),
                        })
                    offset = chunk['last_char_index'] + offset + len(t)
                    texts.pop(0)
        
        for save in saved:
            chunks.append(save)

        return chunks
    except Exception:
        return []


def files_name_loader() -> dict[str, list[str]]:
    """
    Load all .py and .md file paths from the vLLM repository.

    Returns:
        A dict with 'py' and 'md' keys, each mapping to a list
        of file paths as strings.
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
    Read a file and return its text content.

    Args:
        file: Path of the file to read.

    Returns:
        The file content as a string, or empty string on error.
    """
    try:
        with open(file, encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception:
        return ''


def get_all_chunk(max_chunk_size: int) -> list[Chunk]:
    """
    Chunk all .py and .md files from the vLLM repository.

    Args:
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of Chunk dicts with file, text, first_char_index,
        last_char_index keys.
    """
    chunks: list[Chunk] = []
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
