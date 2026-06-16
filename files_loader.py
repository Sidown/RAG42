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
    py_files = glob.glob(repo + '*.py', recursive=True)
    files_dict.update({'py': py_files})
    txt_file =  glob.glob(repo + '*.txt', recursive=True)
    files_dict.update({'txt': txt_file})
    md_file = glob.glob(repo + '*.md', recursive=True)
    files_dict.update({'md': md_file})


