import textwrap


def text_chunker(text: str) -> list[str]:
    chunks = []
    chunks = textwrap.wrap(text, 2000)
    return chunks


with open('./test_file/test.md') as f:
    text = f.read()

print(text_chunker(text))

