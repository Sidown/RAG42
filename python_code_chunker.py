def python_code_chunker(text: str) -> list[str]:
    chunks = []
    chunks = text.split('def')
    for chunk in chunks:
        if chunk == '':
            chunks.remove(chunk)
    return chunks


with open('./test_file/test.py') as f:
    text = f.read()
result = python_code_chunker(text)
i = 0
for r in result:
    print(f"Index {i}:")
    print(r)
    print('\n')
    i += 1