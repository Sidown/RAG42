import glob
repo = './test_file/'
py_files = glob.glob(repo + '*.py', recursive=True)
txt_file =  glob.glob(repo + '*.txt', recursive=True)
md_file = glob.glob(repo + '*.md', recursive=True)
for file in py_files:
    if file == repo:
        py_files.remove(file)
for file in py_files:
    print(file)

for file in py_files:
    try:
        with open(file) as f:
            text = f.read()
            print(text)
    except Exception:
        pass