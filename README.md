*This project has been created as part of the 42 curriculum by clefrere.*
 
---
 
# RAG against the machine
 
A Retrieval-Augmented Generation (RAG) system that answers questions about the vLLM
codebase by retrieving relevant source code and documentation, then generating grounded
answers using a local language model.
 
---
 
## Description
 
This project implements a complete RAG pipeline targeting the vLLM repository. Given
a question, the system retrieves the most relevant code snippets and documentation from
the indexed codebase, passes them to Qwen/Qwen3-0.6B as context, and generates a
natural language answer grounded in the retrieved sources.
 
The system is evaluated using recall@k metrics: it must achieve at least 80% recall@5
on documentation questions and 50% recall@5 on code questions.
 
---
 
## System Architecture
 
The pipeline is divided into two phases:
 
**Indexing phase** (run once):
 
```
vLLM repository (data/raw/vllm-0.10.1/)
        ↓
Chunking (.py → AST-based, .md → section-based)
        ↓
BM25 index + optional Semantic embedding index
        ↓
data/processed/
```
 
**Retrieval and generation phase** (run per query):
 
```
Question
    ↓
BM25 search (+ optional semantic search with RRF fusion)
    ↓
Top-k source locations (file_path, first_char_index, last_char_index)
    ↓
Context extraction from source files
    ↓
Qwen/Qwen3-0.6B answer generation
    ↓
Structured JSON output
```
 
The system is split into the following modules:
 
| Module | Role |
|---|---|
| `chunker.py` | File reading and chunking strategies |
| `index_manager.py` | BM25 indexing, persistence, identifier expansion and search |
| `semantic_embeddings.py` | Sentence-transformer vector index |
| `qwen.py` | Qwen3-0.6B inference wrapper |
| `data_models.py` | Pydantic models for all pipeline data |
| `cli.py` | Python Fire CLI exposing all commands |
 
---
 
## Chunking Strategy
 
Two distinct strategies are implemented depending on file type:
 
**Python files (`.py`) — AST-based chunking**
 
The `ast` module parses each file and extracts every `FunctionDef` and `ClassDef` node.
Each function or class becomes one chunk, preserving logical boundaries. Character
positions (`first_char_index`, `last_char_index`) are computed by accumulating line
lengths. Chunks exceeding `max_chunk_size` are further split line by line to respect
the 2000-character limit imposed by the moulinette.
 
**Markdown files (`.md`) — Section-based chunking**
 
Each file is split on `#` characters, which correspond to section headers. Sections
longer than `max_chunk_size` are further split using `textwrap.wrap`, which respects
word boundaries. Character positions are tracked using `str.find()` with a forward
cursor to handle repeated content correctly.
 
The maximum chunk size is configurable via `--max_chunk_size` (default: 2000). The
moulinette rejects any source longer than 2000 characters, so this limit is enforced on
the character range declared in the JSON output.
 
---
 
## Retrieval Method
 
**BM25 (default)**
 
BM25 (Best Match 25) is a lexical ranking function built on TF-IDF with two key
improvements:
 
- **Term frequency saturation**: repeated occurrences of a term contribute diminishing
  returns beyond a threshold, controlled by parameter `k1`.
- **Length normalization**: longer chunks are penalized relative to the corpus average,
  controlled by parameter `b`.
The `bm25s` library handles tokenization and indexing. The index is persisted under
`data/processed/bm25_index/` and loaded at retrieval time without reindexing.
 
**Identifier expansion**
 
Before indexing and querying, snake_case and camelCase identifiers are split into their
component words. For example, `block_manager` becomes `block manager` and
`PagedAttention` becomes `Paged Attention`. This allows BM25 to match questions
that use natural language terms against code that uses compound identifiers.
 
**Hybrid retrieval with RRF (optional bonus)**
 
When `--hybrid True` is passed, a second search is performed using semantic embeddings
(`all-MiniLM-L6-v2` via `sentence-transformers`). The two ranked lists (BM25 and
semantic) are fused using **Reciprocal Rank Fusion**:
 
```
RRF score = 0.6 * 1/(60 + rank_BM25) + 0.4 * 1/(60 + rank_semantic)
```
 
The constant 60 prevents the top-ranked result of either system from dominating the
fusion. BM25 is weighted at 0.6 as it generally performs better on this codebase.
 
---
 
## Performance Analysis
 
Scores measured on the public datasets with `k=10` and `max_chunk_size=2000`:
 
| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| Docs    | 57%      | 73%      | 81%      | 84%       |
| Code    | 38%      | 55%      | 63%      | 73%       |
 
The docs dataset meets the 80% Recall@5 threshold and the code dataset meets the 50% Recall@5 threshold.
 
BM25 performs well on documentation because questions and answers share vocabulary.
On code, performance is lower because questions often paraphrase concepts using different
terms than the identifiers in source code. Identifier expansion was added to partially
address this gap.
 
---
 
## Design Decisions
 
**AST over regex for Python chunking**
 
Using `ast.parse()` guarantees that function and class boundaries are respected,
regardless of indentation style or nested definitions. A naive `split('def ')` approach
would break nested functions and methods inside classes, producing semantically
incomplete chunks.
 
**BM25 over TF-IDF**
 
BM25's length normalization is important for a codebase where chunk sizes vary
significantly. A one-line utility function and a 200-line class should not be scored
the same way just because the larger one contains more keyword occurrences.
 
**Lazy LLM loading**
 
`QwenChatbot` is instantiated only when `answer` or `answer_dataset` is called, not
at import time. This keeps `index` and `search` commands fast and avoids loading a
multi-gigabyte model unnecessarily.
 
**Query result caching**
 
Answers are cached to `data/output/cache.json` after each generation. On repeated
runs of `answer_dataset`, previously processed questions are returned instantly from
cache, making interrupted runs resumable.
 
**Separate BM25 and semantic indexes**
 
The BM25 index is stored as a directory (`bm25s` native format) alongside `chunks.json`.
The semantic index is stored as a NumPy `.npy` file. This separation allows BM25-only
runs without loading the sentence-transformer model.
 
---
 
## Challenges Faced
 
**UTF-8 encoding**
 
**Chunk position tracking in Markdown**
 
`textwrap.wrap` slightly reformats text (normalizes whitespace), making `str.find()`
fail on the wrapped substrings. The fix was to compute `first_char_of_line` once before
wrapping using `str.find()`, then advance an `offset` counter by `len(s)` for each
sub-chunk rather than re-searching.
 
**Python chunk size enforcement**
 
Large classes can easily exceed 2000 characters. The moulinette rejects any source
exceeding 2000 characters, so large Python chunks are split line by line to respect the
limit while preserving as much logical context as possible.
 
 
---
 
## Instructions
 
### Requirements
 
- Python 3.10+
- `uv` package manager
### Installation
 
```bash
git clone <your-repo-url>
cd RAG42
uv sync
```
 
### Setup
 
Place the vLLM repository under `data/raw/`:
 
```
data/raw/vllm-0.10.1/
```
 
Place the datasets under `data/datasets/`:
 
```
data/datasets/UnansweredQuestions/
data/datasets/AnsweredQuestions/
```
 
### Makefile rules
 
```bash
make install    # install dependencies via uv sync
make run        # display CLI help
make lint       # run flake8 and mypy
make clean      # remove caches and generated index files
make debug      # run in pdb debug mode
```
 
---
 
## Example Usage
 
**Index the repository (BM25 only):**
 
```bash
uv run python -m src index --max_chunk_size 2000
```
 
**Index with semantic embeddings (hybrid mode):**
 
```bash
uv run python -m src index --max_chunk_size 2000 --hybrid True
```
 
**Search a single query:**
 
```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```
 
**Search a full dataset:**
 
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```
 
**Answer a single query:**
 
```bash
uv run python -m src answer "What is the PagedAttention mechanism?" --k 10
```
 
**Generate answers for a dataset:**
 
```bash
uv run python -m src answer_dataset \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```
 
**Evaluate retrieval quality:**
 
```bash
uv run python -m src evaluate \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
 
**Hybrid retrieval search:**
 
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions \
  --hybrid True
```
 
---
 
## Bonus Features
 
**Semantic embeddings** — A vector index is built using `all-MiniLM-L6-v2` from
`sentence-transformers`. Enabled with `--hybrid True` during indexing.
 
**Hybrid retrieval** — BM25 and semantic rankings are fused using Reciprocal Rank
Fusion (RRF) with configurable weights. Enabled with `--hybrid True` during search.
 
**Query caching** — Answers are cached to disk after generation and reused on repeated
queries, making `answer_dataset` resumable after interruption.
 
---
 
## Resources

- [Hybrid Search](https://medium.com/@mahima_agarwal/hybrid-search-bm25-vector-embeddings-the-best-of-both-worlds-in-information-retrieval-0d1075fc2828)
- [RRF](https://medium.com/@mahima_agarwal/hybrid-search-bm25-vector-embeddings-the-best-of-both-worlds-in-information-retrieval-0d1075fc2828)
- [Qwen3 LLM Based Embeddings Explained](https://medium.com/@mandeep0405/qwen3-llm-based-embeddings-explained-deaf6bf3aace)
- [Sentence Transformers](https://huggingface.co/sentence-transformers)
- [Creating A Semantic Search Model With Sentence Transformers For A RAG Application](https://nlpcloud.com/fine-tuning-semantic-search-model-with-sentence-transformers-for-rag-application.html)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [What is BM25 (Best Matching 25) Algorithm](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/)

## IA Usage

- Lint resolution
- Readme
- Docstrings
- Makefile