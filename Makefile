.PHONY: install run debug lint clean clean_cache fclean \
        test search answer \
        search_dataset_docs search_dataset_code \
        evaluate_docs evaluate_code \
        answer_dataset_docs

DATASET_DOCS_UNANSWERED = data/datasets/UnansweredQuestions/dataset_docs_public.json
DATASET_CODE_UNANSWERED = data/datasets/UnansweredQuestions/dataset_code_public.json
DATASET_DOCS_ANSWERED   = data/datasets/AnsweredQuestions/dataset_docs_public.json
DATASET_CODE_ANSWERED   = data/datasets/AnsweredQuestions/dataset_code_public.json

SEARCH_RESULTS_DIR = data/output/search_results/UnansweredQuestions
ANSWER_RESULTS_DIR = data/output/search_results_and_answer/UnansweredQuestions

TEST_QUERY = "How to configure the OpenAI server?"
TEST_K = 5

install:
	uv sync

run: install
	uv run python -m src --help

debug: install
	uv run python -m src --help

lint: install
	flake8 src
	mypy src --follow-imports=skip --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf data/output/search_results
	rm -rf data/output/search_results_and_answer
	rm -rf data/processed

clean_cache:
	rm -rf data/output/cache.json

fclean: clean_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf data/output
	rm -rf data/processed
	rm -rf .venv

index: install
	uv run python -m src index

search: install
	uv run python -m src search $(TEST_QUERY) --k $(TEST_K)

answer: install
	uv run python -m src answer $(TEST_QUERY) --k $(TEST_K)

search_dataset_docs: install
	uv run python -m src search_dataset \
		--dataset_path $(DATASET_DOCS_UNANSWERED) \
		--k $(TEST_K) \
		--save_directory $(SEARCH_RESULTS_DIR)

search_dataset_code: install
	uv run python -m src search_dataset \
		--dataset_path $(DATASET_CODE_UNANSWERED) \
		--k $(TEST_K) \
		--save_directory $(SEARCH_RESULTS_DIR)

evaluate_docs: test-search-dataset-docs
	uv run python -m src evaluate \
		--student_search_results_path $(SEARCH_RESULTS_DIR)/dataset_docs_public.json \
		--dataset_path $(DATASET_DOCS_ANSWERED)

evaluate_code: test-search-dataset-code
	uv run python -m src evaluate \
		--student_search_results_path $(SEARCH_RESULTS_DIR)/dataset_code_public.json \
		--dataset_path $(DATASET_CODE_ANSWERED)

answer_dataset_docs: test-search-dataset-docs
	uv run python -m src answer_dataset \
		--student_search_results_path $(SEARCH_RESULTS_DIR)/dataset_docs_public.json \
		--save_directory $(ANSWER_RESULTS_DIR)

test: evaluate_docs evaluate_code
	@echo "Quick tests done."