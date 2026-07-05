.PHONY: install run debug lint clean clean_cache fclean

install:
	uv sync

run: install
	uv run python -m student --help

debug: install
	uv run python -m student --help

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
