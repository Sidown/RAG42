.PHONY: install run debug lint clean

install:
	uv sync

lint: install
	flake8 src
	mypy src --follow-imports=skip --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf data/output
	rm -rf data/processed
	rm -rf .venv