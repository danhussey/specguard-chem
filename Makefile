.PHONY: setup test lint run smoke

setup:
	uv venv --seed
	uv pip install -e .[dev]
	uv run pre-commit install

test:
	uv run pytest -q

lint:
	uv run pre-commit run -a

smoke:
	uv run specguard-chem run --suite basic --protocol L1 --model heuristic --limit 5
