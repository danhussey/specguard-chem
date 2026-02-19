.PHONY: setup test lint run smoke baselines compare-baselines

setup:
	uv venv --seed
	uv pip install -e .[dev]
	uv run pre-commit install

test:
	uv run pytest -q

lint:
	uv run pre-commit run -a

smoke:
	uv run specguard-chem run --suite basic_plain --protocol L1 --model heuristic --limit 5

baselines:
	uv run specguard-chem run-baselines --suite basic_plain --spec-split train --limit 3

compare-baselines:
	uv run specguard-chem compare-baselines runs/baselines -o runs/baseline_compare.json
