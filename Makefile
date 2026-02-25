.PHONY: setup test lint run smoke baselines compare-baselines freeze-benchmark run-benchmark paper-figures

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

freeze-benchmark:
	uv run specguard-chem freeze-benchmark --benchmark-id sgchem_v0.1 --out benchmarks/releases/sgchem_v0.1 --target-tasks 200 --seed 7

run-benchmark:
	uv run specguard-chem run-benchmark --benchmark benchmarks/releases/sgchem_v0.1 --split test --baselines baselines/paper_baselines.yaml --out runs/paper_sweeps/sgchem_v0.1_test --seed 7

paper-figures:
	uv run python scripts/make_paper_figures.py --runs runs/paper_sweeps/sgchem_v0.1_test --out paper
