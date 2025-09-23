.PHONY: setup test lint run smoke

setup:
	pip install -e . && pre-commit install

test:
	pytest -q

lint:
	pre-commit run -a

smoke:
	specguard-chem run --suite basic --protocol L1 --model heuristic --limit 5
