# sgchem_v1.0

Compiled oracle-backed SpecGuard-Chem benchmark release.

Validate:

```bash
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```

Run baselines after validation:

```bash
uv run specguard-chem run-benchmark --benchmark benchmarks/releases/sgchem_v1.0 --split test --baselines baselines/paper_baselines.yaml --out runs/paper_sweeps/sgchem_v1.0_test --seed 7
```
