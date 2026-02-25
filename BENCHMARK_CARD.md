# BENCHMARK_CARD — SpecGuard-Chem sgchem_v0.2

## Problem Definition
SpecGuard-Chem measures solver-agnostic compliance with structured medicinal-chemistry-style constraints under explicit interaction budgets. A task is defined by structured JSON objects (`task`, `spec`); prompts are optional metadata.

## What Is Measured
- Pass@budget (`pass_at_steps`, `avg_steps_to_accept`, `avg_verify_calls_to_accept`)
- Verifier-call economy and budget adherence
- Edit economy (`edit_distance`, BRICS final/trajectory edit costs)
- Abstention utility and utility sensitivity
- Calibration (`brier_score`, `ece`, reliability behavior via `p_hard_pass`)
- Interrupt/resume robustness (`resume_success_rate`, `avg_extra_steps_after_interrupt`)
- Gaming resistance (`invariance_failure_rate`, `boundary_precision_failure_rate`)
- Tool-gating behavior in L3 (`avg_verify_calls_used`, suite-level outcomes)

`corpus_search` is reported as a retrieval-track baseline (upper bound), not the primary closed-book score.

Formulas and metric definitions are in [`METRICS.md`](/Users/danielhussey/Code/specguard-chem/METRICS.md).

## What Is Not Claimed
- No drug discovery performance claims
- No activity, toxicity, or efficacy prediction
- No synthesis planning feasibility claims

## Label Credibility Policy
- `expected_action=ACCEPT`: requires `feasible_witness_smiles` that hard-passes the effective spec.
- `expected_action=ABSTAIN`: requires contradiction proof; bounds contradictions are machine-checkable.
- Dataset invariants are hard-gated during release creation.

## Reproducibility Commands
1. Freeze release
```bash
specguard-chem freeze-benchmark \
  --benchmark-id sgchem_v0.2 \
  --out benchmarks/releases/sgchem_v0.2 \
  --target-tasks 1000 \
  --seed 7
```
2. Run baseline sweep
```bash
specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v0.2 \
  --split test \
  --baselines baselines/paper_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v0.2_test
```
3. Generate paper artifacts
```bash
uv run python scripts/make_paper_figures.py \
  --runs runs/paper_sweeps/sgchem_v0.2_test \
  --out paper
```
