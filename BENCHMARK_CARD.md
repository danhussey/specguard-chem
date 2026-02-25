# BENCHMARK_CARD — SpecGuard-Chem sgchem_v0.3

## Problem Definition
SpecGuard-Chem measures solver-agnostic structured spec compliance under strict interaction budgets. Canonical benchmark inputs are structured `task` + `spec` JSON objects. Prompts are optional rendering.

## Tracks (Reported Separately)
- `closed_book` (primary): no retrieval, no external/API dependency.
- `retrieval`: retrieval-enabled baselines (`corpus_search`) reported as an upper-bound track.
- `external` (optional snapshot): API/process baselines with cache+replay support.

## What Is Measured
- Pass@budget (`pass_at_steps`, `pass_at_1`, `pass_at_3`)
- Verifier/tool economy (`avg_verify_calls_used`, `l3_avg_verify_calls_used`, `verify_usage_rate_on_L3`)
- Edit economy (SMILES edit distance + BRICS final/trajectory costs)
- Abstention utility + sensitivity
- Calibration (`brier_score`, `ece`, reliability/risk-coverage behavior)
- Interrupt/resume (`resume_success_rate`, `avg_extra_steps_after_interrupt`)
- Gaming resistance:
  - boundary precision failures
  - adversarial invariance failures by subfamily (`stereo`, `tautomer`, `charge`, `aromatic`)
- Bootstrap 95% CIs in aggregate and paper tables.

## Invariance Policy (Explicit)
`equivalent_to_input` hard check supports policy presets:
- `strict_inchi`
- `no_stereo_inchi`
- `tautomer_canonical_inchi`
- `tautomer_canonical_no_stereo_inchi`

Additional switches (`charge_invariant`, `normalize`, `key`) are explicit in task constraints.

## What Is Not Claimed
- No drug-discovery performance claim
- No activity/efficacy/toxicity claim
- No synthesis-feasibility claim

## Label Credibility Policy
- `expected_action=ACCEPT`: requires witness molecule that hard-passes the effective spec.
- `expected_action=ABSTAIN`: requires machine-checkable contradiction proof (bounds contradiction minimum).
- Release creation is hard-gated by dataset invariants.

## Repro Commands
1. Freeze benchmark
```bash
specguard-chem freeze-benchmark \
  --benchmark-id sgchem_v0.3 \
  --out benchmarks/releases/sgchem_v0.3 \
  --target-tasks 1000 \
  --seed 7
```

2. Closed-book + retrieval sweep
```bash
specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v0.3 \
  --split test \
  --baselines baselines/paper_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v0.3_test
```

3. External sweep with cache (optional)
```bash
specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v0.3 \
  --split test \
  --baselines baselines/external_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v0.3_external \
  --allow-external \
  --cache-dir runs/paper_sweeps/sgchem_v0.3_external/cache
```

4. Offline replay of external sweep
```bash
specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v0.3 \
  --split test \
  --baselines baselines/external_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v0.3_external_replay \
  --replay-cache runs/paper_sweeps/sgchem_v0.3_external/cache
```

5. Paper outputs
```bash
specguard-chem paper-figures \
  --runs runs/paper_sweeps/sgchem_v0.3_test \
  --out paper
```
