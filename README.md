# SpecGuard-Chem
[![CI](https://img.shields.io/github/actions/workflow/status/danhussey/specguard-chem/ci.yml?label=CI)](https://github.com/danhussey/specguard-chem/actions/workflows/ci.yml) ![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen)

Spec-driven, programmatically verifiable evaluation of agentic LLMs on safe medicinal-chemistry constraints.

**What it is:** a model-agnostic benchmark harness for rule-following under explicit specs. Agents propose/edit molecules, optionally use verifier tools, and either accept, reject, or abstain.

**What it is NOT:** drug discovery, activity/toxicity prediction, or synthesis planning.

Alert checks support expanded deterministic families (`PAINS_A/B/C`, `BRENK`).

## Quickstart
```bash
uv venv --seed
source .venv/bin/activate
uv pip install -e .[dev]

specguard-chem run basic_plain --protocol L1 --model heuristic --run-path runs/demo_basic_l1
specguard-chem run basic_plain --protocol L3 --model open_source_example --run-path runs/demo_basic_l3
specguard-chem report runs/demo_basic_l3

uv run pytest --cov=src/specguard_chem --cov-report=term-missing
```

`specguard-chem run` also supports `--spec-split train|dev|test` for held-out spec evaluation.

`specguard-chem report` reads `trace.jsonl` from a run directory and writes `report.json` with:
- decision-level confusion and utility
- budget-first efficiency (`pass_at_steps`, step/tool economy)
- calibration and risk/cost curves from `p_hard_pass`
- hard/soft separation and gaming-resistance metrics
- schema/error rates and dataset-version hashes/IDs

## Dataset Tooling
Deterministic benchmark generation/validation is built in:

```bash
specguard-chem build-corpus --output data/corpus.parquet --seed 7
specguard-chem generate-tasks --corpus data/corpus.parquet --output tasks/suites/generated_v1.jsonl --target-tasks 1000 --seed 7
specguard-chem validate-dataset tasks/suites/generated_v1.jsonl
```

Boundary semantics are inclusive with explicit floating tolerance (`BOUNDS_TOLERANCE = 1e-6`).

## Baselines
Run the baseline matrix:

```bash
specguard-chem run-baselines --suite basic_plain --spec-split train --limit 5
```

This emits one run per baseline (`heuristic_non_tool_l2`, `heuristic_tool_l3`, `abstention_guard_l2`) and writes `baseline_summary.json`.

Compare one or more baseline batches:

```bash
specguard-chem compare-baselines runs/baselines -o runs/baseline_compare.json
```

Stratify aggregate rows with `--group-by` (fields: `name,model,protocol,suite,spec_split,source`):

```bash
specguard-chem compare-baselines runs/baselines --group-by name,spec_split -o runs/baseline_compare_by_split.json
```

## Included Adapters
- `heuristic`: deterministic mutator using failure-vector feedback in L2/L3.
- `open_source_example`: tool-using baseline for L3.
- `abstention_guard`: conservative abstention-heavy baseline.
- `process`: delegates each step to an external command (`SPEC_GUARD_PROCESS_ADAPTER_CMD`).
- `openai_chat`: OpenAI Chat Completions-backed adapter (`OPENAI_API_KEY`).

See `docs/adapters.md` for integration details.

## Included Task Suites
- `basic_plain` (10): mixed L1/L2/L3 tasks.
- `basic_checklist` (10): checklist prompt variant of `basic_plain`.
- `repair_ladder_plain` (3): repair-focused tasks.
- `repair_ladder_checklist` (3): checklist variant of repair ladder.
- `interrupts` (3): interrupt handling with abstention-oriented behavior.
- `interrupt_strict` (3): stricter interrupt compliance requirements.
- `interrupt_resume` (3): interrupt acknowledge + resume-token echo + continue.
- `alerts_pains_soft` (4): alert-focused soft-constraint tasks.
- `smiles_invariance` (4): equivalent-SMILES invariance checks.
- `boundary_precision` (3): near-boundary precision/tolerance checks.

## Continuous Integration
CI runs lint/tests, coverage, smoke runs (`run` + `report`), and baseline smoke (`run-baselines`).

For architecture details see `docs/overview.md`. For formulas see `METRICS.md`. For scope guardrails see `SAFETY.md`.
