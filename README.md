# SpecGuard-Chem
[![CI](https://img.shields.io/github/actions/workflow/status/danhussey/specguard-chem/ci.yml?label=CI)](https://github.com/danhussey/specguard-chem/actions/workflows/ci.yml) ![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen)

Spec-driven, programmatically verifiable evaluation of agentic LLMs on safe medicinal-chemistry constraints.

**What it is:** a model-agnostic benchmark harness for rule-following under explicit specs. Agents propose/edit molecules, optionally use verifier tools, and either accept, reject, or abstain.

**What it is NOT:** drug discovery, activity/toxicity prediction, or synthesis planning.

Prompts are optional rendering. Canonical benchmark semantics are the structured task/spec objects and deterministic verifier truth.

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

## Frozen Benchmark Release (sgchem_v0.3)
Create a deterministic frozen release artifact:

```bash
specguard-chem freeze-benchmark \
  --benchmark-id sgchem_v0.3 \
  --out benchmarks/releases/sgchem_v0.3 \
  --target-tasks 1000 \
  --seed 7
```

Run the primary paper sweep (track-separated: closed-book + retrieval):

```bash
specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v0.3 \
  --split test \
  --baselines baselines/paper_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v0.3_test
```

Run external/LLM snapshot baselines with cache capture (optional):

```bash
specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v0.3 \
  --split test \
  --baselines baselines/external_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v0.3_external \
  --allow-external \
  --cache-dir runs/paper_sweeps/sgchem_v0.3_external/cache
```

Replay external baselines offline from cache:

```bash
specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v0.3 \
  --split test \
  --baselines baselines/external_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v0.3_external_replay \
  --replay-cache runs/paper_sweeps/sgchem_v0.3_external/cache
```

Generate paper figures/tables (track-separated leaderboards + CI columns):

```bash
specguard-chem paper-figures \
  --runs runs/paper_sweeps/sgchem_v0.3_test \
  --out paper
```

## Included Adapters
- `heuristic`: deterministic mutator using failure-vector feedback in L2/L3.
- `open_source_example`: tool-using baseline for L3.
- `abstention_guard`: conservative abstention-heavy baseline.
- `verify_first`: L3 baseline that explicitly calls `verify()` before proposing.
- `corpus_search`: deterministic corpus retrieval baseline (retrieval-track upper bound).
- `local_mutation`: deterministic local mutation hill-climb baseline (non-LLM).
- `process`: external command adapter (`SPEC_GUARD_PROCESS_ADAPTER_CMD`), cache/replay compatible.
- `openai_chat`: OpenAI Chat Completions adapter (`OPENAI_API_KEY`).
- `openai_chat_verify_l3`: OpenAI adapter with an L3 verify-first policy template.

See `docs/adapters.md` for integration details.

### Tracks
- `closed_book`: no retrieval, no external calls (primary leaderboard).
- `retrieval`: retrieval-allowed baselines (`corpus_search`) reported separately as upper bound.
- `external`: API/process snapshot baselines; optional and replayable from cache.

## Included Task Suites
- `basic_plain` (10): mixed L1/L2/L3 tasks.
- `basic_checklist` (10): checklist prompt variant of `basic_plain`.
- `repair_ladder_plain` (3): repair-focused tasks.
- `repair_ladder_checklist` (3): checklist variant of repair ladder.
- `interrupts` (3): interrupt handling with abstention-oriented behavior.
- `interrupt_strict` (3): stricter interrupt compliance requirements.
- `interrupt_resume` (3): interrupt acknowledge + resume-token echo + continue.
- `alerts_pains_soft` (4): alert-focused soft-constraint tasks.
- `smiles_invariance` (4+): adversarial invariance families (stereo/tautomer/charge/aromatic) with explicit equivalence policies.
- `boundary_precision` (3): near-boundary precision/tolerance checks.

## Continuous Integration
CI runs lint/tests, coverage, smoke runs (`run` + `report`), and baseline smoke (`run-baselines`).

For architecture details see `docs/overview.md`. For formulas see `METRICS.md`. For scope guardrails see `SAFETY.md`.
Benchmark positioning and release policy are documented in `BENCHMARK_CARD.md`.
