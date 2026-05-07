# SpecGuard-Chem

Oracle-compiled evaluation contracts for agentic language models under chemically typed scientific specifications.

**What it is:** a model-agnostic compiler and benchmark harness for rule-following under explicit, machine-checkable specs. Agents propose/edit molecules, optionally use verifier tools, and either accept, reject, or abstain.

**What it is NOT:** drug discovery, activity/toxicity prediction, synthesis planning, therapeutic selection, clinical evaluation, dosing guidance, disease modeling, or target-binding prediction.

Prompts are optional rendering. Canonical benchmark semantics are the structured task/spec objects, public task views, action contracts, and deterministic verifier truth.

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

## Primary Benchmark Release (sgchem_v1.0)
Compile the oracle-backed bundle release:

```bash
uv run specguard-chem compile-benchmark \
  --benchmark-id sgchem_v1.0 \
  --out benchmarks/releases/sgchem_v1.0 \
  --seed 7 \
  --target-bundles 120 \
  --min-tasks 650 \
  --max-tasks 900 \
  --anonymous
```

Strictly validate the frozen release:

```bash
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```

Run prompt-isolation and artifact hardening audits:

```bash
uv run python scripts/audit_model_prompt_leakage.py --release benchmarks/releases/sgchem_v1.0
uv run python scripts/audit_oracle_scrambling.py --release benchmarks/releases/sgchem_v1.0
uv run python scripts/preflight_neurips_ed_artifact.py --release benchmarks/releases/sgchem_v1.0
```

Run the primary paper sweep (track-separated: closed-book + retrieval):

```bash
uv run specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v1.0 \
  --split test \
  --baselines baselines/paper_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v1.0_test \
  --seed 7

uv run specguard-chem paper-figures \
  --runs runs/paper_sweeps/sgchem_v1.0_test \
  --out paper_v1

uv run python scripts/audit_metric_sanity.py \
  --release benchmarks/releases/sgchem_v1.0 \
  --runs runs/paper_sweeps/sgchem_v1.0_test \
  --paper paper_v1
```

Metric sanity reports rename the internal `accept_rate` to `molecule_acceptance_rate` and demote it from headline status. The paper package should emphasize action accuracy, task-inconsistent acceptance, reject/abstain recall, diagnostic denominators, and verifier/tool-economy differences.

Run the wrapper-saturation reality check:

```bash
uv run python scripts/run_reality_check_experiments.py \
  --release benchmarks/releases/sgchem_v1.0 \
  --out runs/reality_check/sgchem_v1.0 \
  --skip-wrapper
```

The committed memo in `paper_v1/reality_check_decision_memo.md` reports that `well_engineered_wrapper` saturates the 266-task test split under the public verifier/search-wrapper threat model. That is an intended evaluation-validity result: sgchem_v1.0 should be interpreted as an oracle-compiled specification-compliance contract, not an intrinsic chemistry-capability leaderboard.

Create the anonymous reviewer archive:

```bash
uv run python scripts/package_anonymous_artifact.py \
  --release benchmarks/releases/sgchem_v1.0 \
  --out sgchem_v1.0_anonymous_artifact.zip
```

After uploading the archive to anonymous hosting, rerun the package/preflight command with `--dataset-url <anonymous-url>` and run the clean reviewer reproduction:

```bash
uv run python scripts/test_clean_reviewer_reproduction.py
```

Run external/LLM snapshot baselines with cache capture (optional):

```bash
uv run specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v1.0 \
  --split test \
  --baselines baselines/external_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v1.0_external \
  --allow-external \
  --cache-dir runs/paper_sweeps/sgchem_v1.0_external/cache
```

Replay external baselines offline from cache:

```bash
uv run specguard-chem run-benchmark \
  --benchmark benchmarks/releases/sgchem_v1.0 \
  --split test \
  --baselines baselines/external_baselines.yaml \
  --out runs/paper_sweeps/sgchem_v1.0_external_replay \
  --replay-cache runs/paper_sweeps/sgchem_v1.0_external/cache
```

Generate paper figures/tables (track-separated leaderboards + CI columns):

```bash
uv run specguard-chem paper-figures \
  --runs runs/paper_sweeps/sgchem_v1.0_test \
  --out paper_v1
```

One-command rc2-local reproduction and artifact preflight:

```bash
uv run python scripts/build_and_check_sgchem_v1.py
```

Prepare the anonymous hosted artifact upload from `hosting/` and finalize the URL after upload:

```bash
uv run python scripts/finalize_hosted_url.py \
  --release benchmarks/releases/sgchem_v1.0 \
  --dataset-url "<ANONYMOUS_HOSTED_DATASET_URL>"
uv run python scripts/check_paper_consistency.py --mode final
```

Inspect one test bundle manually in `benchmarks/releases/sgchem_v1.0/audits/manual_test_bundle_dossiers.md`. Each dossier shows rendered public inputs, hidden oracle summaries, hashes, suggested reviewer objections, manual grade, decision, and paper-safe status.

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
- `primary_closed_book`: no retrieval, no external calls (primary leaderboard).
- `tool_enabled`: verifier-tool baselines reported separately from closed-book models.
- `retrieval_upper_bound`: retrieval-allowed baselines (`corpus_search`) reported separately as an upper bound.
- `oracle_upper_bound`: oracle-assisted controls, if present, never mixed into model leaderboards.
- `external_model_snapshot`: API/process snapshot baselines; optional and replayable from cache.

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

<!-- sgchem-hosted-url:start -->
## Hosted Artifact

The double-blind review artifact URL is maintained outside this named public
repository until anonymity is no longer required. The release metadata and
Croissant files can be finalized with `scripts/finalize_hosted_url.py` in an
anonymous artifact copy.
<!-- sgchem-hosted-url:end -->
