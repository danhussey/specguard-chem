# SpecGuard-Chem

SpecGuard-Chem is a reproducible evals project for agentic language models that
must operate under explicit, machine-checkable chemistry specifications.

The task is not "generate a plausible molecule." The task is to read a public
specification, decide whether to `ACCEPT`, `REJECT`, or `ABSTAIN`, optionally use
verifier tools, and leave a trace that can be replayed and audited.

**What it is:** a benchmark compiler, deterministic RDKit verifier harness,
runner, baseline suite, paper artifact chain, and cached external-model result
package for chemistry-flavored specification following.

**What it is not:** drug discovery, activity prediction, toxicity prediction,
synthesis planning, therapeutic selection, clinical evaluation, dosing guidance,
disease modeling, or target-binding prediction.

Prompts are only a rendering layer. The canonical semantics are the structured
task/spec objects, public task views, action contracts, deterministic verifier
truth, and replayable traces.

## Why This Exists

Many molecule-generation demos blur together several questions:

- Did the model output syntactically valid SMILES?
- Did the molecule satisfy a visible specification?
- Was accepting a molecule the right action for this task?
- Did the agent use verifier/tool feedback correctly?
- Can the result be reproduced without another live API call?

SpecGuard-Chem separates those questions. The benchmark includes feasible
construction, candidate audit, contradiction/abstention, near-miss repair,
boundary precision, SMILES invariance, tool-forced L3, and interrupt/resume
cases. This makes it useful as a small, controlled testbed for specification
following and tool-mediated agent control.

## What Is Implemented

- A deterministic `sgchem_v1.0` benchmark compiler with train/dev/test splits.
- RDKit-backed verifier checks for property bounds, alerts, synthetic
  accessibility proxies, edit constraints, and invariance policies.
- A runner that emits JSON traces, TSV leaderboards, cacheable external calls,
  replay runs, confusion matrices, calibration fields, and verifier-use metrics.
- Baselines covering closed-book heuristics, abstention, local mutation,
  retrieval, verifier-first policies, a deterministic wrapper, and external
  OpenAI/Anthropic/DeepSeek adapters.
- Artifact checks for prompt leakage, oracle scrambling, dataset validation,
  paper/table consistency, and external interface preflight.
- Frozen offline and strict external result packages committed for review.

## Relationship to SpecGuard-Agent

This repository is the chemistry-specific artifact. It should stay focused on
molecular specification contracts, deterministic verifiers, and frozen
SpecGuard-Chem results.

The broader SpecGuard-Agent direction grew out of this work. The strongest lead
from the strict external runs is not a chemistry claim; it is an interface-design
claim. The current L3 verifier contract lacks candidate history, remaining
budget state, and a clean split between final decisions and tool requests. That
belongs in the general agent-control line of work. SpecGuard-Chem remains the
domain-specific benchmark and evidence base.

## Current Frozen Results

Primary offline package:

```text
paper_final/results_offline_full_2026_05_20/
```

Strict external snapshot:

```text
external_baselines/results_full_2026_05_20_strict_v3/
```

Key readout from the held-out 266-task test split:

| System | Access model | Action accuracy | Molecule acceptance | Reject recall | Abstain recall |
| --- | --- | ---: | ---: | ---: | ---: |
| `corpus_search` | retrieval | 0.673 | 0.868 | 0.000 | 0.000 |
| `local_mutation` | closed-book | 0.650 | 0.846 | 0.000 | 0.000 |
| `heuristic` | closed-book | 0.602 | 0.406 | 1.000 | 0.000 |
| `well_engineered_wrapper` | verifier/search wrapper | 1.000 | 0.673 | 1.000 | 1.000 |

The wrapper's lower molecule-acceptance rate is not a weakness: only 179 of the
266 held-out tasks require `ACCEPT`. The result shows why the action contract
needs to be evaluated directly.

The strict external v3 run completed with 3,968 cached live steps and zero
interface-error steps. That removes the earlier malformed-output confound, but
the L3 verify rows should still be treated as diagnostics rather than a final
model leaderboard because the tool contract itself needs a stateful redesign.

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

## Artifact Map

Start with these files when reviewing the project:

- `paper_final/main.tex`: current manuscript draft.
- `paper_final/README.md`: build notes for the review package.
- `paper_final/reports/`: interpretation memos and artifact checks.
- `paper_final/tables/` and `paper_final/figures/`: selected paper-facing
  assets.
- `scripts/run_paper_v2_results.sh`: offline result orchestration.
- `scripts/run_external_baselines.sh`: strict external baseline orchestration.
- `external_baselines/RUNBOOK.md`: diagnostic/full online runbook.

The most complete offline result package is:

```text
paper_final/results_offline_full_2026_05_20/
```

Useful entry points:

- `RESULTS_SUMMARY.md`: short interpretation and caveats.
- `tables/main_table_representative_baselines_with_ci.md`: representative offline rows with task-level bootstrap CIs.
- `tables/full_offline_baseline_matrix_test.csv`: full held-out test matrix.
- `tables/wrapper_ablation_test.csv`: verifier/search wrapper ablations.
- `tables/protocol_ladder_test.csv`: native L1/L2/L3 protocol slices.
- `notes/*_interpretation.md`: paper-facing interpretation notes.

The complete strict external baseline snapshot is:

```text
external_baselines/results_full_2026_05_20_strict_v3/
```

This run uses strict structured tool outputs for OpenAI, Anthropic, and DeepSeek
adapters. The committed review package keeps the summary tables and metadata:

- `tables/replay/external_baseline_summary.json`
- `tables/replay/external_baseline_metrics.csv`

Treat these rows as external diagnostic snapshots, not as the primary offline
leaderboard. The v3 contract fixed the malformed-output problem, but the traces
also show that the current L3 verifier interface is not a clean agent-control
contract: it lacks candidate history, remaining-budget state, and a clear split
between final decisions and tool requests. That finding is useful for the
broader SpecGuard-Agent direction, but the SpecGuard-Chem paper should keep the
claim grounded in the chemistry benchmark and frozen artifacts.

Raw traces and live-call caches are intentionally kept out of the Git review
diff. They can be regenerated or attached as an external archive if needed.

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
- `anthropic_chat`: Anthropic Messages adapter (`ANTHROPIC_API_KEY`).
- `anthropic_chat_verify_l3`: Anthropic adapter with an L3 verify-first policy template.
- `deepseek_chat`: DeepSeek OpenAI-compatible adapter (`DEEPSEEK_API_KEY`).
- `deepseek_chat_verify_l3`: DeepSeek adapter with an L3 verify-first policy template.

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
