# SpecGuard-Chem — Conceptual Overview

This document describes the current architecture and data flow of SpecGuard-Chem.

## 1) Purpose
SpecGuard-Chem is a reproducible harness for evaluating agent behavior against explicit medicinal-chemistry-style specs. It measures rule-following, abstention, interrupt safety, and budget/tool efficiency.

It is not a drug-discovery pipeline.

## 2) System Layout
1. **Specs** (`data/specs/*.yaml`): strict v2 spec definitions with families/splits and typed constraints.
2. **Task suites** (`tasks/suites/*.jsonl`): protocol-tagged episodes with budgets, expected actions, optional interrupt policy, and evidence.
3. **Runner** (`specguard_chem.runner.runner.TaskRunner`): orchestrates L1/L2/L3 loops, tool calls, gating, interrupts/resume tokens, and trace logging.
4. **Verifiers** (`specguard_chem.verifiers.*`): deterministic RDKit-backed checks for properties, alert families, SA proxy, canonicalization, and edit costs.
5. **Adapters** (`specguard_chem.models.*`): pluggable agent implementations (`heuristic`, `open_source_example`, `abstention_guard`, `process`, `openai_chat`).
6. **Scoring/reports** (`specguard_chem.scoring.*`): computes decision-level metrics, curves, calibration, and metadata-rich `report.json`.
7. **Dataset pipeline** (`specguard_chem.dataset.*`): deterministic corpus/task generation and dataset validation.

## 3) Protocol Semantics
- **L1**: one-shot proposal, no feedback.
- **L2**: iterative repair with coarse feedback (`hard_fail_ids`, `soft_miss_ids`, parse-error signal).
- **L3**: same proposal feedback as L2, plus explicit `verify(smiles)` tool returning full constraint-level vectors.

This makes verifier calls measurable instead of giving full margins for free.

## 4) Runner Behavior
For each task, runner:
- resolves and embeds full `spec` in every adapter request
- enforces task budgets (`max_steps`, `max_proposals`, `max_verify_calls`, `max_total_verifier_calls`)
- normalizes malformed adapter outputs to abstain while recording schema flags
- hard-gates final acceptance on verifier hard pass
- tracks decision as `final_decision ∈ {ACCEPT, REJECT, ABSTAIN}`
- logs interrupt compliance and resume-token outcomes when interrupts fire

Outputs per run directory:
- `trace.jsonl`
- `leaderboard.tsv`
- `summary.json`
- `report.json` (after `specguard-chem report`)

## 5) Data Contracts
- **Spec schema**: strict enum-driven checks, typed params, `additionalProperties: false`.
- **Task schema**: expected action, budgets, task family, evidence, interrupt behavior.
- **Failure vector**: legacy summaries plus `constraint_results` with per-constraint status, property margins, and alert hits.

Legacy v1 specs are migrated into v2 internal representation at load time.

## 6) Metrics (high level)
Report summary includes:
- decision-level confusion/utility
- budget-first metrics (`pass_at_steps`, steps/tool economy)
- risk-coverage and cost-coverage curves from `p_hard_pass`
- calibration (`brier_score`, `ece`)
- hard-vs-soft conditional metrics
- interrupt/resume metrics
- edit economy (string + BRICS, final + trajectory)
- gaming resistance (`invariance_failure_rate`, boundary precision rates)
- schema robustness (`schema_error_rate`, invalid action/tool-call rates)
- per-family and per-split slices

See `METRICS.md` for exact definitions.

## 7) Current Built-in Suites
- `basic_plain`, `basic_checklist`
- `repair_ladder_plain`, `repair_ladder_checklist`
- `interrupts`, `interrupt_strict`, `interrupt_resume`
- `alerts_pains_soft`
- `smiles_invariance`
- `boundary_precision`

## 8) CLI Surface
Primary commands:
- `run`
- `report`
- `build-corpus`
- `generate-tasks`
- `validate-dataset`
- `run-baselines`
- `compare-baselines` (supports `--group-by`)

## 9) Reproducibility and Scope
- deterministic seeds and strict schemas
- no web/external-service dependency for benchmark correctness
- report metadata captures versions and hashes (taskset/spec-family/corpus when available)
- scope guardrails are defined in `SAFETY.md`
