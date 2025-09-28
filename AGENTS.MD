# AGENTS — Build Orchestration

## Roles
1) **Planner**: expands SPEC.md into tasks/issues; maintains scope guardrails.
2) **Verifier Engineer**: implements RDKit verifiers (pure, unit-tested).
3) **Runner Engineer**: protocols, failure vectors, gating, logging.
4) **Adapter Engineer**: model adapters (base interface + demo adapters).
5) **Scoring Engineer**: metrics (compliance, ECE/Brier, reports).
6) **Docs Engineer**: README/CLI help/examples; METRICS; SAFETY.

## Global constraints
- Python 3.11+, **rdkit** (PyPI wheels) only; no docking/toxicity/activity predictors.
- **Do not** implement synthesis planning.
- All modules must be deterministic, pure, and unit-tested.
- No external web calls; offline only.

## Definitions of Done
- Verifiers: functions + tests (property bounds, PAINS) with ≥95% branch coverage on test data.
- Runner: L1/L2/L3 executable; emits JSON traces and TSV leaderboard.
- Scoring: metrics match METRICS.md formulas; includes abstention cost curves.
- Docs: `README.md` runnable examples; `SPEC.md` & `SAFETY.md` complete.

## Step plan
1) **Scaffold** (Planner): create file tree; write pyproject, Makefile, CI, pre-commit.
2) **Schemas** (Verifier/Runner): implement Pydantic models for spec/task/failure_vector.
3) **Verifiers** (Verifier Eng): properties, alerts, SA; unit tests.
4) **Runner** (Runner Eng): protocols L1/L2/L3; failure vector; tool-gated finalize.
5) **Adapters** (Adapter Eng): `BaseAdapter`; `heuristic_mutator`; `open_source_example`.
6) **Scoring** (Scoring Eng): compliance, ECE/Brier, reports; CLI `report`.
7) **Docs & Safety** (Docs Eng): examples, metrics doc, safety guardrails.
8) **Smoke CI**: run 5 tasks in CI; enforce < 120s runtime.

## Commit conventions
- Prefix by area: `verifiers: …`, `runner: …`, `scoring: …`, `docs: …`.
- Include unit tests in same PR.
