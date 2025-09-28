# SpecGuard-Chem — Conceptual Overview

This document gives a narrative tour of the SpecGuard-Chem stack so new contributors can orient quickly. It summarises the motivation, system layout, key abstractions, and the relationships between data, runner, adapters, and scoring.

## 1. Purpose
- Provide a reproducible harness to evaluate chemistry-oriented LLM agents against clearly defined property specs.
- Focus on compliance and safety (hard/soft constraints, interrupts, and abstentions) rather than discovery or synthesis planning.
- Remain model-agnostic: any agent that speaks the adapter protocol can plug into the runner.

## 2. Architecture at a Glance
1. **Data layer** — `data/specs/*.yaml` define constraint sets; `tasks/suites/*.jsonl` define evaluation tasks grouped into suites (e.g., `basic`, `interrupts`).
2. **Schemas & config** — `specguard_chem.config` loads/validates specs and tasks, exposing strongly typed Pydantic models.
3. **Runner core** — `specguard_chem.runner.runner.TaskRunner` orchestrates episodes across protocols L1/L2/L3, feeds failure vectors, handles interrupts, and logs structured traces.
4. **Verifiers** — `specguard_chem.verifiers.*` wraps RDKit descriptors, PAINS alerts, SA scoring, and SMILES canonicalisation used by the runner to judge proposals.
5. **Adapters** — concrete agents (`heuristic`, `open_source_example`, `abstention_guard`) subclass `BaseAdapter`, interpreting requests and crafting responses.
6. **Scoring & reports** — `specguard_chem.scoring` aggregates hard/soft outcomes, edit economy, abstention utility, and calibration metrics; the CLI surfaces summaries.

```
Tasks ──► Config/Schemas ──► Runner ──► Adapter
   ▲          │                │         │
   │          ▼                ▼         ▼
 Specs ◄── Verifiers ◄── Scoring/Reports ◄─┘
```

## 3. Protocols
- **L1 (Single-shot)**: one proposal; no feedback loop. Evaluates immediate spec comprehension.
- **L2 (Assisted repair)**: up to three rounds. Runner returns a failure vector after each proposal. Interrupts can fire mid-episode to test safety acknowledgements.
- **L3 (Tool-in-loop)**: agents may call `verify(smiles)` as a dry-run before finalising. Hard gating still applies on final submission.

## 4. Runner Mechanics
- Builds a `ConstraintEvaluator` per task/spec to compute properties, alerts, and synthetic accessibility.
- Generates a failure vector summarising hard failures, soft misses, and margins to bounds.
- Injects interrupt payloads on the configured round (`interrupt_at_step`) so adapters can respond deterministically.
- Logs each round (action, SMILES, evaluation, confidence, interrupt flag) and persists:
  - `trace.jsonl`: detailed round-by-round log.
  - `leaderboard.tsv`: per-task summary, including decision and confidence.
  - `summary.json`: aggregate stats (pass rate, spec score, rounds, edit distance).

## 5. Adapters
- **`heuristic`**: deterministic RDKit mutator that reacts to failure vectors (switches scaffolds, tweaks polarity) and cites triggered constraints.
- **`open_source_example`**: illustrative baseline that showcases tool-call behaviour in L3.
- **`abstention_guard`**: conservative agent that abstains when margins are too tight; otherwise proposes safe scaffolds.
- The registry in `specguard_chem.models` allows dynamic registration; adapters only need to implement `step(req: AgentRequest)`.

## 6. Scoring & Metrics
- **Hard pass / soft compliance** roll into the spec score (`specguard_chem.scoring.metrics.spec_compliance`).
- **Edit economy**: Levenshtein distance between input and final SMILES.
- **Abstention utility**: cost-weighted utility for accept/reject/abstain decisions with heavy penalties on false accepts.
- **Calibration**: Brier score and Expected Calibration Error computed over final confidences.
- CLI reports highlight `avg_spec_score`, violation rate, accept/abstain rates, rounds, edit economy, calibration, and abstention utility.

## 7. Task Suites
- **`basic`** (10 tasks): mixture of L1 proposals, L2 repairs, L3 verify-in-loop, and abstention prompts.
- **`interrupts`** (3 tasks): every task triggers an interrupt to test acknowledgement and recovery logic.
- Suites live under `tasks/suites/*.jsonl`; new suites must respect `tasks/schema.json` and should ship with targeted tests.

## 8. Safety & Reproducibility
- SAFETY.md codifies scope: no synthesis planning, no claims about biological activity, offline deterministic verifiers only.
- Reproducibility leans on fixed seeds (`utils.seeds`), `uv`-managed environments, CI smoke runs, and Docker.

## 9. Extending the Project
- Add new specs: drop YAML under `data/specs/` and update tests to cover constraints.
- Add suites: extend JSONL files, ensuring coverage in `tests/test_schemas.py`.
- Add adapters: subclass `BaseAdapter`, register with `register_adapter`, and supply tests validating behaviour.
- Enrich reports: modify `scoring/reports.py`, then update CLI and regression tests.

For implementation specifics or deep dives, see:
- `SPEC.md` for the authoritative engineering spec.
- `METRICS.md` for mathematical definitions.
- `SAFETY.md` for guardrails and scope.
