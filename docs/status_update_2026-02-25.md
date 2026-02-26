# SpecGuard-Chem Status Update #1 (Supervisor Brief)

**Meeting date:** Wednesday, February 25, 2026  
**Prepared on:** Thursday, February 26, 2026 (post-meeting update)

## 1) Summary (1 minute)
- What exists now: an end-to-end benchmark harness that tests whether an “agent” follows explicit medicinal-chemistry-style rules (constraints) under limited interaction budgets.
- What it outputs: machine-verifiable pass/fail/abstain decisions plus detailed traces and a report with compliance, efficiency, calibration, and safety metrics.
- Why it matters: it turns “did the model follow the rules?” into something measurable, repeatable, and comparable across models/adapters.

## 2) Context: The Problem We’re Solving
When we ask an LLM (or agentic system) to follow a set of constraints, it can:
- confidently violate a rule,
- produce invalid outputs,
- misuse tools, or
- fail to abstain when a task is infeasible or unsafe.

SpecGuard-Chem is designed to evaluate those behaviors with strict, offline, deterministic checks, so results don’t depend on subjective human grading.

## 3) What SpecGuard-Chem Is (And Is Not)
**Is:**
- A model-agnostic benchmark harness for rule-following under explicit, machine-checkable chemistry constraints.
- A way to measure abstention, interrupt handling, and tool-use efficiency under budgets.

**Is not:**
- Drug discovery, activity/toxicity prediction, or synthesis planning.
- A system that recommends real compounds for real-world use.

## 4) Publishability Framing (A* Bar)
For an A*-level journal/forum submission, the bar is: a clear scientific claim + strong, reproducible evidence (not only software).

Proposed paper story:
- Research question: how reliably do agentic LLM systems follow explicit, machine-checkable chemistry constraints under limited budgets and varying tool access, and when do they appropriately abstain?
- Contributions: an end-to-end, offline-deterministic benchmark harness (Specs → Tasks → Runner → Verifiers → Report) plus deterministic dataset generation + dataset validation (executable “methods”), and a **frozen, checksummed benchmark release artifact** (`sgchem_v0.3`).
- Expected results/figures: track-separated leaderboards (closed-book vs retrieval); violation vs abstention tradeoffs; efficiency under budgets (steps/tool calls); calibration of `p_hard_pass` (ECE/Brier + risk/cost coverage curves); interrupt/resume compliance; gaming-resistance checks (adversarial invariance subfamilies + boundary precision).
- Reproducibility and auditability: benchmark release manifest + SHA256 checksums; seeded generation; `corpus_sha256`/`taskset_sha256` in `*.meta.json`; report metadata includes RDKit/Python/platform + git commit/dirty; runs emit `trace.jsonl`; optional external snapshots support cache+offline replay.
- A* readiness gaps to close next: run multiple strong LLM agents (snapshotted) and do a focused ablation plan; strengthen the causal “verify helps” story (tool-forced L3 effect size); tighten threats-to-validity language around retrieval confounds and dataset design.
- Scope discipline: we explicitly avoid claims about biological activity/toxicity/clinical utility or synthesis planning; this is a rule-following evaluation benchmark.

## 5) What’s Been Built So Far (Concrete Capabilities)
You can do these today via CLI:
- Run benchmark suites end-to-end: `specguard-chem run …` writes traces + summaries.
- Produce a metrics report: `specguard-chem report …` writes `report.json`.
- Generate/validate deterministic datasets: `build-corpus`, `generate-tasks`, `validate-dataset`.
- Run a baseline matrix and compare baselines: `run-baselines`, `compare-baselines`.
- Freeze and evaluate a **frozen benchmark release**: `freeze-benchmark`, `run-benchmark`, and generate paper-ready outputs with `paper-figures`.

Core components implemented:
- **Specs**: typed, strict, versioned spec format (“these are the rules”).
- **Tasks**: protocol-tagged episodes with budgets, expected action, and (optionally) interrupts.
- **Runner**: orchestrates L1/L2/L3 interaction loops, budgets, hard-gating, tool-calls, and logs.
- **Verifiers (RDKit-backed)**: deterministic checks for properties, alerts (PAINS/BRENK), SA proxy, canonicalization, similarity/edit metrics.
- **Adapters**: pluggable agent implementations for demo/baselines and integration tests.
- **Scoring/reports**: compliance + safety + calibration + efficiency metrics with breakdowns and curves.

## 6) How It Works (Plain-Language Mental Model)
Inputs:
- **SMILES**: a text format for describing molecules (e.g., `"CCO"`).
- **Spec**: the rule-set (e.g., “MW between X and Y”, “no PAINS alerts”).
- **Task**: a prompt + budgets + expected action (accept/reject/abstain).

Pipeline:
1. Runner selects a task + resolves the full spec.
2. Runner calls an adapter (the “agent”) with task/spec context.
3. Adapter responds with one of: propose a SMILES, call a verifier tool (L3 only), or abstain.
4. Verifiers deterministically check constraints with RDKit.
5. Runner enforces budgets and hard-gates acceptance (only hard-pass proposals can be accepted).
6. Runner writes trace artifacts; `report` aggregates metrics over the run.

Protocol ladder (why there are L1/L2/L3 modes):
- `L1`: one-shot proposal (no feedback).
- `L2`: iterative repair with *coarse* feedback (which constraints failed, not full details).
- `L3`: same as L2, plus explicit `verify(smiles)` tool calls that return full constraint-level failure vectors.

This setup makes tool usage measurable and prevents “free” detailed feedback in non-tool protocols.

## 7) How It Was Implemented (Engineering Approach)
Key design choices:
- Deterministic, offline correctness: no web calls required for benchmark truth.
- Strict schemas/contracts: specs and tasks are validated and versioned; legacy v1 specs are migrated to v2 internal form at load time.
- Budgeted interaction: max steps/proposals/tool calls are enforced and logged.
- Hard safety gating: acceptance is only allowed after a hard-pass verifier check.
- Robustness logging: malformed/invalid adapter outputs are normalized to abstain and recorded (schema/invalid-action/tool-call rates).
- Reproducibility: reports embed environment info (RDKit/Python/platform), git commit/dirty state, and hashes/IDs for datasets/specs when present.

## 8) Evidence It’s Working (As of Feb 26, 2026)
Code health:
- Tests: `76 passed` (`uv run pytest`).
- Coverage: `82%` overall (`uv run pytest --cov=src/specguard_chem`).
- Verifier branch coverage: `83%` (`uv run pytest --cov=src/specguard_chem/verifiers --cov-branch`).
- CI: runs lint/tests/coverage plus smoke runs for `run`, `report`, and `run-baselines`.

Benchmark evidence (paper pipeline exists and runs end-to-end):
- Frozen release `sgchem_v0.3` exists with manifest + checksums (`benchmarks/releases/sgchem_v0.3/MANIFEST.json`).
- Track-separated paper outputs were generated under `paper/` (tables + figures) from `runs/paper_sweeps/sgchem_v0.3_test`.
- Current audit memo conclusion: `sgchem_v0.3` is benchmark-credible as a hardening release; main residual paper risk is tool-gating effect size, not harness integrity (`RESULTS_AUDIT_MEMO_sgchem_v0.3.md`).

## 9) What’s Not Done Yet (Risks / Gaps)
- Baseline behavior gaps (still meaningful for publishability):
  - Interrupt/resume success remains low in closed-book baselines (some at 0.0), so “agent robustness” is not solved yet (it is measurable, which is the point).
- Paper-claim risk:
  - Tool-gating is exercised, but the causal “verify helps” lift on tool-forced L3 is still modest; this needs either benchmark hardening or stronger tool-using baselines.
- Quality gates (per internal DoD intent) are not yet met:
  - Overall coverage is `82%`.
  - Verifier branch coverage is `83%` (below a 95% target).
  - CLI coverage is relatively low compared to core runner/scoring.

## 10) Timeline (Since Ideation, High Level)
- January 13, 2026: runner/protocol semantics landed (L1/L2/L3), suite variants (plain/checklist), metrics/doc alignment.
- February 16, 2026: scoring updates (calibration `p_hard_pass`, spec formula fixes, failure vector extensions).
- February 20, 2026: dataset pipeline (corpus/tasks/validator), baseline orchestration commands, and documentation refresh.
- February 25, 2026: sgchem_v0.1→v0.3 hardening cycle: frozen releases, track separation, adversarial invariance, cache+replay, bootstrap CIs, and paper figures/tables.

## 11) Plan Before Next Update
- Decide the “A* claim set” and lock an experiment matrix (which models count as the headline comparisons, and which are appendix).
- Run a larger closed-book vs retrieval sweep on `sgchem_v0.3` and freeze the paper tables/figures for a stable draft.
- Strengthen the tool-forced L3 story (either benchmark hardening or better tool-using baselines) and rerun.
- Add targeted branch tests for verifiers and key benchmark/CLI flows while keeping CI time bounded.

## 12) Decisions / Input Requested From Supervisor
- Paper direction: primary goal is a benchmark/methods submission (artifact + evaluation) vs a broader “agent reliability” story with stronger LLM comparisons.
- Scope of claims: confirm we should treat retrieval as an upper-bound track (separate leaderboard) and keep external APIs as optional cached snapshots.
- Prioritization: tighten tool-gating causality + baseline behavior first vs expand dataset/spec diversity first.
- Acceptance thresholds for next milestone: minimum experimental sweep size, minimum coverage, and a short “threats-to-validity” checklist we must satisfy.

---

## Appendix A) Quick Demo Script (Optional)
Run a small demo end-to-end:

```bash
uv run specguard-chem run basic_plain --protocol L1 --model heuristic --limit 5 --run-path runs/demo_supervisor_basic
uv run specguard-chem report runs/demo_supervisor_basic
```

Key artifacts produced:
- `trace.jsonl`: step-by-step interaction and verifier outcomes
- `leaderboard.tsv`: task-level summary
- `summary.json`: small aggregate
- `report.json`: full metrics + breakdowns + metadata

## Appendix B) Where Things Live (Optional)
- Overview: `README.md`, `docs/overview.md`, `SPEC.md`, `METRICS.md`, `SAFETY.md`
- Specs: `data/specs/*.yaml`
- Task suites: `tasks/suites/*.jsonl`
- Implementation: `src/specguard_chem/`
