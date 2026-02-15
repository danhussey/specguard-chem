# SpecGuard-Chem
[![CI](https://img.shields.io/github/actions/workflow/status/danhussey/specguard-chem/ci.yml?label=CI)](https://github.com/danhussey/specguard-chem/actions/workflows/ci.yml) ![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen) 
Spec-driven, programmatically verifiable evaluation of agentic LLMs on safe medicinal-chemistry constraints.

**What it is:** a small, model-agnostic test rig. Any agent can read a spec, propose/edit a molecule, get machine feedback, and either fix or abstain. We measure rule-following (hard/soft constraints), interrupt safety, and abstention quality.

**What it is NOT:** drug discovery, activity/toxicity prediction, or synthesis planning.

PAINS-style alerts use a minimal motif subset and are treated as soft constraints in the base spec.

## Quickstart
```bash
uv venv --seed
source .venv/bin/activate
uv pip install -e .[dev]
specguard-chem run --suite basic_plain --protocol L1 --model heuristic
specguard-chem run --suite basic_plain --protocol L3 --model open_source_example
specguard-chem report --run-path runs/2025-01-01_basic_plain_L3/
uv run pytest --cov=src/specguard_chem --cov-report=term-missing
```

Repro in <10 minutes on a laptop. No proprietary data; all tasks are synthetic and safe.

`specguard-chem report` summarises spec compliance, abstention behaviour, edit economy, and
calibration metrics (Brier/ECE over `p_hard_pass`) from the generated `trace.jsonl` artefacts and writes
`report.json` into the run directory with embedded definitions for expected/observed outcomes.


For adapter integration details see [docs/adapters.md](docs/adapters.md).
For a narrative tour of the system architecture, see [`docs/overview.md`](docs/overview.md).

### Included adapters

- `heuristic` – deterministic mutator that iteratively repairs hard failures using the runner's
  failure vector feedback in L2/L3 protocols.
- `open_source_example` – demonstrates tool calls during L3 protocols.
- `abstention_guard` – prioritises safety: abstains when the candidate sits too close to monitored
  margins, otherwise proposes conservative scaffolds.
- `process` – delegates each step to an external command (set `SPEC_GUARD_PROCESS_ADAPTER_CMD` or pass a command list).
- `openai_chat` – calls the OpenAI Chat Completions API; set `OPENAI_API_KEY` or inject a client.

### Task suites

- `basic_plain` – mixed L1/L2/L3 tasks (10 total) with plain prompts covering single-shot proposals,
  repair rounds, and verify-in-the-loop flows.
- `basic_checklist` – prompt-variant of `basic_plain` that asks for a checklist-first response.
- `repair_ladder_plain` – small repair-focused suite with plain prompts.
- `repair_ladder_checklist` – prompt-variant of `repair_ladder_plain` with checklist-first prompts.
- `interrupts` – focused interrupt-handling scenarios that trigger pauses mid-protocol and expect safe abstention.
- `alerts_pains_soft` – PAINS-style alert reporting tasks (soft constraints; not hard-gated).

### Continuous integration

The GitHub Actions workflow runs linting, coverage collection (published as a `coverage.xml`
artifact), and smoke evaluations on both `basic_plain` and `interrupts` suites using the built-in adapters.
