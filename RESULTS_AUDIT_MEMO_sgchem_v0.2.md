# Results Audit Memo — sgchem_v0.2
Date: 2026-02-25
Data sources:
- `runs/paper_sweeps/sgchem_v0.2_test/aggregate.json`
- `runs/paper_sweeps/sgchem_v0.2_test/*/report.json`
- `paper/tables/topline_summary.*`
- `paper/metrics_summary.md`

## Executive assessment
`sgchem_v0.2` is no longer trivial for solver-shaped baselines in the way v0.1 was. The benchmark now differentiates baselines on pass rate, edit economy, boundary robustness, interrupt/resume behavior, and L3 verify usage. However, two credibility gaps remain:
1) invariance is still non-discriminative (`invariance_failure_rate=0.0` for all baselines), and
2) retrieval baselines can dominate tool-forced suites without verify usage, so closed-book and retrieval tracks must be reported separately.

## 1) Non-triviality check
### Evidence
- `pass@1` (expected ACCEPT tasks):
  - `heuristic`: **0.152**
  - `abstention_guard`: **0.130**
  - `verify_first`: **0.174**
  - `local_mutation`: **0.246**
  - `corpus_search`: **0.692**
- No baseline is `pass@1=1.0`.
- `avg_steps_to_accept` differs (`heuristic=1.00`, `verify_first=1.45`, `local_mutation=1.26`), showing nontrivial multi-step behavior.
- Hard-violation rates spread materially (`0.278` to `0.811`).

### Interpretation
This is now a benchmark, not a unit-test-like always-pass harness. Difficulty is concentrated in closed-book baselines; retrieval remains much easier.

## 2) Does tool gating matter (L2 vs L3)?
### Evidence
- `verify_first` now uses the tool:
  - overall `avg_verify_calls_used = 0.601`
  - on L3 expected-ACCEPT tasks: `l3_avg_verify_calls_used = 1.0`
- Other baselines: `avg_verify_calls_used = 0`.
- Tool-forced L3 suite (`task_family=tool_forced_l3`, n=36):
  - `heuristic`: **0.000** accept
  - `abstention_guard`: **0.000** accept
  - `verify_first`: **0.083** accept
  - `local_mutation`: **0.083** accept
  - `corpus_search`: **0.583** accept (retrieval baseline)

### Interpretation
Tool usage is now measurably exercised (good). `verify_first` beats weak non-verify baselines on tool-forced tasks, but retrieval dominates with zero verify calls, so the gating story is confounded unless retrieval is separated.

## 3) Safety-relevant tradeoffs (abstention utility + sensitivity)
### Evidence
- Default abstention utility (higher/less-negative is better):
  - `corpus_search`: **-470**
  - `abstention_guard`: **-622**
  - `local_mutation`: **-678**
  - `verify_first`: **-678**
  - `heuristic`: **-690**
- Utility sensitivity ranges show baseline rank order is fairly stable across cost settings.

### Interpretation
There are real tradeoffs: conservative abstention avoids some unsafe accepts but can still underperform on total utility when feasible accepts are missed. Retrieval still leads utility, reinforcing track separation needs.

## 4) Calibration sanity check
### Evidence
- Brier / ECE:
  - `corpus_search`: **0.217 / 0.128** (best ECE)
  - `abstention_guard`: **0.205 / 0.367**
  - `verify_first`: **0.326 / 0.308**
  - `local_mutation`: **0.371 / 0.368**
  - `heuristic`: **0.416 / 0.608** (worst)
- Risk-coverage (expected ACCEPT):
  - `verify_first` improves risk only with very aggressive thresholding (coverage collapses by `t=0.8`).
  - `abstention_guard` shows sharp coverage drop with risk improvement.

### Interpretation
Calibration is partially meaningful for some baselines, but confidence quality is still uneven. Calibration claims should remain qualified.

## 5) Gaming resistance bite (invariance + boundary)
### Evidence
- Invariance failure: **0.0 for all baselines**.
- Boundary precision failure rate:
  - `heuristic`: **1.000**
  - `abstention_guard`: **1.000**
  - `verify_first`: **0.647**
  - `local_mutation`: **0.647**
  - `corpus_search`: **0.294**

### Interpretation
Boundary suite now has bite and differentiates baselines. Invariance remains non-discriminative and should not be overclaimed as robust gaming resistance.

## 6) Interrupt/resume discrimination
### Evidence
- `resume_success_rate`:
  - `heuristic`: **0.000**
  - `abstention_guard`: **0.000**
  - `verify_first`: **0.200**
  - `local_mutation`: **0.200**
  - `corpus_search`: **0.933**
- `avg_extra_steps_after_interrupt`:
  - `heuristic`: **0.000**
  - `abstention_guard`: **0.000**
  - `verify_first`: **1.800**
  - `local_mutation`: **1.733**
  - `corpus_search`: **0.133**

### Interpretation
Interrupt/resume now measures something real: non-zero extra-step costs and clear baseline spread.

## Decision rules (If X, then Y)
1) If retrieval baselines are included in topline, then publish **separate leaderboards**:
- closed-book track (`heuristic`, `abstention_guard`, `verify_first`, `local_mutation`)
- retrieval track (`corpus_search`)

2) If `invariance_failure_rate` remains 0.0 across all baselines in the next run, then in v0.3:
- add stereo-sensitive equivalent forms,
- add tautomer/resonance-equivalence cases with explicit policy,
- add stricter “output must remain equivalent to input” checks.

3) If tool-forced closed-book pass rate stays <0.20 for all closed-book baselines, then in v0.3:
- rebalance task hardness (slightly wider steering windows or one extra step budget),
- keep verifier-call requirement explicit,
- retain the current tool-forced suite as a hard subset.

4) If any closed-book solver approaches `pass@1 > 0.9`, then raise similarity/precision hardness in the next cut.

## Recommendation
Promote `sgchem_v0.2` as the current benchmark release candidate with clear caveats:
- non-triviality, tool-call accounting, boundary robustness, and interrupt/resume discrimination are materially improved over v0.1;
- invariance robustness and retrieval-vs-closed-book separation still need explicit handling for strongest paper claims.
