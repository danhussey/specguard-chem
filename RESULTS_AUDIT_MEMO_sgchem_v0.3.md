# Results Audit Memo — sgchem_v0.3
Date: 2026-02-25

Inputs audited:
- `benchmarks/releases/sgchem_v0.3/MANIFEST.json`
- `runs/paper_sweeps/sgchem_v0.3_test/aggregate.json`
- `runs/paper_sweeps/sgchem_v0.3_test/*/report.json`
- `runs/paper_sweeps/sgchem_v0.3_external/aggregate.json`
- `runs/paper_sweeps/sgchem_v0.3_external_replay/aggregate.json`
- `paper/metrics_summary.md`, `paper/tables/*`, `paper/figures/*`

## 1) Is the benchmark non-trivial?
Short answer: yes for the primary closed-book track.

Closed-book TEST results (`n=308`) are clearly separated:
- `heuristic`: pass@1=0.155, pass@3=0.155, hard_violation_rate=0.775
- `abstention_guard`: pass@1=0.129, pass@3=0.144, hard_violation_rate=0.812
- `verify_first`: pass@1=0.173, pass@3=0.309, hard_violation_rate=0.623
- `local_mutation`: pass@1=0.248, pass@3=0.309, hard_violation_rate=0.623

Retrieval remains much easier:
- `corpus_search` (retrieval track): pass@1=0.759, pass@3=0.759, hard_violation_rate=0.218

Interpretation:
- This is no longer a unit-test-like benchmark where all solver-shaped baselines pass instantly.
- Track separation is necessary and now present: retrieval should remain an upper-bound track, not combined into primary ranking.

## 2) Does tool gating matter (L2/L3)?
Partially yes; tooling is exercised, but causal lift is still modest.

Evidence:
- `verify_first` has `l3_avg_verify_calls_used=1.0`, `verify_usage_rate_on_L3=1.0`.
- Non-tool baselines remain at `0.0`.
- On `tool_forced_l3` tasks (`n=36`, all expected ACCEPT):
  - `heuristic`: accept_rate=0.000, avg_verify=0.0
  - `verify_first`: accept_rate=0.083, avg_verify=1.0
  - `local_mutation`: accept_rate=0.083, avg_verify=0.0

Interpretation:
- L3 verify usage is real (not dead code).
- Current `tool_forced_l3` suite shows verify behavior, but not a large performance separation versus strong non-tool mutation yet.

## 3) Are there safety-relevant tradeoffs?
Yes, and the utility matrix exposes them.

Observed utility and safety pattern:
- `abstention_guard`: lower accept rate (0.179), high violation (0.812), utility `-626`
- `heuristic`: utility `-692`, high violation (0.775)
- `verify_first` / `local_mutation`: better pass/violation profile, but utility still negative (`-684`)
- `corpus_search` retrieval: best utility (`-434`) but different track

Interpretation:
- Decision tradeoffs are measurable.
- Under the default cost matrix, utility remains negative across all baselines; this should be framed as current benchmark pressure, not a solved abstention policy story.

## 4) Calibration sanity check
Calibration is mixed and baseline-dependent.

Closed-book:
- `verify_first`: Brier=0.324, ECE=0.308
- `local_mutation`: Brier=0.374, ECE=0.373
- `heuristic`: Brier=0.415, ECE=0.607 (poor)

Retrieval:
- `corpus_search`: Brier=0.175, ECE=0.068

Interpretation:
- Confidence-quality claims should stay scoped to specific baselines.
- Keep reliability/risk-coverage plots, but avoid blanket “well calibrated” language.

## 5) Gaming resistance has bite?
Yes, now including adversarial invariance subfamilies with charge tasks in TEST.

Invariance overall:
- Closed-book baselines: invariance_failure_rate=0.233
- Retrieval baseline: invariance_failure_rate=0.000

Invariance subfamily behavior (closed-book):
- `aromatic`: 0.0 failures
- `stereo`: 0.0 failures
- `tautomer`: 0.5 failures
- `charge`: 0.5 failures

Boundary precision:
- `heuristic` / `abstention_guard`: boundary_precision_failure_rate=1.0
- `verify_first` / `local_mutation`: 0.706
- `corpus_search`: 0.118

Interpretation:
- Boundary suite is strongly discriminative.
- Invariance now has non-zero failure and includes `charge` in TEST, but difficulty is concentrated in tautomer/charge; stereo/aromatic are mainly regression/sanity checks.

## 6) Interrupt/resume actually tests anything?
Yes.

- `heuristic` / `abstention_guard`: resume_success_rate=0.0, extra_steps_after_interrupt=0.0
- `verify_first`: resume_success_rate=0.2, extra_steps_after_interrupt=1.8
- `local_mutation`: resume_success_rate=0.2, extra_steps_after_interrupt=1.733
- `corpus_search`: resume_success_rate=0.933, extra_steps_after_interrupt=0.133

Interpretation:
- Interrupt/resume is no longer degenerate and creates measurable behavior differences.

## Additional v0.3 reproducibility checks
- External cache/replay works on v0.3 snapshot run (`limit=25`):
  - `aggregate.json` live vs replay: identical (zero diff)
  - per-baseline `report.json` live vs replay: identical
- L3 verify usage appears in external verify policy baseline (`openai_chat_verify_l3`):
  - `l3_avg_verify_calls_used=1.0`
  - `verify_usage_rate_on_L3=1.0`

## Confidence intervals
Bootstrap CIs are present in `aggregate.json` for key metrics.
Example pass@1 95% CI:
- `heuristic`: [0.113, 0.195]
- `abstention_guard`: [0.091, 0.168]
- `verify_first`: [0.126, 0.217]
- `local_mutation`: [0.196, 0.299]
- `corpus_search`: [0.709, 0.809]

## Decision block (If X, then Y)
1. If primary paper claim is solver discrimination, then keep `closed_book` as the headline leaderboard and show retrieval separately as upper bound.
2. If we need a strong “verify causes lift” claim, then harden `tool_forced_l3` in v0.4 until verify-using baselines outperform non-tool baselines by a clear margin under fixed budgets.
3. If calibration is a headline claim, then require baseline-level calibration gates (ECE/Brier thresholds) before broad wording.
4. If invariance is a headline claim, then increase stereo/charge adversarial diversity so failures are not concentrated only in tautomer/charge edge cases.
5. If release timing is now, then ship v0.3 as a credible hardening release with explicit limitations and a focused v0.4 tool-gating/invariance-strengthening roadmap.

## Bottom line
`sgchem_v0.3` is benchmark-credible as a hardening release: it is non-trivial in closed-book mode, enforces track separation, exercises L3 verify paths, includes adversarial invariance with explicit policies (including charge), and provides reproducible cache/replay plus bootstrap CIs. The main residual paper risk is effect size on tool-forced L3 causality, not harness integrity.
