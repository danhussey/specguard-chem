# Results Audit Memo — sgchem_v0.1 (test split)

Date: 2026-02-25  
Data sources: `runs/paper_sweeps/sgchem_v0.1_test/aggregate.json`, `runs/paper_sweeps/sgchem_v0.1_test/*/report.json`, `paper/tables/topline_summary.md`, `paper/figures/*`  

## Executive Takeaway
sgchem_v0.1 is **discriminative for weak baselines** but **trivial for solver‑shaped baselines** (`corpus_search`, `local_mutation`) on the held‑out TEST split. Tool‑gating (L2 vs L3) is not exercised because no baseline uses `verify`, calibration quality is mixed, and gaming‑resistance suites appear too easy (zero invariance/boundary failures across all baselines).  

**Recommendation:** treat sgchem_v0.1 as a functional harness release, not yet a publishable benchmark. Prioritize a v0.2 dataset expansion that hardens task difficulty, removes corpus leakage, and forces L3 tool usage before claiming “A*‑credible.”

## 1) Non‑triviality Check
From `paper/tables/topline_summary.md`, pass@step 1 and accept rates:
- `heuristic`: pass@1 = **0.340**, accept_rate = **0.400**, hard_violation_rate = **0.569**
- `abstention_guard`: pass@1 = **0.277**, pass@3 = **0.702**, accept_rate = **0.673**, hard_violation_rate = **0.275**
- `corpus_search`: pass@1 = **1.000**, accept_rate = **1.000**, hard_violation_rate = **0.000**
- `local_mutation`: pass@1 = **1.000**, accept_rate = **1.000**, hard_violation_rate = **0.000**

Interpretation:
- The benchmark is **not trivial** for weaker baselines (heuristic, abstention_guard): pass@1 far below 1.0, and violations are substantial for `heuristic`.
- The benchmark **is trivial** for solver‑shaped baselines: both non‑LLM baselines hit 100% pass@1 and 0 hard violations, implying the test split is solvable by deterministic corpus/mutation search.

Conclusion: The benchmark **fails the “non‑trivial” bar** for solver‑shaped baselines. This is likely corpus leakage: the same corpus used to generate tasks is accessible to `corpus_search` and to `local_mutation` via simple transforms.

## 2) Tool Gating (L2 vs L3)
From per‑baseline `report.json`:
- `avg_verify_calls_used = 0` for all baselines (no tool calls).
- `accept_rate_by_protocol`:
  - `heuristic`: L1 0.8125, L2 0.2222, L3 0.25
  - `abstention_guard`: L1 0.625, L2 0.6667, L3 0.75
  - `corpus_search`, `local_mutation`: L1/L2/L3 all 1.0

Interpretation:
- L3 is **not actually exercising tool gating**; no baseline calls `verify`.
- L2/L3 differences are therefore due to protocol budgets or task distribution, not verifier‑call availability.

Conclusion: **Tool gating is currently untested**. This is a red flag for the “verifier economy under L3” claim.

## 3) Safety‑Relevant Tradeoffs (Abstention vs Unsafe Accepts)
From `paper/tables/topline_summary.md` and `report.json` utility sensitivity grids:
- `abstention_guard` has **better utility** (−68) than `heuristic` (−120), while also having **lower hard violation rate**. This is not a tradeoff; it is domination.
- `corpus_search` / `local_mutation` have **hard_violation_rate = 0** but **abstention_utility = −80**, and their utility sensitivity worsens when `C_ACCEPT_INFEASIBLE` increases. This indicates they are accepting on tasks labeled `expected_action=ABSTAIN` (unsafe completion) even if they satisfy hard constraints.

Conclusion: The expected “safety vs performance” tradeoff is **not visible**. The dataset is easy enough that strong baselines can both accept widely and avoid hard violations; utility is driven primarily by label misalignment (accepting on abstain‑labeled tasks).

## 4) Calibration Sanity Check
From `paper/tables/topline_summary.md` and risk–coverage curves in `report.json`:
- `heuristic`: Brier 0.357, ECE 0.533 (poor calibration).
  - Risk–coverage worsens at mid thresholds (risk spikes to 1.0 at ~0.55 coverage), indicating uninformative/confused confidence.
- `abstention_guard`: Brier 0.242, ECE 0.235 (better but still coarse).
  - Risk–coverage improves somewhat as coverage drops, but curve is step‑like (few distinct confidence levels).
- `corpus_search` / `local_mutation`: Brier 0.0225 / 0.0625, ECE 0.15 / 0.25.
  - Risk is 0 across thresholds because they already pass everything; calibration is not meaningful in that regime.

Conclusion: **Calibration story is weak** in v0.1. There are signals for `abstention_guard`, but confidence is coarse and not tied to actual uncertainty in L3 contexts.

## 5) Gaming Resistance (Invariance + Boundary Precision)
From `paper/tables/topline_summary.md`:
- Invariance failure rate: **0.0** for all baselines.
- Boundary precision failure rate: **0.0** for all baselines.

Conclusion: Either the suites are too easy or the baselines are too robust. This does **not demonstrate resistance to gaming** yet. It should not be claimed as a differentiator at v0.1.

## 6) Interrupt / Resume Tests
From `paper/tables/topline_summary.md` and `report.json`:
- `resume_success_rate`: 0.0 for `heuristic` and `abstention_guard`, 1.0 for `corpus_search` and `local_mutation`.
- `avg_extra_steps_after_interrupt`: 0.0 across baselines.

Conclusion: The interrupt/resume suite is **not discriminative** and likely too small or too trivial (no extra step cost). This does not yet test real robustness.

## Decision Points (If X, then Y)
1. **If any non‑LLM baseline has pass@1 ≥ 0.9 on TEST**, then:
   - **Do**: v0.2 dataset hardening. Separate corpus for generation vs evaluation, add hold‑out families not reachable via corpus, and increase “repair” complexity.  

2. **If `avg_verify_calls_used` == 0 across baselines in L3**, then:
   - **Do**: add L3 tasks requiring verification (e.g., ambiguous near‑misses) and add at least one baseline that calls `verify` by default.  

3. **If invariance/boundary failure rates are 0 across baselines**, then:
   - **Do**: strengthen gaming suites with epsilon‑over‑bound tasks and mixed SMILES variants that aren’t trivially canonical.  

4. **If calibration ECE > 0.25 for baseline(s) meant to claim calibration**, then:
   - **Do**: introduce more granular confidence outputs (or remove calibration claims until baselines produce meaningful `p_hard_pass`).  

## Recommendation
Classify sgchem_v0.1 as **“functional harness release”** rather than “A*‑credible benchmark.”  
Proceed with a **v0.2 dataset expansion** before making strong claims:
- enforce harder test split with corpus hold‑out,
- require L3 verify usage,
- expand invariance/boundary/interrupt suites,
- and tune expected_action labeling to reveal real safety tradeoffs.

