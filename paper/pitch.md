# SpecGuard-Chem (sgchem_v0.3) — 1-Page Paper Pitch

## Provisional Title
SpecGuard-Chem: A Frozen Benchmark for Budgeted Constraint Following, Tool Use, and Abstention in Agentic Molecular Editing

## 150-Word Abstract (Draft)
Agentic LLM systems are increasingly asked to follow explicit constraints, use tools, and decide when to abstain. In chemistry-flavored settings, failures are safety-relevant: models can output invalid molecules, violate hard rules, or refuse to abstain on infeasible tasks. We present SpecGuard-Chem, a solver-agnostic benchmark harness that evaluates structured constraint compliance under strict interaction budgets with deterministic RDKit-backed verifiers. The benchmark supports a protocol ladder (L1 one-shot, L2 repair with coarse feedback, L3 tool-in-loop with explicit verify calls) and reports decision utility, verifier economy, calibration, interrupt/resume safety, and gaming resistance (boundary precision and adversarial SMILES invariance). We release a frozen benchmark artifact (sgchem_v0.3) with manifest checksums, train/dev/test splits, machine-checkable labels (witnesses for ACCEPT, contradiction proofs for ABSTAIN), and bootstrap confidence intervals. Initial results show clear separation on the closed-book track and motivate track-separated leaderboards for retrieval-enabled solvers.

## Positioning (Default)
- Audience: LLM/agent evaluation and reliability (benchmarks, tool use, abstention, robustness).
- Why chemistry: it supplies deterministic, offline verifiers and safety-relevant constraints, while keeping the core claim about budgeted rule-following general.
- Scope: no claims about biological activity/toxicity/clinical utility or synthesis planning; this is an evaluation benchmark.

## Core Claims (What We Can Defend)
- Structured, machine-checkable specs + budgets expose measurable differences in (i) hard-rule compliance, (ii) abstention safety/utility, and (iii) tool-use efficiency.
- Track separation is necessary: retrieval-enabled baselines materially change the difficulty regime and should be reported as an upper-bound track.
- Adversarial invariance, boundary precision, and interrupt/resume suites reveal non-trivial, safety-relevant failure modes that are invisible in pass-rate-only evaluations.

## Key Contributions (3)
- Benchmark design: strict spec/task contracts, explicit accept/reject/abstain decision semantics, and a protocol ladder that gates detailed feedback behind measurable L3 verifier calls.
- Frozen release + reproducibility: sgchem_v0.3 "freeze-benchmark" artifact with split files, checksums/manifest, dataset invariants (witnesses/proofs), and end-to-end offline evaluation (plus cache+replay for optional external snapshots).
- Metrics + analysis suite: pass@budget, violation rates, verifier economy, utility/cost curves, calibration (ECE/Brier), interrupt/resume success, and adversarial invariance subfamily breakdowns with bootstrap 95% CIs.

## Headline Results Snapshot (sgchem_v0.3, TEST, n=308)
- Track-separated pass@1 (bootstrap 95% CIs in paper tables): closed_book = {heuristic 0.155, abstention_guard 0.129, verify_first 0.173, local_mutation 0.248}; retrieval (upper bound) = {corpus_search 0.759}.
- Tooling is exercised in L3: verify_first has `verify_usage_rate_on_L3=1.0` and `l3_avg_verify_calls_used=1.0` (others are 0.0), enabling a measurable verifier economy axis.
- Test split includes tool-forced L3 tasks (n=36) to ensure verify paths are exercised.
- Robustness suites have bite: invariance_failure_rate is 0.233 (closed_book) vs 0.0 (retrieval); boundary_precision_failure_rate is 0.706–1.0 (closed_book) vs 0.118 (retrieval).
- Interrupt/resume is discriminative: resume_success_rate is 0.0 (heuristic/abstention_guard), 0.2 (verify_first/local_mutation), and 0.933 (corpus_search), with non-zero extra-step costs.

## 4 Key Figures (Paper-Ready)
- Fig 1: Track-separated pass@budget curves (closed-book vs retrieval) with bootstrap 95% CIs; includes pass@1 and pass@3.
- Fig 2: Safety tradeoffs: hard violation rate vs abstention utility (with sensitivity sweep) and cost-coverage curves from p(hard-pass).
- Fig 3: Tool gating and verifier economy: performance on tool-forced L3 tasks vs verify usage rate / avg verify calls; highlights causal question.
- Fig 4: Robustness suites: adversarial invariance failure rates by subfamily (stereo/tautomer/charge/aromatic), boundary precision failure, and interrupt/resume outcomes.

## Experiment Matrix (Minimal A* Storyboard)
| Axis | Setting |
| --- | --- |
| Benchmark | sgchem_v0.3 frozen release (1000 tasks; train/dev/test splits; manifest + checksums) |
| Tracks | closed_book (primary), retrieval (upper bound), external snapshot (optional; cache+replay) |
| Baselines | closed_book: heuristic, abstention_guard, verify_first, local_mutation; retrieval: corpus_search |
| Protocols | mixed L1/L2/L3; includes tool-forced L3 tasks to exercise verify paths |
| Primary metrics | pass@1, pass@3, hard_violation_rate, abstention_utility (with sensitivity), verify economy |
| Secondary metrics | calibration (ECE/Brier), edit economy, invariance/boundary robustness, interrupt/resume |
| Statistics | bootstrap 95% CIs in aggregate and paper tables; track-separated leaderboards |

## A* Risks / What We Still Need To Nail
- Tool-causality effect size: strengthen tool-forced L3 so verify-using baselines clearly outperform non-tool baselines under fixed budgets.
- External validity: evaluate multiple strong LLM agents (snapshotted with cache+replay) and show robustness across model families.
- Threats to validity: explicitly address corpus leakage, retrieval confounds, and spec/task distribution design choices.

## Repro (For Reviewers / Artifact Appendix)
```bash
# Freeze benchmark
specguard-chem freeze-benchmark --benchmark-id sgchem_v0.3 --out benchmarks/releases/sgchem_v0.3 --target-tasks 1000 --seed 7

# Run track-separated sweep
specguard-chem run-benchmark --benchmark benchmarks/releases/sgchem_v0.3 --split test --baselines baselines/paper_baselines.yaml --out runs/paper_sweeps/sgchem_v0.3_test

# Generate figures/tables
specguard-chem paper-figures --runs runs/paper_sweeps/sgchem_v0.3_test --out paper
```
