# SpecGuard-Chem sgchem_v1.0 Paper Pitch

## Provisional Title

SpecGuard-Chem: Oracle-Compiled Evaluation Contracts for Scientific Language Agents

## Core Claim

SpecGuard-Chem is an oracle-compiled evaluation contract for scientific language-agent specification compliance. The sgchem_v1.0 release contains 118 bundles, 656 tasks, and 266 test tasks compiled from machine-checkable witnesses and certificates.

## Why It Matters

The benchmark shows that molecule acceptance alone is misleading. Deterministic local and verify-first baselines reach molecule_acceptance_rate=0.846, while the retrieval upper bound reaches 0.868. Their action accuracies are 0.650, 0.650, and 0.673, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and task_inconsistent_accept_rate=0.598. A deterministic well_engineered_wrapper solves the full test split, so the paper should foreground verifier-wrapper saturation and action-aware evaluation rather than benchmark hardness.

## Scope

SpecGuard-Chem uses molecular property ranges, substructure constraints, alert filters, similarity guards, and representation-equivalence checks as a chemically typed substrate. It evaluates specification compliance, construction, repair, candidate audit, rejection, abstention, verifier-tool use, boundary precision, representation invariance, and interrupt/resume behavior.

SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.

## Artifact Status

- strict validation: valid=true, num_errors=0
- prompt leakage: zero leaks across 656 prompts
- oracle scrambling: passed
- negative controls: 18 corruptions covered
- clean-clone reproduction: passed from anonymous archive
- manual test-bundle review: A=36, B=0, C=0, D=0
- reviewer attack report: no red flags
- current yellow flag: hosted anonymous dataset URL pending
