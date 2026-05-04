# SpecGuard-Chem sgchem_v1.0 Paper Pitch

## Provisional Title

SpecGuard-Chem: Oracle-First Evaluation Under Medicinal-Chemistry Specifications

## Core Claim

SpecGuard-Chem is an oracle-first compiler and evaluation harness for agentic language models under medicinal-chemistry specifications. The sgchem_v1.0 release contains 120 bundles, 688 tasks, and 244 test tasks compiled from machine-checkable witnesses and certificates.

## Why It Matters

The benchmark shows that molecule acceptance alone is misleading. Several deterministic baselines reach molecule_acceptance_rate=0.852, but their overall_task_success and action_accuracy are 0.664, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and unsafe_accept_rate=0.561. The paper should therefore emphasize action-aware evaluation, unsafe acceptance, rejection, abstention, and protocol diagnostics.

## Scope

SpecGuard-Chem uses molecular property ranges, substructure constraints, alert filters, similarity guards, and representation-equivalence checks. It evaluates specification compliance, construction, repair, candidate audit, rejection, abstention, verifier-tool use, boundary precision, representation invariance, and interrupt/resume behavior.

SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.

## Artifact Status

- strict validation: valid=true, num_errors=0
- prompt leakage: zero leaks across 688 prompts
- oracle scrambling: passed
- negative controls: 18 corruptions covered
- clean-clone reproduction: passed from anonymous archive
- manual test-bundle review: A=36, B=0, C=0, D=0
- reviewer attack report: no red flags
- current yellow flag: hosted anonymous dataset URL pending
