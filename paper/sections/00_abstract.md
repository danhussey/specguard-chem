# Abstract

SpecGuard-Chem is an oracle-first compiler and evaluation harness for agentic language models under medicinal-chemistry specifications. The sgchem_v1.0 release contains 120 bundles, 688 tasks, and 244 test tasks. Each task is compiled from machine-checkable witnesses or certificates and evaluated through public task views that isolate hidden oracle fields from model prompts.

The benchmark covers construction, repair, candidate audit, rejection, abstention, verifier-tool use, boundary precision, representation invariance, and interrupt/resume behavior. Its action space contains ACCEPT=426, REJECT=142, and ABSTAIN=120 tasks in the full release, with test-set counts ACCEPT=162, REJECT=46, and ABSTAIN=36.

The central empirical result is that molecule acceptance alone is misleading: three deterministic baselines reach molecule_acceptance_rate=0.852, but their overall_task_success and action_accuracy are 0.664, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and unsafe_accept_rate=0.561. We therefore report action-aware metrics, unsafe acceptance, reject/abstain recall, and protocol diagnostics rather than raw acceptance alone.

SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.
