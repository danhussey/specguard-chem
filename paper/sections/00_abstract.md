# Abstract

SpecGuard-Chem is an oracle-first compiler and evaluation harness for agentic language models under medicinal-chemistry specifications. The sgchem_v1.0 release contains 118 bundles, 656 tasks, and 266 test tasks. Each task is compiled from machine-checkable witnesses or certificates and evaluated through public task views that isolate hidden oracle fields from model prompts.

The benchmark covers construction, repair, candidate audit, rejection, abstention, verifier-tool use, boundary precision, representation invariance, and interrupt/resume behavior. Its action space contains ACCEPT=418, REJECT=138, and ABSTAIN=100 tasks in the full release, with test-set counts ACCEPT=179, REJECT=52, and ABSTAIN=35.

The central empirical result is that molecule acceptance alone is misleading: deterministic local and verify-first baselines reach molecule_acceptance_rate=0.846, while the retrieval upper bound reaches 0.868. Their action accuracy remains 0.650, 0.650, and 0.673 respectively, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and unsafe_accept_rate=0.598. We therefore report action-aware metrics, unsafe acceptance, reject/abstain recall, and protocol diagnostics rather than raw acceptance alone.

SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.
