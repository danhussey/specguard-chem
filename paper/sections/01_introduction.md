# Introduction

Language-model agents are increasingly asked to follow structured scientific specifications, interact with tools, and make accept/reject/abstain decisions. A defensible evaluation needs to separate visible instructions from hidden answers, prevent leakage across related examples, and report metrics that distinguish successful construction from unsafe acceptance or failure to abstain.

SpecGuard-Chem uses medicinal-chemistry specifications as a scoped domain for this evaluation problem. The specifications use molecular property ranges, substructure constraints, alert filters, similarity guards, and representation-equivalence checks. This domain is useful because it supports deterministic local verification while still exercising realistic specification-following behavior.

The benchmark is not a molecule-discovery claim. SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success. The scientific object of study is evaluation design: oracle-backed task compilation, public/private task isolation, validation gates, and action-aware reporting.

The sgchem_v1.0 release has 118 bundles, 656 tasks, and 266 test tasks. The paper-facing claim ledger is `paper_v1/claim_ledger.yaml`, and every empirical claim is tied to a denominator table or audit report.
