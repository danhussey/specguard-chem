# Abstract Variants

## A. Conservative

SpecGuard-Chem is an oracle-first compiler and evaluation harness for agentic language models under medicinal-chemistry specifications. The sgchem_v1.0 release contains 120 bundles, 688 tasks, and 244 test tasks compiled from machine-checkable witnesses and certificates. It evaluates construction, repair, candidate audit, rejection, abstention, verifier-tool use, boundary precision, representation invariance, and interrupt/resume behavior while isolating hidden oracle fields through PublicTaskView and deterministic validation.

The benchmark uses an ACCEPT/REJECT/ABSTAIN action space. A metric sanity pass shows why molecule acceptance is insufficient: three deterministic baselines reach molecule_acceptance_rate=0.852, but their overall_task_success and action_accuracy are 0.664, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and unsafe_accept_rate=0.561. SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.

## B. Balanced

We introduce SpecGuard-Chem, an oracle-first compiler and benchmark harness for evaluating agentic language models under medicinal-chemistry specifications. sgchem_v1.0 compiles 120 bundles and 688 tasks from machine-checkable witnesses and certificates, including 244 held-out test tasks covering construction, repair, candidate audit, rejection, abstention, verifier-tool use, boundary precision, representation invariance, and interrupt/resume behavior. The artifact includes bundle-aware splits, public/private task isolation, strict validation, prompt-leakage audits, oracle-scrambling audits, and negative controls.

Our results show that aggregate molecule acceptance can be misleading. Deterministic construction/retrieval baselines reach molecule_acceptance_rate=0.852, but task/action accuracy remains 0.664 with REJECT_recall=0.000, ABSTAIN_recall=0.000, and unsafe_accept_rate=0.561. The benchmark therefore emphasizes action-aware evaluation, unsafe acceptance, rejection, abstention, and protocol diagnostics. SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.

## C. Ambitious But Safe

SpecGuard-Chem reframes medicinal-chemistry language-model evaluation as an oracle-first benchmark compilation problem. Rather than relying on template-filled prompts or hidden human labels, sgchem_v1.0 compiles 120 bundles and 688 tasks from machine-checkable witnesses and certificates, then evaluates agents through sanitized public task views. The 244-task test split covers specification construction, repair, candidate audit, rejection, abstention, verifier-tool use, boundary precision, representation invariance, and interrupt/resume behavior.

The release is accompanied by deterministic validation, 18 negative controls, prompt-leakage and oracle-scrambling audits, clean-clone reproduction, Croissant metadata, and a claim ledger tying paper statements to evidence. Empirically, molecule_acceptance_rate=0.852 for several deterministic baselines masks action failures: overall_task_success is 0.664, REJECT_recall=0.000, ABSTAIN_recall=0.000, and unsafe_accept_rate=0.561. This supports action-aware reporting rather than acceptance-only scoring. SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.
