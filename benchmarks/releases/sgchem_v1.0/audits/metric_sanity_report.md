# Metric Sanity Report

The internal `accept_rate` reported by earlier sweep summaries is renamed in paper-facing tables to `molecule_acceptance_rate`.
It means the fraction of tasks whose final decision was ACCEPT, not the fraction of tasks answered correctly.

The three deterministic baselines `local_mutation_or_repair`, `verify_first`, and `corpus_retrieval_upper_bound` have the same 0.852 molecule_acceptance_rate because all three deterministically search for or retrieve a hard-passing molecule on nearly the same set of visible specification instances. That agreement is benign for construction/repair coverage, but it is not a headline success metric: these baselines differ on action accuracy, unsafe acceptance, rejection, abstention, tool use, edit economy, and calibration.

Paper implication: do not claim non-saturation from molecule acceptance alone. The defensible claim is that metric decomposition reveals different failure modes that aggregate acceptance hides.

## Release Counts

- full expected_action counts: {'ABSTAIN': 120, 'ACCEPT': 426, 'REJECT': 142}
- test expected_action counts: {'ABSTAIN': 36, 'ACCEPT': 162, 'REJECT': 46}
- full task_type counts: {'abstain_contradiction': 120, 'audit_accept': 120, 'audit_reject': 120, 'boundary_precision': 44, 'construct_feasible': 120, 'interrupt_resume': 27, 'repair_multi_violation': 30, 'repair_near_miss': 36, 'smiles_invariance': 44, 'tool_forced_l3': 27}
- test task_type counts: {'abstain_contradiction': 36, 'audit_accept': 36, 'audit_reject': 36, 'boundary_precision': 20, 'construct_feasible': 36, 'interrupt_resume': 10, 'repair_multi_violation': 20, 'repair_near_miss': 20, 'smiles_invariance': 20, 'tool_forced_l3': 10}

## 0.852 Baselines

| baseline | track | num_tasks | overall_task_success | action_accuracy | molecule_acceptance_rate | ACCEPT_precision | ACCEPT_precision_n | ACCEPT_recall | ACCEPT_recall_n | REJECT_precision | REJECT_precision_n | REJECT_recall | REJECT_recall_n | ABSTAIN_precision | ABSTAIN_precision_n | ABSTAIN_recall | ABSTAIN_recall_n | unsafe_accept_rate | unsafe_accept_n | false_abstain_rate | false_abstain_n | hard_violation_rate | hard_violation_n | schema_error_rate | schema_error_n | invalid_molecule_rate | invalid_molecule_n | pass_at_1 | pass_at_1_n | pass_at_3 | pass_at_3_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_mutation_or_repair | primary_closed_book | 244 | 0.664 | 0.664 | 0.852 | 0.779 | 208 | 1.000 | 162 | 0.000 | 36 | 0.000 | 46 | NA | 0 | 0.000 | 36 | 0.561 | 82 | 0.000 | 208 | 0.148 | 244 | 0.000 | 244 | 0.000 | 244 | 1.000 | 162 | 1.000 | 162 |
| verify_first | tool_enabled | 244 | 0.664 | 0.664 | 0.852 | 0.779 | 208 | 1.000 | 162 | 0.000 | 36 | 0.000 | 46 | NA | 0 | 0.000 | 36 | 0.561 | 82 | 0.000 | 208 | 0.148 | 244 | 0.000 | 320 | 0.000 | 244 | 0.605 | 162 | 1.000 | 162 |
| corpus_retrieval_upper_bound | retrieval_upper_bound | 244 | 0.664 | 0.664 | 0.852 | 0.779 | 208 | 1.000 | 162 | 0.000 | 36 | 0.000 | 46 | NA | 0 | 0.000 | 36 | 0.561 | 82 | 0.000 | 208 | 0.148 | 244 | 0.000 | 244 | 0.000 | 244 | 1.000 | 162 | 1.000 | 162 |


## Saturation Interpretation

Deterministic local/retrieval baselines partly saturate molecule construction on some feasible slices. Those slices should be framed as sanity-check slices. Rejection, abstention, boundary, invariance, interrupt/resume, calibration, and tool-economy slices remain interpretation-critical and must be reported with denominators.
