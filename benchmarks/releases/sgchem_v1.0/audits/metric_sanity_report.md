# Metric Sanity Report

The internal `accept_rate` reported by earlier sweep summaries is renamed in paper-facing tables to `molecule_acceptance_rate`.
It means the fraction of tasks whose final decision was ACCEPT, not the fraction of tasks answered correctly.

The deterministic local, verify-first, and retrieval baselines can have high and sometimes similar molecule_acceptance_rate values because they search for or retrieve hard-passing molecules on many of the same visible specification instances. Those values are not headline task success: these baselines differ on action accuracy, unsafe acceptance, rejection, abstention, tool use, edit economy, and calibration.
Current tracked molecule_acceptance_rate values: {'corpus_retrieval_upper_bound': '0.868', 'local_mutation_or_repair': '0.846', 'verify_first': '0.846'}.

Paper implication: do not claim non-saturation from molecule acceptance alone. The defensible claim is that metric decomposition reveals different failure modes that aggregate acceptance hides.

## Release Counts

- full expected_action counts: {'ABSTAIN': 100, 'ACCEPT': 418, 'REJECT': 138}
- test expected_action counts: {'ABSTAIN': 35, 'ACCEPT': 179, 'REJECT': 52}
- full task_type counts: {'abstain_contradiction': 100, 'audit_accept': 118, 'audit_reject': 116, 'boundary_precision': 44, 'construct_feasible': 116, 'interrupt_resume': 27, 'repair_multi_violation': 29, 'repair_near_miss': 36, 'smiles_invariance': 44, 'tool_forced_l3': 26}
- test task_type counts: {'abstain_contradiction': 35, 'audit_accept': 42, 'audit_reject': 41, 'boundary_precision': 22, 'construct_feasible': 41, 'interrupt_resume': 11, 'repair_multi_violation': 19, 'repair_near_miss': 22, 'smiles_invariance': 20, 'tool_forced_l3': 13}

## Tracked Baselines

| baseline | track | num_tasks | overall_task_success | action_accuracy | molecule_acceptance_rate | ACCEPT_precision | ACCEPT_precision_n | ACCEPT_recall | ACCEPT_recall_n | REJECT_precision | REJECT_precision_n | REJECT_recall | REJECT_recall_n | ABSTAIN_precision | ABSTAIN_precision_n | ABSTAIN_recall | ABSTAIN_recall_n | unsafe_accept_rate | unsafe_accept_n | false_abstain_rate | false_abstain_n | hard_violation_rate | hard_violation_n | schema_error_rate | schema_error_n | invalid_molecule_rate | invalid_molecule_n | pass_at_1 | pass_at_1_n | pass_at_3 | pass_at_3_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_mutation_or_repair | primary_closed_book | 266 | 0.650 | 0.650 | 0.846 | 0.769 | 225 | 0.966 | 179 | 0.000 | 41 | 0.000 | 52 | NA | 0 | 0.000 | 35 | 0.598 | 87 | 0.000 | 231 | 0.154 | 266 | 0.000 | 284 | 0.000 | 284 | 0.966 | 179 | 0.966 | 179 |
| verify_first | tool_enabled | 266 | 0.650 | 0.650 | 0.846 | 0.769 | 225 | 0.966 | 179 | 0.000 | 41 | 0.000 | 52 | NA | 0 | 0.000 | 35 | 0.598 | 87 | 0.000 | 231 | 0.154 | 266 | 0.000 | 362 | 0.000 | 278 | 0.609 | 179 | 0.966 | 179 |
| corpus_retrieval_upper_bound | retrieval_upper_bound | 266 | 0.673 | 0.673 | 0.868 | 0.775 | 231 | 1.000 | 179 | 0.000 | 35 | 0.000 | 52 | NA | 0 | 0.000 | 35 | 0.598 | 87 | 0.000 | 231 | 0.132 | 266 | 0.000 | 266 | 0.000 | 266 | 1.000 | 179 | 1.000 | 179 |


## Saturation Interpretation

Deterministic local/retrieval baselines partly saturate molecule construction on some feasible slices. Those slices should be framed as sanity-check slices. Rejection, abstention, boundary, invariance, interrupt/resume, calibration, and tool-economy slices remain interpretation-critical and must be reported with denominators.
