# Challenge Slice Report

Challenge membership is structural. It is assigned from task type, oracle evidence, protocol, and constraint metadata before any baseline output is read.

- challenge test tasks: 210 / 266
- classification: primary_reportable
- expected_action counts: {'ABSTAIN': 35, 'ACCEPT': 123, 'REJECT': 52}
- task_type counts: {'abstain_contradiction': 35, 'audit_accept': 13, 'audit_reject': 41, 'boundary_precision': 22, 'construct_feasible': 14, 'interrupt_resume': 11, 'repair_multi_violation': 19, 'repair_near_miss': 22, 'smiles_invariance': 20, 'tool_forced_l3': 13}
- difficulty_tag counts: {'abstain_explicit_contradiction': 35, 'interrupt_state_required': 11, 'invariance_equivalent_representation': 20, 'minimal_edit_required': 22, 'mixed_hard_soft_tradeoff': 210, 'multi_constraint_violation': 19, 'reject_near_miss': 41, 'tight_property_boundary': 22, 'tool_required_by_protocol': 84}

## Results

| baseline | track | n | overall_task_success | action_accuracy | molecule_acceptance_rate | unsafe_accept_rate | REJECT_recall | ABSTAIN_recall | hard_violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_accept | primary_closed_book | 210 | 0.486 | 0.486 | 0.667 | 0.517 | 0.135 | 0.000 | 0.333 |
| always_reject | primary_closed_book | 210 | 0.248 | 0.248 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| always_abstain | primary_closed_book | 210 | 0.167 | 0.167 | 0.000 | 0.000 | 0.000 | 1.000 | NA |
| random_action | primary_closed_book | 210 | 0.310 | 0.310 | 0.110 | 0.000 | 0.673 | 0.200 | 0.848 |
| schema_valid_dummy | primary_closed_book | 210 | 0.486 | 0.486 | 0.667 | 0.517 | 0.135 | 0.000 | 0.333 |
| heuristic | primary_closed_book | 210 | 0.514 | 0.514 | 0.267 | 0.000 | 1.000 | 0.000 | 0.719 |
| abstention_guard | primary_closed_book | 210 | 0.529 | 0.529 | 0.614 | 0.402 | 0.327 | 0.000 | 0.352 |
| local_mutation_or_repair | primary_closed_book | 210 | 0.557 | 0.557 | 0.805 | 0.598 | 0.000 | 0.000 | 0.195 |
| verify_first | tool_enabled | 210 | 0.557 | 0.557 | 0.805 | 0.598 | 0.000 | 0.000 | 0.195 |
| verifier_guided_greedy | tool_enabled | 210 | 0.562 | 0.562 | 0.638 | 0.391 | 0.346 | 0.000 | 0.362 |
| corpus_retrieval_upper_bound | retrieval_upper_bound | 210 | 0.586 | 0.586 | 0.833 | 0.598 | 0.000 | 0.000 | 0.167 |
