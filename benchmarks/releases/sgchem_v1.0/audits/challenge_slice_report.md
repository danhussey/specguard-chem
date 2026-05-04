# Challenge Slice Report

Challenge membership is structural. It is assigned from task type, oracle evidence, protocol, and constraint metadata before any baseline output is read.

- challenge test tasks: 196 / 244
- classification: primary_reportable
- expected_action counts: {'ABSTAIN': 36, 'ACCEPT': 114, 'REJECT': 46}
- task_type counts: {'abstain_contradiction': 36, 'audit_accept': 13, 'audit_reject': 36, 'boundary_precision': 20, 'construct_feasible': 11, 'interrupt_resume': 10, 'repair_multi_violation': 20, 'repair_near_miss': 20, 'smiles_invariance': 20, 'tool_forced_l3': 10}
- difficulty_tag counts: {'abstain_explicit_contradiction': 36, 'interrupt_state_required': 10, 'invariance_equivalent_representation': 20, 'minimal_edit_required': 20, 'mixed_hard_soft_tradeoff': 196, 'multi_constraint_violation': 20, 'reject_near_miss': 36, 'tight_property_boundary': 20, 'tool_required_by_protocol': 76}

## Results

| baseline | track | n | overall_task_success | action_accuracy | molecule_acceptance_rate | unsafe_accept_rate | REJECT_recall | ABSTAIN_recall | hard_violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_accept | primary_closed_book | 196 | 0.520 | 0.520 | 0.684 | 0.476 | 0.152 | 0.000 | 0.316 |
| always_reject | primary_closed_book | 196 | 0.235 | 0.235 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| always_abstain | primary_closed_book | 196 | 0.184 | 0.184 | 0.000 | 0.000 | 0.000 | 1.000 | NA |
| random_action | primary_closed_book | 196 | 0.286 | 0.286 | 0.087 | 0.000 | 0.609 | 0.306 | 0.876 |
| schema_valid_dummy | primary_closed_book | 196 | 0.520 | 0.520 | 0.684 | 0.476 | 0.152 | 0.000 | 0.316 |
| heuristic | primary_closed_book | 196 | 0.500 | 0.500 | 0.265 | 0.000 | 1.000 | 0.000 | 0.720 |
| abstention_guard | primary_closed_book | 196 | 0.592 | 0.592 | 0.663 | 0.366 | 0.348 | 0.000 | 0.301 |
| local_mutation_or_repair | primary_closed_book | 196 | 0.582 | 0.582 | 0.816 | 0.561 | 0.000 | 0.000 | 0.184 |
| verify_first | tool_enabled | 196 | 0.582 | 0.582 | 0.816 | 0.561 | 0.000 | 0.000 | 0.184 |
| verifier_guided_greedy | tool_enabled | 196 | 0.602 | 0.602 | 0.663 | 0.354 | 0.370 | 0.000 | 0.337 |
| corpus_retrieval_upper_bound | retrieval_upper_bound | 196 | 0.582 | 0.582 | 0.816 | 0.561 | 0.000 | 0.000 | 0.184 |
