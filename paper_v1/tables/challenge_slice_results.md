# Challenge Slice Results

| baseline | track | n | overall_task_success | action_accuracy | molecule_acceptance_rate | task_inconsistent_accept_rate | REJECT_recall | ABSTAIN_recall | hard_violation_rate |
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
