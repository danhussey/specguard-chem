# Challenge Slice Results

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
