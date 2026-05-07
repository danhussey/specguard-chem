| adapter | protocol | n_tasks | action_accuracy | balanced_action_accuracy | molecule_acceptance_rate | task_inconsistent_accept_rate | reject_recall | abstain_recall | hard_violation_rate | schema_error_rate | mean_steps | mean_proposals | mean_verify_calls | p95_verify_calls | budget_exhaustion_rate | not_applicable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | L1 | 97 | 0.619 | 0.653 | 0.485 | 0.000 | 1.000 | 0.000 | 0.515 | 0.000 | 1 | 1 | 0 | 0 | 0.515 | NA |
| heuristic | L2 | 85 | 0.718 | 0.800 | 0.424 | 0.000 | 1.000 | NA | 0.576 | 0.000 | 2.153 | 2.153 | 0 | 0 | 0.576 | NA |
| heuristic | L3 | 84 | 0.464 | 0.679 | 0.298 | 0.000 | 1.000 | NA | 0.658 | 0.000 | 2.560 | 2.274 | 0 | 0 | 0.571 | NA |
| abstention_guard | L1 | 97 | 0.608 | 0.646 | 0.474 | 0.000 | 1.000 | 0.000 | 0.526 | 0.000 | 1 | 1 | 0 | 0 | 0.526 | NA |
| abstention_guard | L2 | 85 | 0.706 | 0.535 | 0.929 | 0.880 | 0.120 | NA | 0.071 | 0.000 | 1.659 | 1.659 | 0 | 0 | 0.071 | NA |
| abstention_guard | L3 | 84 | 0.524 | 0.343 | 0.667 | 0.929 | 0.071 | NA | 0.233 | 0.000 | 1.988 | 1.845 | 0 | 0 | 0.202 | NA |
| local_mutation | L1 | 97 | 0.505 | 0.333 | 0.639 | 0.271 | 0.000 | 0.000 | 0.361 | 0.000 | 1 | 1 | 0 | 0 | 0.361 | NA |
| local_mutation | L2 | 85 | 0.706 | 0.500 | 1.000 | 1.000 | 0.000 | NA | 0.000 | 0.000 | 1 | 1 | 0 | 0 | 0.000 | NA |
| local_mutation | L3 | 84 | 0.762 | 0.457 | 0.929 | 1.000 | 0.000 | NA | 0.071 | 0.000 | 1.214 | 1.214 | 0 | 0 | 0.071 | NA |
| well_engineered_wrapper | L1 | 97 | 1.000 | 1.000 | 0.505 | 0.000 | 1.000 | 1.000 | 0.210 | 0.000 | 1 | 0.639 | 0 | 0 | 0.134 | NA |
| well_engineered_wrapper | L2 | 85 | 1.000 | 1.000 | 0.706 | 0.000 | 1.000 | NA | 0.294 | 0.000 | 1.588 | 1.588 | 0 | 0 | 0.294 | NA |
| well_engineered_wrapper | L3 | 84 | 1.000 | 1.000 | 0.833 | 0.000 | 1.000 | NA | 0.167 | 0.000 | 2.167 | 1.333 | 0.833 | 1 | 0.167 | NA |
| verify_first | L3 | 84 | 0.762 | 0.457 | 0.929 | 1.000 | 0.000 | NA | 0.071 | 0.000 | 2.143 | 1.143 | 1 | 1 | 0.071 | NA |
| verifier_guided_greedy | L3 | 84 | 0.631 | 0.436 | 0.750 | 0.857 | 0.143 | NA | 0.250 | 0.000 | 2.452 | 1.452 | 1 | 1 | 0.250 | NA |
| verify_first | L1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | adapter is verifier-oriented; benchmark entry restricted to L3 |
| verify_first | L2 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | adapter is verifier-oriented; benchmark entry restricted to L3 |
| verifier_guided_greedy | L1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | adapter is verifier-oriented; benchmark entry restricted to L3 |
| verifier_guided_greedy | L2 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | adapter is verifier-oriented; benchmark entry restricted to L3 |
