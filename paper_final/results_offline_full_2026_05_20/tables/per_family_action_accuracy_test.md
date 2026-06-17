# Per-family Action Accuracy

| adapter | abstain_contradiction | audit_accept | audit_reject | boundary_precision | construct_feasible | interrupt_resume | repair_multi_violation | repair_near_miss | smiles_invariance | tool_forced_l3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_accept | 0.000 | 0.857 | 0.146 | 0.500 | 0.854 | 0.818 | 0.316 | 0.818 | 0.900 | 0.846 |
| always_abstain | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heuristic | 0.000 | 1.000 | 1.000 | 1.000 | 0.854 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| abstention_guard | 0.000 | 1.000 | 0.390 | 0.545 | 0.854 | 0.000 | 0.316 | 0.909 | 1.000 | 0.923 |
| local_mutation | 0.000 | 1.000 | 0.000 | 0.500 | 0.976 | 1.000 | 0.737 | 1.000 | 1.000 | 1.000 |
| verify_first | 0.000 | 1.000 | 0.000 | 0.500 | 0.976 | 1.000 | 0.737 | 1.000 | 1.000 | 1.000 |
| corpus_search | 0.000 | 1.000 | 0.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| well_engineered_wrapper | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
