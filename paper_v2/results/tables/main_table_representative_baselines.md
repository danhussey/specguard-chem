| system | access_model | action_accuracy | molecule_acceptance_rate | task_inconsistent_accept_rate | reject_recall | abstain_recall | schema_error_rate | mean_verify_calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_accept | closed-book | 0.564 | 0.707 | 0.517 | 0.135 | 0.000 | 0.000 | 0 |
| always_abstain | closed-book | 0.132 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| heuristic | closed-book | 0.602 | 0.406 | 0.000 | 1.000 | 0.000 | 0.000 | 0 |
| abstention_guard | closed-book | 0.613 | 0.680 | 0.402 | 0.327 | 0.000 | 0.000 | 0 |
| local_mutation | closed-book | 0.650 | 0.846 | 0.598 | 0.000 | 0.000 | 0.000 | 0 |
| verify_first | public-verifier | 0.650 | 0.846 | 0.598 | 0.000 | 0.000 | 0.000 | 0.316 |
| corpus_search | retrieval | 0.673 | 0.868 | 0.598 | 0.000 | 0.000 | 0.000 | 0 |
| well_engineered_wrapper | verifier/search wrapper | 1.000 | 0.673 | 0.000 | 1.000 | 1.000 | 0.000 | 0.263 |
