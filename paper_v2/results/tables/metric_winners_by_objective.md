# Metric Winners by Objective

| objective_metric | rank_1_system | rank_1_access_model | rank_1_value | rank_2_system | rank_2_value | hidden_failure_mode | paper_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_accuracy | well_engineered_wrapper | verifier/search wrapper | 1.000 | corpus_search | 0.673 | depends on access model and action distribution | primary decision-contract metric |
| balanced_action_accuracy | well_engineered_wrapper | verifier/search wrapper | 1.000 | heuristic | 0.534 | depends on access model and action distribution | use as a diagnostic slice with denominator |
| molecule_acceptance_rate | corpus_search | retrieval | 0.868 | local_mutation | 0.846 | can reward accept-biased systems on reject/abstain tasks | not a task-success headline metric |
| task_inconsistent_accept_rate | always_reject | closed-book | 0.000 | always_abstain | 0.000 | exposes reject/abstain collapse into accept decisions | use as a diagnostic slice with denominator |
| reject_recall | always_reject | closed-book | 1.000 | heuristic | 1.000 | isolates one action class rather than aggregate task success | use as a diagnostic slice with denominator |
| abstain_recall | always_abstain | closed-book | 1.000 | well_engineered_wrapper | 1.000 | isolates one action class rather than aggregate task success | use as a diagnostic slice with denominator |
| hard_violation_rate | corpus_search | retrieval | 0.132 | local_mutation | 0.154 | depends on access model and action distribution | use as a diagnostic slice with denominator |
| schema_error_rate | always_accept | closed-book | 0.000 | always_reject | 0.000 | depends on access model and action distribution | use as a diagnostic slice with denominator |
| mean_verify_calls | always_accept | closed-book | 0 | always_reject | 0 | ignores accuracy unless thresholded | use as a diagnostic slice with denominator |
| balanced_action_score | well_engineered_wrapper | verifier/search wrapper | 1.000 | heuristic | 0.534 | depends on access model and action distribution | use as a diagnostic slice with denominator |
| safe_action_score | well_engineered_wrapper | verifier/search wrapper | 0.775 | always_abstain | 0.132 | depends on access model and action distribution | penalizes unsafe accepts, hard violations, and schema failures |
| cost_adjusted_action_score_lambda_0_01 | well_engineered_wrapper | verifier/search wrapper | 0.992 | corpus_search | 0.673 | depends on access model and action distribution | separates high-accuracy systems by verifier economy |
| cost_adjusted_action_score_lambda_0_05 | well_engineered_wrapper | verifier/search wrapper | 0.958 | corpus_search | 0.673 | depends on access model and action distribution | separates high-accuracy systems by verifier economy |
