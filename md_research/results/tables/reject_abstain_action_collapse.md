# Reject And Abstain Action Collapse

| system | provider | action_accuracy | molecule_acceptance_rate | reject_recall | abstain_recall | task_inconsistent_accept_rate |
| --- | --- | --- | --- | --- | --- | --- |
| always_accept | deterministic | 0.650 | 0.738 | 0.111 | 0.000 | 0.471 |
| always_abstain | deterministic | 0.100 | 0.000 | 0.000 | 1.000 | 0.000 |
| heuristic | deterministic | 0.487 | 0.375 | 1.000 | 0.000 | 0.000 |
| abstention_guard | deterministic | 0.613 | 0.650 | 0.333 | 0.000 | 0.353 |
| local_mutation | deterministic | 0.738 | 0.850 | 0.000 | 0.000 | 0.529 |
| verify_first | deterministic | 0.738 | 0.850 | 0.000 | 0.000 | 0.529 |
| corpus_search | deterministic | 0.787 | 0.900 | 0.000 | 0.000 | 0.529 |
| well_engineered_wrapper | deterministic | 1.000 | 0.787 | 1.000 | 1.000 | 0.000 |
| openai_fast_closed | openai | 0.100 | 0.000 | 0.000 | 1.000 | 0.000 |
| openai_fast_verify_l3 | openai | 0.113 | 0.013 | 0.000 | 1.000 | 0.000 |
| openai_strong_closed | openai | 0.900 | 0.688 | 1.000 | 1.000 | 0.000 |
| openai_strong_verify_l3 | openai | 0.675 | 0.463 | 1.000 | 1.000 | 0.000 |
| anthropic_fast_closed | anthropic | 0.100 | 0.000 | 0.000 | 1.000 | 0.000 |
| anthropic_fast_verify_l3 | anthropic | 0.100 | 0.000 | 0.000 | 1.000 | 0.000 |
| anthropic_strong_closed | anthropic | 0.775 | 0.562 | 1.000 | 1.000 | 0.000 |
| anthropic_strong_verify_l3 | anthropic | 0.588 | 0.375 | 1.000 | 1.000 | 0.000 |
| deepseek_fast_closed | deepseek | 0.675 | 0.475 | 0.889 | 1.000 | 0.000 |
| deepseek_fast_verify_l3 | deepseek | 0.738 | 0.537 | 0.889 | 1.000 | 0.000 |
| deepseek_strong_closed | deepseek | 0.175 | 0.062 | 0.111 | 1.000 | 0.000 |
| deepseek_strong_verify_l3 | deepseek | 0.175 | 0.075 | 0.000 | 1.000 | 0.000 |
