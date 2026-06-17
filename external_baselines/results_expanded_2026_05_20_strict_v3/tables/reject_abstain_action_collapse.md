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
| openai_strong_strict-tool-call_closed | openai | 0.838 | 0.625 | 1.000 | 1.000 | 0.000 |
| openai_strong_strict-tool-call_verify_l3 | openai | 0.738 | 0.525 | 1.000 | 1.000 | 0.000 |
| anthropic_fast_strict-tool-call_closed | anthropic | 0.675 | 0.463 | 1.000 | 1.000 | 0.000 |
| anthropic_fast_strict-tool-call_verify_l3 | anthropic | 0.625 | 0.412 | 1.000 | 1.000 | 0.000 |
| anthropic_strong_strict-tool-call_closed | anthropic | 0.787 | 0.575 | 1.000 | 1.000 | 0.000 |
| anthropic_strong_strict-tool-call_verify_l3 | anthropic | 0.637 | 0.425 | 1.000 | 1.000 | 0.000 |
| deepseek_fast_strict-tool-call_closed | deepseek | 0.762 | 0.550 | 1.000 | 1.000 | 0.000 |
| deepseek_fast_strict-tool-call_verify_l3 | deepseek | 0.562 | 0.350 | 1.000 | 1.000 | 0.000 |
