# Reject And Abstain Action Collapse

| system | provider | action_accuracy | molecule_acceptance_rate | reject_recall | abstain_recall | task_inconsistent_accept_rate |
| --- | --- | --- | --- | --- | --- | --- |
| always_accept | deterministic | 0.600 | 0.600 | 0.500 | 0.000 | 0.333 |
| always_abstain | deterministic | 0.100 | 0.000 | 0.000 | 1.000 | 0.000 |
| heuristic | deterministic | 0.500 | 0.300 | 1.000 | 0.000 | 0.000 |
| abstention_guard | deterministic | 0.500 | 0.500 | 0.500 | 0.000 | 0.333 |
| local_mutation | deterministic | 0.600 | 0.800 | 0.000 | 0.000 | 0.667 |
| verify_first | deterministic | 0.600 | 0.800 | 0.000 | 0.000 | 0.667 |
| corpus_search | deterministic | 0.700 | 0.900 | 0.000 | 0.000 | 0.667 |
| well_engineered_wrapper | deterministic | 1.000 | 0.700 | 1.000 | 1.000 | 0.000 |
| openai_strong_strict-tool-call_closed | openai | 1.000 | 0.700 | 1.000 | 1.000 | 0.000 |
| openai_strong_strict-tool-call_verify_l3 | openai | 0.700 | 0.400 | 1.000 | 1.000 | 0.000 |
| anthropic_fast_strict-tool-call_closed | anthropic | 0.600 | 0.300 | 1.000 | 1.000 | 0.000 |
| anthropic_fast_strict-tool-call_verify_l3 | anthropic | 0.600 | 0.300 | 1.000 | 1.000 | 0.000 |
| anthropic_strong_strict-tool-call_closed | anthropic | 0.700 | 0.400 | 1.000 | 1.000 | 0.000 |
| anthropic_strong_strict-tool-call_verify_l3 | anthropic | 0.700 | 0.400 | 1.000 | 1.000 | 0.000 |
| deepseek_fast_strict-tool-call_closed | deepseek | 0.700 | 0.400 | 1.000 | 1.000 | 0.000 |
| deepseek_fast_strict-tool-call_verify_l3 | deepseek | 0.700 | 0.400 | 1.000 | 1.000 | 0.000 |
