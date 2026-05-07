| baseline | model | track | primary_leaderboard |
| --- | --- | --- | --- |
| always_accept | always_accept | primary_closed_book | true |
| always_reject | always_reject | primary_closed_book | true |
| always_abstain | always_abstain | primary_closed_book | true |
| random_action | random_action | primary_closed_book | true |
| schema_valid_dummy | schema_valid_dummy | primary_closed_book | true |
| heuristic | heuristic | primary_closed_book | true |
| abstention_guard | abstention_guard | primary_closed_book | true |
| local_mutation_or_repair | local_mutation | primary_closed_book | true |
| verify_first | verify_first | tool_enabled | false |
| verifier_guided_greedy | verifier_guided_greedy | tool_enabled | false |
| corpus_retrieval_upper_bound | corpus_search | retrieval_upper_bound | false |

Display note: `corpus_retrieval_upper_bound` should appear in prose and final paper tables as the molecule-corpus retrieval baseline. It is a molecule-retrieval ceiling from the available corpus, not an upper bound on action-correct task performance.
