# Reject And Abstain Metrics

| baseline | track | REJECT_precision | REJECT_precision_n | REJECT_recall | REJECT_recall_n | ABSTAIN_precision | ABSTAIN_precision_n | ABSTAIN_recall | ABSTAIN_recall_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_accept | primary_closed_book | 0.090 | 78 | 0.135 | 52 | NA | 0 | 0.000 | 35 |
| always_reject | primary_closed_book | 0.195 | 266 | 1.000 | 52 | NA | 0 | 0.000 | 35 |
| always_abstain | primary_closed_book | NA | 0 | 0.000 | 52 | 0.132 | 266 | 1.000 | 35 |
| random_action | primary_closed_book | 0.230 | 152 | 0.673 | 52 | 0.093 | 75 | 0.200 | 35 |
| schema_valid_dummy | primary_closed_book | 0.090 | 78 | 0.135 | 52 | NA | 0 | 0.000 | 35 |
| heuristic | primary_closed_book | 0.354 | 147 | 1.000 | 52 | 0.000 | 11 | 0.000 | 35 |
| abstention_guard | primary_closed_book | 0.230 | 74 | 0.327 | 52 | 0.000 | 11 | 0.000 | 35 |
| local_mutation_or_repair | primary_closed_book | 0.000 | 41 | 0.000 | 52 | NA | 0 | 0.000 | 35 |
| verify_first | tool_enabled | 0.000 | 41 | 0.000 | 52 | NA | 0 | 0.000 | 35 |
| verifier_guided_greedy | tool_enabled | 0.225 | 80 | 0.346 | 52 | NA | 0 | 0.000 | 35 |
| corpus_retrieval_upper_bound | retrieval_upper_bound | 0.000 | 35 | 0.000 | 52 | NA | 0 | 0.000 | 35 |
