# Reject And Abstain Metrics

| baseline | track | REJECT_precision | REJECT_precision_n | REJECT_recall | REJECT_recall_n | ABSTAIN_precision | ABSTAIN_precision_n | ABSTAIN_recall | ABSTAIN_recall_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_accept | primary_closed_book | 0.101 | 69 | 0.152 | 46 | NA | 0 | 0.000 | 36 |
| always_reject | primary_closed_book | 0.189 | 244 | 1.000 | 46 | NA | 0 | 0.000 | 36 |
| always_abstain | primary_closed_book | NA | 0 | 0.000 | 46 | 0.148 | 244 | 1.000 | 36 |
| random_action | primary_closed_book | 0.194 | 144 | 0.609 | 46 | 0.155 | 71 | 0.306 | 36 |
| schema_valid_dummy | primary_closed_book | 0.101 | 69 | 0.152 | 46 | NA | 0 | 0.000 | 36 |
| heuristic | primary_closed_book | 0.333 | 138 | 1.000 | 46 | 0.000 | 10 | 0.000 | 36 |
| abstention_guard | primary_closed_book | 0.271 | 59 | 0.348 | 46 | 0.000 | 10 | 0.000 | 36 |
| local_mutation_or_repair | primary_closed_book | 0.000 | 36 | 0.000 | 46 | NA | 0 | 0.000 | 36 |
| verify_first | tool_enabled | 0.000 | 36 | 0.000 | 46 | NA | 0 | 0.000 | 36 |
| verifier_guided_greedy | tool_enabled | 0.243 | 70 | 0.370 | 46 | NA | 0 | 0.000 | 36 |
| corpus_retrieval_upper_bound | retrieval_upper_bound | 0.000 | 36 | 0.000 | 46 | NA | 0 | 0.000 | 36 |
