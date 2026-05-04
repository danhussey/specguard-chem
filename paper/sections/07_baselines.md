# Baselines

Baselines are reported in separated tracks.

The primary_closed_book track contains deterministic baselines that consume PublicTaskView without hidden oracle fields. Tool-enabled baselines are reported separately because they use verifier access. Retrieval upper bounds are not primary leaderboard entries because they reflect access to corpus retrieval rather than closed-book behavior.

The baseline set includes always_accept, always_reject, always_abstain, random_action, schema_valid_dummy, heuristic, abstention_guard, local_mutation_or_repair, verify_first, verifier_guided_greedy, and corpus_retrieval_upper_bound.

The paper should not mix primary_closed_book, tool_enabled, and retrieval_upper_bound rows without track labels. The table `paper_v1/tables/baseline_tracks.md` is the track source of truth.
