# Baselines

Baselines are reported in separated tracks.

The primary_closed_book track contains deterministic baselines that consume PublicTaskView without hidden oracle fields. Tool-enabled baselines are reported separately because they use verifier access. Molecule-retrieval baselines are not primary leaderboard entries because they reflect access to corpus search rather than closed-book behavior. The wrapper_guarded track is a reality-check ceiling: it asks what happens when a system is engineered directly around the public specification, verifier semantics, and public candidate search.

The baseline set includes always_accept, always_reject, always_abstain, random_action, schema_valid_dummy, heuristic, abstention_guard, local_mutation_or_repair, verify_first, verifier_guided_greedy, and corpus_retrieval_upper_bound.

The reality-check pass also includes `well_engineered_wrapper`. This baseline is not a model leaderboard entry. It is a saturation test for whether the public task contract is formally solvable by deterministic verifier engineering.

The paper should not mix primary_closed_book, tool_enabled, retrieval_upper_bound, external_model_snapshot, and wrapper_guarded rows without track labels. The table `paper_v1/tables/baseline_tracks.md` is the track source of truth.

## Threat Model

| system class | sees hidden certificates? | sees expected action? | uses public verifier? | uses corpus retrieval? | intended interpretation |
| --- | --- | --- | --- | --- | --- |
| closed-book baselines | no | no | no | no | model/policy behavior under PublicTaskView |
| tool-enabled baselines | no | no | yes, when protocol permits | no | verifier-mediated behavior |
| molecule-corpus retrieval baseline | no | no | indirectly through scoring/search | yes | molecule acceptance ceiling from available corpus, not task-action ceiling |
| engineered verifier/search wrapper | no | no | yes, public semantics only | yes, public corpus/local search | saturation under engineered public verification and search |
| oracle/debug checks | yes | yes | yes | yes | validation only; not reportable as agent performance |
