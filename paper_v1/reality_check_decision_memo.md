# Reality Check Decision Memo

## Question

Does sgchem_v1.0 support a compelling NeurIPS E&D paper, or is it mostly an over-engineered deterministic rule puzzle?

## Experiment Status

- subset policy: all REJECT test tasks + all ABSTAIN test tasks + stratified ACCEPT tasks
- subset size: 127
- full-test size run: 266
- external model status: External runs requested with --allow-external; see per-run directories/cache for successful runs and *_ERROR.txt files for failures.

## Results

| run | n | action_accuracy | molecule_acceptance_rate | task_inconsistent_accept_rate | REJECT_recall | ABSTAIN_recall | accepted_same_candidate_after_failed_verifier | substituted_passing_molecule_for_audit | accepted_on_contradictory_spec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| well_engineered_wrapper_subset | 127 | 1.000 | 0.315 | 0.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| well_engineered_wrapper_full | 266 | 1.000 | 0.673 | 0.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| openai_cheap_no_tool_subset | 127 | 0.276 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| openai_cheap_tool_available_subset | 127 | 0.276 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| openai_cheap_forced_verify_first_subset | 127 | 0.276 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| openai_frontier_no_tool_subset | 30 | 0.333 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| openai_frontier_tool_available_subset | 30 | 0.333 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| openai_frontier_forced_verify_first_subset | 30 | 0.333 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |

## Answers

- Do actual LLM agents fail in interesting ways? The successful OpenAI runs failed in a simple but paper-relevant way: they collapsed to ABSTAIN across the stratified subset, producing zero molecule acceptance, zero REJECT recall, and perfect ABSTAIN recall. That is an action-policy failure rather than evidence of robust constructive specification compliance.
- Are failures mostly trivial prompt/schema issues? Mostly yes for the current OpenAI snapshot: the dominant failure is blanket abstention/action policy, not hidden-oracle leakage or verifier misuse.
- Does a deterministic wrapper solve the task? It is the strongest current baseline; see table above.
- Is the benchmark measuring LLM capability or wrapper engineering? Current evidence strongly measures wrapper engineering and harness correctness. Actual LLM runs, when available, should be treated as adapter/protocol sanity checks unless richer failure modes appear.
- Is the main result worth submitting? Recommendation: reframe.
- Should we reframe, revise, or pivot? Reframe as an artifact/compiler and evaluation harness; do not claim raw-agent nontriviality until live model data supports it.

## Recommendation

REFRAME: The deterministic wrapper has high action accuracy and perfect REJECT/ABSTAIN recall when it uses the public verifier/spec interface. This suggests the artifact is best framed as an oracle-first compiler and harness for testing public/private isolation, task semantics, and wrapper-vs-agent behavior, not as standalone evidence that raw LLM agents need this benchmark. Actual LLM runs were executed; in the current snapshot the observed failure mode is dominated by action-policy behavior rather than rich molecular search.

## Wrapper Baseline Interpretation

- action_accuracy: 1.0
- molecule_acceptance_rate: 0.6729323308270677
- task_inconsistent_accept_rate: 0.0
- REJECT_recall: 1.0
- ABSTAIN_recall: 1.0
