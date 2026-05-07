# Results

The main empirical finding is not that sgchem_v1.0 is unsolved by every deterministic baseline. Instead, it is that aggregate molecule acceptance hides action-specific failures.

On the 266-task test split, local_mutation_or_repair and verify_first reach molecule_acceptance_rate=0.846, while the molecule-corpus retrieval baseline reaches 0.868. However, their action accuracies are 0.650, 0.650, and 0.673, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and task_inconsistent_accept_rate=0.598. This shows that a method can often produce or retrieve a hard-passing molecule while still failing the action semantics of candidate audit, rejection, and abstention.

The baseline named `corpus_retrieval_upper_bound` is an upper bound for finding acceptable molecules from the available corpus, not an upper bound on action-correct task performance. We refer to it in prose as the molecule-corpus retrieval baseline to avoid confusing molecule acceptance with task success.

## Verifier-wrapper Saturation

The deterministic `well_engineered_wrapper` saturates the full 266-task test split under the public verifier/search-wrapper threat model: action_accuracy=1.000, REJECT_recall=1.000, ABSTAIN_recall=1.000, and task_inconsistent_accept_rate=0.000. This result is expected because sgchem_v1.0 is intentionally machine-checkable. It establishes that the artifact should not be interpreted as a hard chemistry leaderboard for systems allowed to engineer directly against the public specification contract.

The intended use is narrower and more defensible: measuring whether language-agent interfaces, output schemas, tool-use policies, rejection behavior, abstention behavior, and audit boundaries preserve oracle-certified specification compliance. The wrapper result is therefore a main result, not a caveat to hide.

## Diagnostic Model Behavior

Cached OpenAI snapshot runs in `paper_v1/reality_check_decision_memo.md` show action-policy failure modes under the current adapter and prompt path. These cached frontier-model snapshots are diagnostic examples, not the main scientific claim.

The challenge slice is structurally defined from task/spec/oracle metadata and contains 210 of 266 test tasks. Challenge membership does not use baseline success or failure. The slice is large enough for aggregate reporting, but its subfamilies retain their own denominators.

Boundary precision, representation invariance, and interrupt/resume are diagnostic slices in this release: boundary_precision has 22 test tasks, smiles_invariance has 20 test tasks, and interrupt_resume has 11 test tasks. `repair_multi_violation` has 19 test tasks and is diagnostic-only in paper claims.

## Supported Claims

| claim type | supported? | why |
| --- | --- | --- |
| Agent follows machine-checkable specifications | yes | public task view plus hidden oracle certificates |
| Agent distinguishes ACCEPT, REJECT, ABSTAIN | yes | action-aware labels, metrics, and confusion matrices |
| Agent can use verifier tools correctly | yes, in L3/tool-forced tasks | explicit tool budgets and tool-compliance traces |
| Agent is good at medicinal chemistry | no | no activity, toxicity, synthesis, or therapeutic validation |
| Agent can design drugs | no | explicitly out of scope |
| Benchmark is a hard chemistry leaderboard for engineered systems | no | verifier/search-wrapper saturation result |
| Benchmark is useful for audit methodology | yes | leakage audits, oracle scrambling, negative controls, clean-clone reproduction, and denominator reports |

Generated result tables:

- `paper_v1/tables/primary_results.md`
- `paper_v1/tables/baseline_metric_sanity.md`
- `paper_v1/tables/baseline_action_confusion_matrices.md`
- `paper_v1/tables/wrapper_saturation.md`
- `paper_v1/tables/challenge_slice_results.md`
- `paper_v1/tables/diagnostic_slice_results.md`
