# Results

The main empirical finding is not that sgchem_v1.0 is unsolved by every deterministic baseline. Instead, it is that aggregate molecule acceptance hides action-specific failures.

On the 266-task test split, local_mutation_or_repair and verify_first reach molecule_acceptance_rate=0.846, while corpus_retrieval_upper_bound reaches 0.868. However, their overall_task_success/action_accuracy values are 0.650, 0.650, and 0.673, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and task_inconsistent_accept_rate=0.598. This shows that a method can often produce or retrieve a hard-passing molecule while still failing the action semantics of candidate audit, rejection, and abstention.

## Verifier-wrapper Saturation

The deterministic `well_engineered_wrapper` solves the full 266-task test split: action_accuracy=1.000, REJECT_recall=1.000, ABSTAIN_recall=1.000, and task_inconsistent_accept_rate=0.000. This result is expected because sgchem_v1.0 is intentionally machine-checkable. It establishes that the artifact should not be interpreted as intrinsically hard for systems allowed to engineer directly against the public specification contract.

The intended use is narrower and more defensible: measuring whether language-agent interfaces, output schemas, tool-use policies, rejection behavior, abstention behavior, and audit boundaries preserve oracle-certified specification compliance. The wrapper result is therefore a main result, not a caveat to hide.

## Diagnostic Model Behavior

Cached OpenAI snapshot runs in `paper_v1/reality_check_decision_memo.md` show two action-policy failure modes: cheaper model variants are over-conservative on constructive tasks, while frontier-cache variants accept more often but lose rejection recall and incur task-inconsistent acceptance. These model snapshots are diagnostic examples, not the main scientific claim.

The challenge slice is structurally defined from task/spec/oracle metadata and contains 210 of 266 test tasks. Challenge membership does not use baseline success or failure. The slice is primary_reportable by count, but individual diagnostic subfamilies should still be interpreted with their own denominators.

Boundary precision, representation invariance, and interrupt/resume are diagnostic slices in this release: boundary_precision has 22 test tasks, smiles_invariance has 20 test tasks, and interrupt_resume has 11 test tasks. `repair_multi_violation` has 19 test tasks and is diagnostic-only in paper claims.

Generated result tables:

- `paper_v1/tables/primary_results.md`
- `paper_v1/tables/baseline_metric_sanity.md`
- `paper_v1/tables/baseline_action_confusion_matrices.md`
- `paper_v1/tables/wrapper_saturation.md`
- `paper_v1/tables/challenge_slice_results.md`
- `paper_v1/tables/diagnostic_slice_results.md`
