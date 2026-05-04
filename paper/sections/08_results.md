# Results

The main empirical finding is not that sgchem_v1.0 is unsolved by every deterministic baseline. Instead, it is that aggregate molecule acceptance hides action-specific failures.

On the 266-task test split, local_mutation_or_repair and verify_first reach molecule_acceptance_rate=0.846, while corpus_retrieval_upper_bound reaches 0.868. However, their overall_task_success/action_accuracy values are 0.650, 0.650, and 0.673, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and unsafe_accept_rate=0.598. This shows that a method can often produce or retrieve a hard-passing molecule while still failing the action semantics of candidate audit, rejection, and abstention.

The challenge slice is structurally defined from task/spec/oracle metadata and contains 210 of 266 test tasks. Challenge membership does not use baseline success or failure. The slice is primary_reportable by count, but individual diagnostic subfamilies should still be interpreted with their own denominators.

Boundary precision, representation invariance, and interrupt/resume are diagnostic slices in this release: boundary_precision has 22 test tasks, smiles_invariance has 20 test tasks, and interrupt_resume has 11 test tasks. `repair_multi_violation` has 19 test tasks and is diagnostic-only in paper claims.

Generated result tables:

- `paper_v1/tables/primary_results.md`
- `paper_v1/tables/baseline_metric_sanity.md`
- `paper_v1/tables/baseline_action_confusion_matrices.md`
- `paper_v1/tables/challenge_slice_results.md`
- `paper_v1/tables/diagnostic_slice_results.md`
