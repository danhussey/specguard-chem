# Metrics

The paper reports action-aware metrics rather than raw molecule acceptance alone.

Primary metrics include overall_task_success, action_accuracy, ACCEPT precision/recall, REJECT precision/recall, ABSTAIN precision/recall, task_inconsistent_accept_rate, false_abstain_rate, hard_violation_rate, schema_error_rate, invalid_molecule_rate, and pass@1/pass@3 where meaningful.

The metric formerly called `accept_rate` is now reported as `molecule_acceptance_rate`. It means the fraction of tasks where the final decision is ACCEPT. It is not task success and is not a headline metric.

The central sanity result is that local_mutation_or_repair and verify_first have molecule_acceptance_rate=0.846, while the molecule-corpus retrieval baseline has molecule_acceptance_rate=0.868. Their action accuracies are 0.650, 0.650, and 0.673, with REJECT_recall=0.000, ABSTAIN_recall=0.000, and task_inconsistent_accept_rate=0.598. This motivates the paper's action-aware evaluation story.

The wrapper-saturation result is reported separately: `well_engineered_wrapper` reaches action_accuracy=1.000, REJECT_recall=1.000, ABSTAIN_recall=1.000, and task_inconsistent_accept_rate=0.000 on the 266-task test split. This is a ceiling for engineered verifier integration, not a primary model result.

Metric definitions and denominators are generated in:

- `paper_v1/tables/metric_definitions.md`
- `paper_v1/tables/baseline_metric_sanity.md`
- `paper_v1/tables/reject_abstain_metrics.md`
- `paper_v1/tables/task_inconsistent_accept_rate.md`
- `paper_v1/tables/evaluation_denominators.md`
