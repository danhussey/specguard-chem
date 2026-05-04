# Metrics

The paper reports action-aware metrics rather than raw molecule acceptance alone.

Primary metrics include overall_task_success, action_accuracy, ACCEPT precision/recall, REJECT precision/recall, ABSTAIN precision/recall, unsafe_accept_rate, false_abstain_rate, hard_violation_rate, schema_error_rate, invalid_molecule_rate, and pass@1/pass@3 where meaningful.

The metric formerly called `accept_rate` is now reported as `molecule_acceptance_rate`. It means the fraction of tasks where the final decision is ACCEPT. It is not task success and is not a headline metric.

The central sanity result is that local_mutation_or_repair, verify_first, and corpus_retrieval_upper_bound each have molecule_acceptance_rate=0.852, but overall_task_success=0.664, action_accuracy=0.664, REJECT_recall=0.000, ABSTAIN_recall=0.000, and unsafe_accept_rate=0.561. This motivates the paper's action-aware evaluation story.

Metric definitions and denominators are generated in:

- `paper_v1/tables/metric_definitions.md`
- `paper_v1/tables/baseline_metric_sanity.md`
- `paper_v1/tables/reject_abstain_metrics.md`
- `paper_v1/tables/unsafe_accept_rate.md`
- `paper_v1/tables/evaluation_denominators.md`
