# Metric Definitions

| metric | definition | denominator |
| --- | --- | --- |
| overall_task_success | fraction of tasks where predicted_action equals expected_action | all evaluated tasks |
| action_accuracy | same decision-level exact match as overall_task_success for sgchem_v1.0 | all evaluated tasks |
| molecule_acceptance_rate | fraction of tasks ending with final_decision=ACCEPT; this was formerly the internal accept_rate and is not task success | all evaluated tasks |
| ACCEPT/REJECT/ABSTAIN precision | true positives for the action divided by all predictions of that action | predicted action count |
| ACCEPT/REJECT/ABSTAIN recall | true positives for the action divided by all tasks expecting that action | expected action count |
| task_inconsistent_accept_rate | ACCEPT predictions on tasks whose expected action is REJECT or ABSTAIN; task-level noncompliance, not molecular safety | expected REJECT plus expected ABSTAIN tasks |
| false_abstain_rate | ABSTAIN predictions on tasks whose expected action is ACCEPT or REJECT | expected ACCEPT plus expected REJECT tasks |
| hard_violation_rate | non-abstain outputs that fail hard constraints under the verifier | non-abstain outputs |
| invalid_molecule_rate | proposal rounds whose verifier feedback indicates invalid SMILES | proposal rounds |
| pass@1/pass@3 | expected-ACCEPT tasks solved with final ACCEPT within one or three steps | expected ACCEPT tasks |
