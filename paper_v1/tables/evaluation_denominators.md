# Evaluation Denominators

test tasks: 244
test bundles: 36

| metric | denominator | n | classification |
| --- | --- | ---: | --- |
| overall compliance | all_test_tasks | 244 | primary_reportable |
| ACCEPT precision/recall | expected_ACCEPT | 162 | primary_reportable |
| REJECT precision/recall | expected_REJECT | 46 | primary_reportable |
| ABSTAIN precision/recall | expected_ABSTAIN | 36 | primary_reportable |
| unsafe accept rate | expected_REJECT_or_ABSTAIN | 82 | primary_reportable |
| hard violation rate | all_test_tasks | 244 | primary_reportable |
| repair success | repair_tasks | 40 | primary_reportable |
| boundary precision | boundary_tasks | 20 | diagnostic_only |
| SMILES invariance | invariance_tasks | 20 | diagnostic_only |
| interrupt/resume success | interrupt_tasks | 10 | diagnostic_only |
| tool economy | L3_tasks | 76 | primary_reportable |
| protocol comparison L1/L2/L3 | min_protocol_count | 76 | primary_reportable |
| calibration | all_test_tasks | 244 | primary_reportable |
| risk coverage | all_test_tasks | 244 | primary_reportable |

## Expected action

- ABSTAIN: 36
- ACCEPT: 162
- REJECT: 46

## Protocol

- L1: 92
- L2: 76
- L3: 76

## Task type

- abstain_contradiction: 36
- audit_accept: 36
- audit_reject: 36
- boundary_precision: 20
- construct_feasible: 36
- interrupt_resume: 10
- repair_multi_violation: 20
- repair_near_miss: 20
- smiles_invariance: 20
- tool_forced_l3: 10

## Task type x expected action

- abstain_contradiction x ABSTAIN: 36
- audit_accept x ACCEPT: 36
- audit_reject x REJECT: 36
- boundary_precision x ACCEPT: 10
- boundary_precision x REJECT: 10
- construct_feasible x ACCEPT: 36
- interrupt_resume x ACCEPT: 10
- repair_multi_violation x ACCEPT: 20
- repair_near_miss x ACCEPT: 20
- smiles_invariance x ACCEPT: 20
- tool_forced_l3 x ACCEPT: 10

## Task type x protocol

- abstain_contradiction x L1: 36
- audit_accept x L1: 11
- audit_accept x L2: 12
- audit_accept x L3: 13
- audit_reject x L1: 13
- audit_reject x L2: 11
- audit_reject x L3: 12
- boundary_precision x L2: 20
- construct_feasible x L1: 12
- construct_feasible x L2: 13
- construct_feasible x L3: 11
- interrupt_resume x L3: 10
- repair_multi_violation x L3: 20
- repair_near_miss x L2: 20
- smiles_invariance x L1: 20
- tool_forced_l3 x L3: 10

## Oracle type

- boundary_certificate: 20
- equivalence_certificate: 20
- feasible_witness: 72
- interrupt_certificate: 10
- repair_witness: 50
- unsat_certificate: 36
- violation_certificate: 36

## Spec family

- spec_v1_basic: 20
- spec_v2_alert_soft_train: 30
- spec_v2_alert_strict_dev: 8
- spec_v2_amide_friendly_dev: 33
- spec_v2_aromatic_pref_test: 18
- spec_v2_cns_like_test: 17
- spec_v2_fragment_ultra_dev: 19
- spec_v2_low_rotor_train: 18
- spec_v2_polar_bias_train: 22
- spec_v2_ro5_balanced_test: 31
- spec_v2_sa_relaxed_dev: 17
- spec_v2_sa_strict_test: 11

## Groups

- boundary_groups: 10
- interrupt_groups: 10
- invariance_groups: 10
- tool_forced_groups: 10
