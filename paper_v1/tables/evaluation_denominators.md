# Evaluation Denominators

test tasks: 266
test bundles: 42

| metric | denominator | n | classification |
| --- | --- | ---: | --- |
| overall compliance | all_test_tasks | 266 | primary_reportable |
| ACCEPT precision/recall | expected_ACCEPT | 179 | primary_reportable |
| REJECT precision/recall | expected_REJECT | 52 | primary_reportable |
| ABSTAIN precision/recall | expected_ABSTAIN | 35 | primary_reportable |
| unsafe accept rate | expected_REJECT_or_ABSTAIN | 87 | primary_reportable |
| hard violation rate | all_test_tasks | 266 | primary_reportable |
| repair success | repair_tasks | 41 | primary_reportable |
| boundary precision | boundary_tasks | 22 | diagnostic_only |
| SMILES invariance | invariance_tasks | 20 | diagnostic_only |
| interrupt/resume success | interrupt_tasks | 11 | diagnostic_only |
| tool economy | L3_tasks | 84 | primary_reportable |
| protocol comparison L1/L2/L3 | min_protocol_count | 84 | primary_reportable |
| calibration | all_test_tasks | 266 | primary_reportable |
| risk coverage | all_test_tasks | 266 | primary_reportable |

## Expected action

- ABSTAIN: 35
- ACCEPT: 179
- REJECT: 52

## Protocol

- L1: 97
- L2: 85
- L3: 84

## Task type

- abstain_contradiction: 35
- audit_accept: 42
- audit_reject: 41
- boundary_precision: 22
- construct_feasible: 41
- interrupt_resume: 11
- repair_multi_violation: 19
- repair_near_miss: 22
- smiles_invariance: 20
- tool_forced_l3: 13

## Task type x expected action

- abstain_contradiction x ABSTAIN: 35
- audit_accept x ACCEPT: 42
- audit_reject x REJECT: 41
- boundary_precision x ACCEPT: 11
- boundary_precision x REJECT: 11
- construct_feasible x ACCEPT: 41
- interrupt_resume x ACCEPT: 11
- repair_multi_violation x ACCEPT: 19
- repair_near_miss x ACCEPT: 22
- smiles_invariance x ACCEPT: 20
- tool_forced_l3 x ACCEPT: 13

## Task type x protocol

- abstain_contradiction x L1: 35
- audit_accept x L1: 15
- audit_accept x L2: 14
- audit_accept x L3: 13
- audit_reject x L1: 13
- audit_reject x L2: 14
- audit_reject x L3: 14
- boundary_precision x L2: 22
- construct_feasible x L1: 14
- construct_feasible x L2: 13
- construct_feasible x L3: 14
- interrupt_resume x L3: 11
- repair_multi_violation x L3: 19
- repair_near_miss x L2: 22
- smiles_invariance x L1: 20
- tool_forced_l3 x L3: 13

## Oracle type

- boundary_certificate: 22
- equivalence_certificate: 20
- feasible_witness: 83
- interrupt_certificate: 11
- repair_witness: 54
- unsat_certificate: 35
- violation_certificate: 41

## Spec family

- spec_v1_basic: 20
- spec_v2_alert_soft_train: 30
- spec_v2_alert_strict_dev: 8
- spec_v2_amide_friendly_dev: 41
- spec_v2_aromatic_pref_test: 18
- spec_v2_cns_like_test: 21
- spec_v2_fragment_ultra_dev: 26
- spec_v2_low_rotor_train: 22
- spec_v2_polar_bias_train: 22
- spec_v2_ro5_balanced_test: 30
- spec_v2_sa_relaxed_dev: 17
- spec_v2_sa_strict_test: 11

## Groups

- boundary_groups: 11
- interrupt_groups: 11
- invariance_groups: 10
- tool_forced_groups: 13
