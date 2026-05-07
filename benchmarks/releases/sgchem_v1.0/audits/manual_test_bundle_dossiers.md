# Manual Test Bundle Dossiers

## sgchem_v1.0__bundle__00002

- split: test
- spec_id: spec_v2_low_rotor_train
- source molecule: 7f65f44ba6c1914c
- source canonical SMILES: CC(=O)NC(C)C(C)N
- scaffold hash: bd3ca4c729175f99
- task IDs: sgchem_v1.0__bundle__00002__abstain_contradiction__04, sgchem_v1.0__bundle__00002__audit_accept__02, sgchem_v1.0__bundle__00002__audit_reject__03, sgchem_v1.0__bundle__00002__boundary_precision__06, sgchem_v1.0__bundle__00002__boundary_precision__07, sgchem_v1.0__bundle__00002__construct_feasible__01, sgchem_v1.0__bundle__00002__interrupt_resume__08, sgchem_v1.0__bundle__00002__repair_multi_violation__05, sgchem_v1.0__bundle__00002__repair_near_miss__09, sgchem_v1.0__bundle__00002__smiles_invariance__10, sgchem_v1.0__bundle__00002__smiles_invariance__11, sgchem_v1.0__bundle__00002__tool_forced_l3__12
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible, interrupt_resume, repair_multi_violation, repair_near_miss, smiles_invariance, smiles_invariance, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct, repair, repair, repair, representation_invariance, representation_invariance, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 9, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1, 'boundary_precision': 2, 'interrupt_resume': 1, 'repair_near_miss': 1, 'smiles_invariance': 2, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 3, 'boundary_certificate': 2, 'interrupt_certificate': 1, 'equivalence_certificate': 2}, 'expected_actions': {'ACCEPT': 9, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 6}
- agent_visible_hashes: sha256:bd839a936b6575c2f77b4ad9df98bee223fbf92c79e8667ef55c0076c9d84f98, sha256:8b7e41d3237b467eb7fd3e9368613c66cbd48b8ca8f2e8b47cdd62b2ec59849f, sha256:85a5b9b0d0602247554bacdf7410c2e2782ab3ec5b35cb58e6b4977b14121e83, sha256:867b9a3ea146861c901d698ded00e93820fcb486028640e31d98bf74839b5b8c, sha256:3e5aac5ffc8d393acf2372ed6b8ae77743747f3cbb21ac5a38bed2a7e048b434, sha256:3f5f057318c70713b5ea2c49f0b0c233f3d123771145a0c7f4e8d71becddb232, sha256:047a0b4f9dd8ad01d032791784d93589a44bed0bdd6955425cde66d98b2d45ca, sha256:47c2344ec2a608056162e4f76eb400f47dc882d48919e95853a5621ebc7c8492, sha256:37438ca0562cf2508e2bbf736689e7e010beedfb91ae03e5c5c465882cade8bf, sha256:0c64f2219a311c6aca833b0a74bbbff6b624e5a27a0058d545991e486ada6548, sha256:b36a098d9c1b24cf478979cbbccdcd8aa0ab007624138e2a650f5029b9cc1dcc, sha256:d1d120b2c11a08249182f9afcce41a8cd9378304fd0ba0398cc92860755bd599
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00002__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500
2. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00002__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00002__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00002__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00002__boundary_precision__07

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00002__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00002__interrupt_resume__08

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00002__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(CN)NCCN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500
2. repair_distinct_property_guard: HBA between 1.750 and 2.250

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00002__repair_near_miss__09

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00002__smiles_invariance__10

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00002__smiles_invariance__11

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
C(NC(C)=O)(C)C(C)N

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00002__tool_forced_l3__12

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00003

- split: test
- spec_id: spec_v2_polar_bias_train
- source molecule: 7f65f44ba6c1914c
- source canonical SMILES: CC(=O)NC(C)C(C)N
- scaffold hash: bd3ca4c729175f99
- task IDs: sgchem_v1.0__bundle__00003__abstain_contradiction__04, sgchem_v1.0__bundle__00003__audit_accept__02, sgchem_v1.0__bundle__00003__audit_reject__03, sgchem_v1.0__bundle__00003__boundary_precision__05, sgchem_v1.0__bundle__00003__boundary_precision__06, sgchem_v1.0__bundle__00003__construct_feasible__01, sgchem_v1.0__bundle__00003__interrupt_resume__07, sgchem_v1.0__bundle__00003__repair_multi_violation__08, sgchem_v1.0__bundle__00003__repair_near_miss__09, sgchem_v1.0__bundle__00003__smiles_invariance__10, sgchem_v1.0__bundle__00003__smiles_invariance__11, sgchem_v1.0__bundle__00003__tool_forced_l3__12
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible, interrupt_resume, repair_multi_violation, repair_near_miss, smiles_invariance, smiles_invariance, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct, repair, repair, repair, representation_invariance, representation_invariance, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 9, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2, 'interrupt_resume': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1, 'smiles_invariance': 2, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2, 'interrupt_certificate': 1, 'repair_witness': 3, 'equivalence_certificate': 2}, 'expected_actions': {'ACCEPT': 9, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 6}
- agent_visible_hashes: sha256:02332e3cae854a2073701cf8fb0e480c106988711ca2226dc0d3a4c602ca586e, sha256:01d847b5896358d629f55f2d73a21efda7bed6a6ae74b9fefd576c2cf8d2a1c2, sha256:ec0b3c2a73eb646d5cd53335279310be8cd938869cbe50668a91cb014844adc7, sha256:5bac42bc504e5ef7fb61bd7d30a5b035f77044ba931217d44b622bd2d84e5d08, sha256:31378333546b7878fc0bd2d175c4505a6500dccc2fd99d14955ccdd16b34789a, sha256:8a5d0da844c88548d0d32fd82e093d736a025161144ff513f4e612783aea40dc, sha256:7c1a4ad682ebe26ce8759e4ad36e748fb0799ca30165a5dace6d19d689f593b7, sha256:fb5ccf1bbc717ce256c71f110add05ff36020dedaebe2fe13a149cb9ea01660a, sha256:f21740e2bf7df6ed877f1a536d8ac4f68c9e6ce5323acc91c730b0117da7ea97, sha256:3ff1d75e005da86f14ad7f9789a80760d0029b6ae5fa3aed1698c0e4f803891b, sha256:b30681f099ae6f9b4abe0e7d4198d8551c4238b07cd95e6a9aa4cb0b220ea888, sha256:cd27f1dfbc2eb6fd7db89f66cf560da81c1c1d8167e74284305bc07a6f95359b
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00003__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000
2. contradict_hba_minimum: HBA between 13.000 and 14.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00003__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00003__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00003__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00003__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00003__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00003__interrupt_resume__07

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00003__repair_multi_violation__08

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)F

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000
2. repair_distinct_property_guard: MW between 110.191 and 150.191

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00003__repair_near_miss__09

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00003__smiles_invariance__10

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00003__smiles_invariance__11

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
C(NC(C)=O)(C)C(C)N

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00003__tool_forced_l3__12

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00008

- split: test
- spec_id: spec_v2_alert_soft_train
- source molecule: 7f65f44ba6c1914c
- source canonical SMILES: CC(=O)NC(C)C(C)N
- scaffold hash: bd3ca4c729175f99
- task IDs: sgchem_v1.0__bundle__00008__abstain_contradiction__04, sgchem_v1.0__bundle__00008__audit_accept__02, sgchem_v1.0__bundle__00008__audit_reject__03, sgchem_v1.0__bundle__00008__boundary_precision__06, sgchem_v1.0__bundle__00008__boundary_precision__07, sgchem_v1.0__bundle__00008__construct_feasible__01, sgchem_v1.0__bundle__00008__interrupt_resume__08, sgchem_v1.0__bundle__00008__repair_multi_violation__05, sgchem_v1.0__bundle__00008__repair_near_miss__09, sgchem_v1.0__bundle__00008__tool_forced_l3__10
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible, interrupt_resume, repair_multi_violation, repair_near_miss, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct, repair, repair, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 7, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1, 'boundary_precision': 2, 'interrupt_resume': 1, 'repair_near_miss': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 3, 'boundary_certificate': 2, 'interrupt_certificate': 1}, 'expected_actions': {'ACCEPT': 7, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 6}
- agent_visible_hashes: sha256:b186c8e8beba577ea7e34463331edf58495474d78c860c209efd9934dd096281, sha256:29710af7acc6fe5de338fc7d3d069cbae27b2a80e26f81aa120a832b5d691d1f, sha256:f83cdec989fe119e942bb655b00c694f11cbff84c9b03eb544e1d575b1c31a9c, sha256:dffdd9c337e90074b83eb8975c57a33b60bbab69fda041380a310aca6e8f2153, sha256:c0ec760d72c66d3be49f85cc5dd63693a7b69a7d066f4fd7a597ab2fb85e8c94, sha256:37cb149b68236246186963c1cfe60fda06d1335719ccdb19b5f6cd1dfd10b72e, sha256:787bcb12c72fc2d9eb4335a6809b1c048a5c93fd221025f10fc6159d5bc4892b, sha256:d1624de43f253e531b492ce9defa667fa4889a6b5edba0cd673ac4ba5d4f28f9, sha256:9c6d3ea3b385082db95e48b76b73c026c61b77dff3cc7da3b25f0110161b9bb9, sha256:c773396ddeb1d6433cb34622081a96557b29bad577004a1ba2f825fcde0abb08
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00008__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500
2. contradict_hba_minimum: HBA between 13.000 and 14.000

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00008__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00008__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00008__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00008__boundary_precision__07

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00008__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00008__interrupt_resume__08

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00008__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(N)C(N)=O

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500
2. repair_distinct_property_guard: MW between 110.191 and 150.191

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00008__repair_near_miss__09

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00008__tool_forced_l3__10

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00010

- split: test
- spec_id: spec_v2_amide_friendly_dev
- source molecule: 7f65f44ba6c1914c
- source canonical SMILES: CC(=O)NC(C)C(C)N
- scaffold hash: bd3ca4c729175f99
- task IDs: sgchem_v1.0__bundle__00010__abstain_contradiction__04, sgchem_v1.0__bundle__00010__audit_accept__02, sgchem_v1.0__bundle__00010__audit_reject__03, sgchem_v1.0__bundle__00010__construct_feasible__01, sgchem_v1.0__bundle__00010__interrupt_resume__07, sgchem_v1.0__bundle__00010__repair_multi_violation__08, sgchem_v1.0__bundle__00010__repair_near_miss__09, sgchem_v1.0__bundle__00010__smiles_invariance__05, sgchem_v1.0__bundle__00010__smiles_invariance__06, sgchem_v1.0__bundle__00010__tool_forced_l3__10
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, interrupt_resume, repair_multi_violation, repair_near_miss, smiles_invariance, smiles_invariance, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair, repair, representation_invariance, representation_invariance, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 8, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'smiles_invariance': 2, 'interrupt_resume': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'equivalence_certificate': 2, 'interrupt_certificate': 1, 'repair_witness': 3}, 'expected_actions': {'ACCEPT': 8, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 6}
- agent_visible_hashes: sha256:1dec1810ee1c94b3e8a1d4ad9f8d8e8707ccd74b3acf871e75fbf724c1f009ab, sha256:736be82ec9024f6ec0e3f5731fd5fa95c072f2339c406ef3c63d3c51b7dfb69f, sha256:a7dbda22f537cddc1c0888f94d6006903e8542ba530341befc2e4c665b873357, sha256:3ff4e40f04e59e1d636c6cc77cd580f4091061c6b08ca84fe02e842d09b21095, sha256:6c2e67c40bcb831189f119876714342e28c92feb05943a70c2b98187d69158ac, sha256:e52905a4b3237d1fe73db4420f10d5cc14753a61e0f7bd592f0c1a22e515b6cf, sha256:3173f2c3ff912954a7b3bda1c47d3870cc5318aee8af38f4fe1113f70e7cb146, sha256:15438e5300c06a72d8d9c7b4c477cb9589bcefe6d370feb396b7ddc513520d99, sha256:5b3dda0e28132871da2185011c72dab069321aee78ad7bec83535e0b0a71aa92, sha256:469ee890e14b702ce98d5ff9abcee6662f42777e8e9cf0566f74c4eeb02bfaa6
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00010__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500
2. contradict_hba_minimum: HBA between 12.000 and 13.000

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00010__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00010__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00010__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00010__interrupt_resume__07

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00010__repair_multi_violation__08

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(C)Cl

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500
2. repair_distinct_property_guard: MW between 110.191 and 150.191

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00010__repair_near_miss__09

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00010__smiles_invariance__05

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00010__smiles_invariance__06

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
C(NC(C)=O)(C)C(C)N

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00010__tool_forced_l3__10

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 85.191 and 175.191; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00012

- split: test
- spec_id: spec_v2_cns_like_test
- source molecule: c5ac454de96fcc34
- source canonical SMILES: CC(=O)Nc1c(Cl)cccc1C(N)=O
- scaffold hash: 13cad05ca8f49c50
- task IDs: sgchem_v1.0__bundle__00012__abstain_contradiction__04, sgchem_v1.0__bundle__00012__audit_accept__02, sgchem_v1.0__bundle__00012__audit_reject__03, sgchem_v1.0__bundle__00012__construct_feasible__01, sgchem_v1.0__bundle__00012__tool_forced_l3__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:50536be9d0bdbe5aa1e8a686e915dc6606827015d788eda8a4bc59827b534c5c, sha256:fb3300cb05c5328f2a03b0e5e2d24e405fb23db997d74e243939cfd23362bdce, sha256:fba48611cb64a36d8b252f5e5ff923a1a8a62d970ec4ad7bafc322dece2cfb4a, sha256:4a1dbaa9a0fda775e614631894e592151c58bf53906cf19ad8406901c6cd30c7, sha256:09ce54be9d9902f3f8a6a18d852fc86c5317dc7425bf7468db43c58c02909d36
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00012__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500
3. contradict_hba_minimum: HBA between 9.000 and 10.000

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00012__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1c(Cl)cccc1C(N)=O

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00012__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00012__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00012__tool_forced_l3__05

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00016

- split: test
- spec_id: spec_v2_polar_bias_train
- source molecule: 296a93091525a6b5
- source canonical SMILES: CC(=O)NC(C)C(C)O
- scaffold hash: e16e277766198a24
- task IDs: sgchem_v1.0__bundle__00016__abstain_contradiction__04, sgchem_v1.0__bundle__00016__audit_accept__02, sgchem_v1.0__bundle__00016__audit_reject__03, sgchem_v1.0__bundle__00016__construct_feasible__01, sgchem_v1.0__bundle__00016__interrupt_resume__07, sgchem_v1.0__bundle__00016__repair_multi_violation__08, sgchem_v1.0__bundle__00016__repair_near_miss__09, sgchem_v1.0__bundle__00016__smiles_invariance__05, sgchem_v1.0__bundle__00016__smiles_invariance__06, sgchem_v1.0__bundle__00016__tool_forced_l3__10
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, interrupt_resume, repair_multi_violation, repair_near_miss, smiles_invariance, smiles_invariance, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair, repair, representation_invariance, representation_invariance, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 8, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'smiles_invariance': 2, 'interrupt_resume': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'equivalence_certificate': 2, 'interrupt_certificate': 1, 'repair_witness': 3}, 'expected_actions': {'ACCEPT': 8, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 6}
- agent_visible_hashes: sha256:ffad3588d573d24300b5b780b04850feb25aeb789f686a1d6a5fb1262761f92d, sha256:3aaffad6881ede9c2bd2f3ad37595e77318009c119e32d7663d23e1965c31521, sha256:8167b323aa95c9cfcf5df7c01bfcdc00396fd392fc1c5610490fe74daf25a7f8, sha256:eb1d2d02d6864c384e2145065ee86cb99b36c90e1b994443832129a251af8d52, sha256:cc04935eb8cc723bb185d47013f3148cc9e754f62a88f3bd8b75c3465cf4fd92, sha256:fb5db3ea7b0643240834df9fd07fd91d66ebc5b814ca672c47f59be54496bdd5, sha256:8831cf9591ee51bd40994180e5f5545d709167800696019a9a29133bc08c9d69, sha256:d38a666f99a31d50047e224954c719ce54315f55b503794dfd591d6f204e5977, sha256:14e86322e57151c45086d95272b1175134b17db1ea7b4a26470e44455de2037d, sha256:15c51fa8c8d143b4789162d4e3776bd386fdd763c59c9f136e577c3bef2fea16
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00016__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000
2. contradict_hba_minimum: HBA between 13.000 and 14.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00016__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)O

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00016__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00016__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00016__interrupt_resume__07

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00016__repair_multi_violation__08

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)F

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000
2. repair_distinct_property_guard: MW between 111.175 and 151.175

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00016__repair_near_miss__09

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00016__smiles_invariance__05

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)O

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00016__smiles_invariance__06

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
C(NC(C)=O)(C)C(C)O

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00016__tool_forced_l3__10

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. polar_bounds: HBA between 1.000 and 12.000; HBD between 0.000 and 6.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; TPSA between 40.000 and 180.000; logP between -1.000 and 3.000

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00017

- split: test
- spec_id: spec_v2_ro5_balanced_test
- source molecule: 296a93091525a6b5
- source canonical SMILES: CC(=O)NC(C)C(C)O
- scaffold hash: e16e277766198a24
- task IDs: sgchem_v1.0__bundle__00017__abstain_contradiction__04, sgchem_v1.0__bundle__00017__audit_accept__02, sgchem_v1.0__bundle__00017__audit_reject__03, sgchem_v1.0__bundle__00017__construct_feasible__01, sgchem_v1.0__bundle__00017__interrupt_resume__05, sgchem_v1.0__bundle__00017__repair_multi_violation__06, sgchem_v1.0__bundle__00017__repair_near_miss__07, sgchem_v1.0__bundle__00017__tool_forced_l3__08
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, interrupt_resume, repair_multi_violation, repair_near_miss, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 6, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'interrupt_resume': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'interrupt_certificate': 1, 'repair_witness': 3}, 'expected_actions': {'ACCEPT': 6, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 6}
- agent_visible_hashes: sha256:321e7f03ed2c73f82680412818590a2bfaa7221112325c2254082afacb6e7837, sha256:30b6a410d7ef327c209ac34d691a7861921f5f3509d37aaf8b3e8ce3c5e7781b, sha256:ea091c30a6ab9be250c86677d7750d7a5f957c6ea7c6ac6f7a2076e555382898, sha256:b765038bcece552d5c00e6d3615a480d2eb1cae1da12a091d8ab4a841f4ca20c, sha256:11868eadf3f25af9bb1992d8f80e64a1c728311eef2638a4cef46694a29135a8, sha256:51e38aee4150b0eddaee73dec3529ead541864f0d56d2525c84ef36fd51ea3ed, sha256:7b8bd3ce7875f4a47c9df81c814c4b37f9a09c8feee06b213a69c07b4a8c7e9c, sha256:e18365ec4af280a1de25c6ae7a04aac5697a9f53069279831b46b1bb4b8a0f9d
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00017__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK
3. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00017__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)O

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00017__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00017__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00017__interrupt_resume__05

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00017__repair_multi_violation__06

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)F

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK
3. repair_distinct_property_guard: MW between 111.175 and 151.175

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00017__repair_near_miss__07

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00017__tool_forced_l3__08

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00020

- split: test
- spec_id: spec_v1_basic
- source molecule: d10dd842b0a3abd0
- source canonical SMILES: CC(=O)NC(C)F
- scaffold hash: 127e5e4d7bb8788c
- task IDs: sgchem_v1.0__bundle__00020__abstain_contradiction__04, sgchem_v1.0__bundle__00020__audit_accept__02, sgchem_v1.0__bundle__00020__audit_reject__03, sgchem_v1.0__bundle__00020__construct_feasible__01, sgchem_v1.0__bundle__00020__interrupt_resume__06, sgchem_v1.0__bundle__00020__repair_multi_violation__05, sgchem_v1.0__bundle__00020__repair_near_miss__07, sgchem_v1.0__bundle__00020__tool_forced_l3__08
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, interrupt_resume, repair_multi_violation, repair_near_miss, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 6, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1, 'interrupt_resume': 1, 'repair_near_miss': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 3, 'interrupt_certificate': 1}, 'expected_actions': {'ACCEPT': 6, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 6}
- agent_visible_hashes: sha256:b12d4170dfe75385fc3759dff031f2095a9a0d272f0121a80bfcff3821652b39, sha256:5d67b51b6223349709009f8146152c9392289bf243158b9ed04d6a1a58c32698, sha256:9b363a3f9e2463ad5182c0301d9b59f6450dd35c63a2d2b7966920bcb3aee280, sha256:9317dd79816028ff0c9bceed2c12fe2208cdf726422d03da2dc156bf81726228, sha256:c6e8a14575ebdaad7f7062721b8a0980ee04fa19355d33f27f72be78decf3b2a, sha256:b86e6e994041e2e66ce668b2b0b4c2489424d9bb8f5dccb6ebb0e620956c74ec, sha256:bd379cdf2b4c9a40649a354a6d42e51967701cb1f9b638af267384e4d3a84a4c, sha256:2e53ae244d4d3a867a3bf695ca4fb78e1040a1b6141cea8323d6bf7a027b4a6f
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00020__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000
2. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00020__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)F

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00020__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00020__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00020__interrupt_resume__06

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00020__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NCN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000
2. repair_distinct_property_guard: TPSA between 14.100 and 44.100

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00020__repair_near_miss__07

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00020__tool_forced_l3__08

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00021

- split: test
- spec_id: spec_v2_alert_soft_train
- source molecule: 296a93091525a6b5
- source canonical SMILES: CC(=O)NC(C)C(C)O
- scaffold hash: e16e277766198a24
- task IDs: sgchem_v1.0__bundle__00021__abstain_contradiction__04, sgchem_v1.0__bundle__00021__audit_accept__02, sgchem_v1.0__bundle__00021__audit_reject__03, sgchem_v1.0__bundle__00021__boundary_precision__05, sgchem_v1.0__bundle__00021__boundary_precision__06, sgchem_v1.0__bundle__00021__construct_feasible__01, sgchem_v1.0__bundle__00021__interrupt_resume__07, sgchem_v1.0__bundle__00021__repair_multi_violation__08, sgchem_v1.0__bundle__00021__repair_near_miss__09
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible, interrupt_resume, repair_multi_violation, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct, repair, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 6, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2, 'interrupt_resume': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2, 'interrupt_certificate': 1, 'repair_witness': 2}, 'expected_actions': {'ACCEPT': 6, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 5}
- agent_visible_hashes: sha256:0b62b5badc8267177c235dc5bc90ea9e142d78751a3ed70f8edf266de814af16, sha256:a28ed2287398ea6626b1275f5589a04d3eefde2ddc61f3caece74dd5f0473de4, sha256:6156d83dc6f6d36b96b279b0a0838347f04d063463cd0578229b2f38d363cb53, sha256:46c14733cd39899ed6302a9911458411ab67f8885794b4085e5e5ef28e10d9df, sha256:f577036a03fe33dbd0b62441e47e7e91fa3dbbd1ac290b30abd2262a15b50a6f, sha256:74b5f2ffc4650f9cab681f59a8007fa114af927733858be3a73f64795ae97900, sha256:ecbc8b15221f782d5917fb5be1dff01e5ab86cc5f6d832de90cd882eb570dba7, sha256:807d6fc6450c6a4c695eeb0b3dd3295a06f04c7bad0f74f9116b141df2d95277, sha256:5b96d4901683d65c0088713da14c26b26a881ce9943456d7def6195ffacf7429
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00021__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500
2. contradict_hba_minimum: HBA between 13.000 and 14.000

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00021__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)O

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00021__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00021__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)O

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00021__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00021__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00021__interrupt_resume__07

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00021__repair_multi_violation__08

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(N)C(N)=O

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500
2. repair_distinct_property_guard: MW between 111.175 and 151.175

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00021__repair_near_miss__09

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00022

- split: test
- spec_id: spec_v2_alert_strict_dev
- source molecule: 296a93091525a6b5
- source canonical SMILES: CC(=O)NC(C)C(C)O
- scaffold hash: e16e277766198a24
- task IDs: sgchem_v1.0__bundle__00022__abstain_contradiction__04, sgchem_v1.0__bundle__00022__audit_accept__02, sgchem_v1.0__bundle__00022__audit_reject__03, sgchem_v1.0__bundle__00022__construct_feasible__01, sgchem_v1.0__bundle__00022__repair_multi_violation__07, sgchem_v1.0__bundle__00022__repair_near_miss__08, sgchem_v1.0__bundle__00022__smiles_invariance__05, sgchem_v1.0__bundle__00022__smiles_invariance__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation, repair_near_miss, smiles_invariance, smiles_invariance
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair, representation_invariance, representation_invariance
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 6, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'smiles_invariance': 2, 'repair_multi_violation': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'equivalence_certificate': 2, 'repair_witness': 2}, 'expected_actions': {'ACCEPT': 6, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 4}
- agent_visible_hashes: sha256:62fceabec78cbac43d0d49f6d6a8badde4c263e651740538faf0c3f62cfd0667, sha256:7190d2980474060efeda1bede5812d1a6f927266845efb1f84622eb8eb8c4b22, sha256:805c6d8e31bbf25cc465311009a42b7783cf16f7f6653a527670b9d3351af42a, sha256:7ed93b64d20ef7c282935438c03abfff257ff0b63283c7d76bdf9359ca85df3f, sha256:8f91f9b3823ad54f25d5461c91fa0d06491e1103fbfdf8e8809a052479d54d61, sha256:7389e257df7671039dac507d874df16a474731961f03e71d367bd0be6eee4ffc, sha256:745c9fa95804cdb34bbdfe90a6a8683494c43bbcc691843c03ce8a20c627d73c, sha256:aaf6ce1b21f3092a9848052ee6511f0a49e29ee5a2e9272874df7cfae0367021
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00022__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. pains_a_block: alert set absent: PAINS_A
3. pains_b_block: alert set absent: PAINS_B
4. brenk_block: alert set absent: BRENK
5. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00022__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)O

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. pains_a_block: alert set absent: PAINS_A
3. pains_b_block: alert set absent: PAINS_B
4. brenk_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00022__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. pains_a_block: alert set absent: PAINS_A
3. pains_b_block: alert set absent: PAINS_B
4. brenk_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00022__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. pains_a_block: alert set absent: PAINS_A
3. pains_b_block: alert set absent: PAINS_B
4. brenk_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00022__repair_multi_violation__07

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NCF

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. pains_a_block: alert set absent: PAINS_A
3. pains_b_block: alert set absent: PAINS_B
4. brenk_block: alert set absent: BRENK
5. repair_distinct_property_guard: MW between 111.175 and 151.175

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00022__repair_near_miss__08

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. pains_a_block: alert set absent: PAINS_A
3. pains_b_block: alert set absent: PAINS_B
4. brenk_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00022__smiles_invariance__05

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)O

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. pains_a_block: alert set absent: PAINS_A
3. pains_b_block: alert set absent: PAINS_B
4. brenk_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00022__smiles_invariance__06

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
C(NC(C)=O)(C)C(C)O

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. pains_a_block: alert set absent: PAINS_A
3. pains_b_block: alert set absent: PAINS_B
4. brenk_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. contextual_property_preference: MW between 86.175 and 176.175; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

## sgchem_v1.0__bundle__00027

- split: test
- spec_id: spec_v2_fragment_ultra_dev
- source molecule: df004b6818d7a63f
- source canonical SMILES: CC(=O)NC(C)CO
- scaffold hash: ed5a74145d23df89
- task IDs: sgchem_v1.0__bundle__00027__abstain_contradiction__04, sgchem_v1.0__bundle__00027__audit_accept__02, sgchem_v1.0__bundle__00027__audit_reject__03, sgchem_v1.0__bundle__00027__boundary_precision__05, sgchem_v1.0__bundle__00027__boundary_precision__06, sgchem_v1.0__bundle__00027__construct_feasible__01
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:1224e16568c3ec329d0d678b228fa4b6b74644a2b809403181f3a0c311ef83ce, sha256:d9a96128576f1169b3f03b844f6e430fb3e40bb22e52830583342e933e050d6f, sha256:4e93f79f8bab59d54d8d8bbea8c551f153239447209413ccfd5ffbfbf23c0c70, sha256:2d947f55a84d36b4fd0b2a677fa5be8218a8b94644c5f23c215a2bbcdda899bd, sha256:c23aab579e5ac4b81c689d636ab11fe0918257585949012799fba51079f277eb, sha256:6b3b50329fd8a07991f93082a987433d7f587c7db68132255a4831b447ae531e
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00027__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500
2. contradict_hba_minimum: HBA between 6.000 and 7.000

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00027__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CO

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00027__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00027__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CO

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00027__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00027__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00030

- split: test
- spec_id: spec_v2_ro5_balanced_test
- source molecule: d8e2500f0c65dab2
- source canonical SMILES: CC(=O)NC(CN)CF
- scaffold hash: bdede2db5ecffff1
- task IDs: sgchem_v1.0__bundle__00030__abstain_contradiction__04, sgchem_v1.0__bundle__00030__audit_accept__02, sgchem_v1.0__bundle__00030__audit_reject__03, sgchem_v1.0__bundle__00030__construct_feasible__01, sgchem_v1.0__bundle__00030__repair_multi_violation__06, sgchem_v1.0__bundle__00030__repair_near_miss__07, sgchem_v1.0__bundle__00030__tool_forced_l3__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation, repair_near_miss, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 5, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'tool_forced_l3': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 3}, 'expected_actions': {'ACCEPT': 5, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 5}
- agent_visible_hashes: sha256:becbbd199abcf0e562351072fc9160926891be634fd01ee05c2b313ca98f78bc, sha256:0a5f417f34c4df7796b4b2decdeffd4e60850ff877cc729b20cea494eef0736f, sha256:d4020efde3c14688a63c76e2d43c97bdb7798c6da82ee1970f42bf2593335a8b, sha256:08c5f5435be628cae6ee46252767649528fc721d470e2e872b3956fa04ff3b86, sha256:16bc7bbdcb9b762152e3e12ff91be37a9f9e3f9ce2b1bc0b648b42b8264da45d, sha256:4f99d4c2b854a33482a1944683bb2ceba4cf09ea4b22783d8fb250f086eb05d0, sha256:6be59d64a765df75ea4fb7f38c588c42f5b69f97d57de2711fc33e7369899cf1
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00030__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK
3. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00030__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CF

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00030__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00030__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00030__repair_multi_violation__06

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)F

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK
3. repair_distinct_property_guard: MW between 114.154 and 154.154

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00030__repair_near_miss__07

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00030__tool_forced_l3__05

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00036

- split: test
- spec_id: spec_v2_amide_friendly_dev
- source molecule: 2101172a8034eec6
- source canonical SMILES: CC(=O)NC(C)Cl
- scaffold hash: f7286adb4e2e4004
- task IDs: sgchem_v1.0__bundle__00036__abstain_contradiction__04, sgchem_v1.0__bundle__00036__audit_accept__02, sgchem_v1.0__bundle__00036__audit_reject__03, sgchem_v1.0__bundle__00036__construct_feasible__01, sgchem_v1.0__bundle__00036__repair_multi_violation__06, sgchem_v1.0__bundle__00036__repair_near_miss__07, sgchem_v1.0__bundle__00036__tool_forced_l3__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation, repair_near_miss, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 5, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'tool_forced_l3': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 3}, 'expected_actions': {'ACCEPT': 5, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 5}
- agent_visible_hashes: sha256:46a1058e2e24034fb2286e0c5d462818fb99cc1f095413e86bcd5b51ee68fd85, sha256:9b011fed100ab5000018634967cb9141b106d1a0d38af10174159a9ada52cf8a, sha256:1a377820f174454af1d31f4dc1ea546dcf24c8d73af3bb16664afc6fcaf98c62, sha256:0d4bf4879abc493b5f23488e782fdb5f4b908ff4cbb20413ae752274c34f63e9, sha256:e742799241a4f5b9414711476482c0d061729eaf7034a55551d7c07d02d4b5d0, sha256:883042d2edf0a10cbde79e47d1454eb183359fbdef72617b573d476e30b2cca7, sha256:163e4bb666fbed3195511e0ad441b81b2bc7f0db98c0f0801adce2191872dcac
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00036__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500
2. contradict_hba_minimum: HBA between 12.000 and 13.000

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00036__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00036__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00036__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00036__repair_multi_violation__06

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(C)Cl

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500
2. repair_distinct_property_guard: MW between 101.567 and 141.567

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00036__repair_near_miss__07

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00036__tool_forced_l3__05

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00037

- split: test
- spec_id: spec_v2_aromatic_pref_test
- source molecule: 55fdbec34d6d211e
- source canonical SMILES: CC(=O)NC(C)CN
- scaffold hash: 500dd24665fbc7f0
- task IDs: sgchem_v1.0__bundle__00037__abstain_contradiction__04, sgchem_v1.0__bundle__00037__audit_accept__02, sgchem_v1.0__bundle__00037__audit_reject__03, sgchem_v1.0__bundle__00037__construct_feasible__01, sgchem_v1.0__bundle__00037__repair_multi_violation__06, sgchem_v1.0__bundle__00037__repair_near_miss__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 4, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_near_miss': 1, 'repair_multi_violation': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 2}, 'expected_actions': {'ACCEPT': 4, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 4}
- agent_visible_hashes: sha256:f1660b0f29243cb3687cd2a9f98e96ff4f9865342817b9222c9da8e5cfefc971, sha256:5404b6399f1906afedeb7c62a4ed30fe7b93f1dbd24684c8b6132ec6808117ef, sha256:4acd73a5c9a4f5de6905baedec8732c54cdd25e75e8c37b65a2c0dea47a5b25f, sha256:56b19c944a1509ea83715c97af7ba898955f7d2585c1624c3f2c35b56e4b4c98, sha256:e5b799e151db83f51372eebe1b107b1fd1f7d3b21a84eb4d06b550f6c277e32b, sha256:981778c7b99a2572641b43b8e1a3afcb1fb06ac84e5adee6c4e4ceb54cc13816
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00037__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 71.164 and 161.164; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00037__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 71.164 and 161.164; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00037__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 71.164 and 161.164; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00037__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 71.164 and 161.164; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00037__repair_multi_violation__06

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(N)C(N)=O

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. repair_distinct_property_guard: MW between 96.164 and 136.164

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 71.164 and 161.164; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00037__repair_near_miss__05

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 71.164 and 161.164; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00038

- split: test
- spec_id: spec_v2_cns_like_test
- source molecule: 7b4ea3caa44e9b74
- source canonical SMILES: CC(=O)Nc1c(F)cccc1Cl
- scaffold hash: 13cad05ca8f49c50
- task IDs: sgchem_v1.0__bundle__00038__abstain_contradiction__04, sgchem_v1.0__bundle__00038__audit_accept__02, sgchem_v1.0__bundle__00038__audit_reject__03, sgchem_v1.0__bundle__00038__construct_feasible__01, sgchem_v1.0__bundle__00038__repair_multi_violation__05, sgchem_v1.0__bundle__00038__repair_near_miss__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 4, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 2}, 'expected_actions': {'ACCEPT': 4, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 4}
- agent_visible_hashes: sha256:1cc16451dcbd1514583cf9bec9ecddf7b9eeb14286cd13055224180e2202b920, sha256:4cc61ff896501a4b4a7fb626dd8014452f33313cf3a418165832c0ce046a95b7, sha256:4408468d19fb09ccfc645434d9b43201c8385e7e0b4763b6ee170da61127e6c7, sha256:51bdce3cae2f8919de680da6f8fb397e6dfc4a5364dbab71bf851af1d8e48b5f, sha256:3c322825365a7fbefeadb31fbd8a5253fc1d727fd32dd29aabaa0f28f90f7c7a, sha256:a3cb4d66764d64ad1fe6cbb144c8abd28c41e2d39637c8d1ce9b6e2c384fb1c8
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00038__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500
3. contradict_hba_minimum: HBA between 9.000 and 10.000

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00038__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1c(F)cccc1Cl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00038__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00038__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00038__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500
3. repair_distinct_property_guard: MW between 167.601 and 207.601

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00038__repair_near_miss__06

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00040

- split: test
- spec_id: spec_v2_fragment_ultra_dev
- source molecule: 2101172a8034eec6
- source canonical SMILES: CC(=O)NC(C)Cl
- scaffold hash: f7286adb4e2e4004
- task IDs: sgchem_v1.0__bundle__00040__abstain_contradiction__04, sgchem_v1.0__bundle__00040__audit_accept__02, sgchem_v1.0__bundle__00040__audit_reject__03, sgchem_v1.0__bundle__00040__construct_feasible__01, sgchem_v1.0__bundle__00040__repair_multi_violation__07, sgchem_v1.0__bundle__00040__repair_near_miss__08, sgchem_v1.0__bundle__00040__smiles_invariance__05, sgchem_v1.0__bundle__00040__smiles_invariance__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation, repair_near_miss, smiles_invariance, smiles_invariance
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair, representation_invariance, representation_invariance
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 6, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'smiles_invariance': 2, 'repair_multi_violation': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'equivalence_certificate': 2, 'repair_witness': 2}, 'expected_actions': {'ACCEPT': 6, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 4}
- agent_visible_hashes: sha256:90ab81d93c3bbcab7c36bfc2125100d2aada1bc25b0415a3fefa1828a8510b12, sha256:07c8959aca9b47d11e2e74ca25b54426b9076b70f42b8a6f896af60ab21a4d20, sha256:15bc354eb1aa3814c474246518183484ac20a042668cacfa3531964fb96551b0, sha256:984f7c42d2c58d9745e33cc76f51cb60afe1092051733d1a75c9b759739afab5, sha256:a932539db865b3223e9f6cd0f2eb82a5424ae71666dc345b560d5067d62f3eb3, sha256:a0ef11432ba778644ee274887c98dbd81ca1d38d52ef0f6ea2f0b8cac2bee641, sha256:edd3e041b578d50bbafdd7e16a8d332c3d527bf44d9a7695911c1f6bcea9ff65, sha256:2604922eca390f12c9024d84d64331a50466b2e07add7e53d6fb90daf88f2200
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00040__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500
2. contradict_hba_minimum: HBA between 6.000 and 7.000

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00040__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00040__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00040__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00040__repair_multi_violation__07

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500
2. repair_distinct_property_guard: TPSA between 14.100 and 44.100

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00040__repair_near_miss__08

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00040__smiles_invariance__05

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00040__smiles_invariance__06

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
N(C(C)=O)C(C)Cl

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

## sgchem_v1.0__bundle__00041

- split: test
- spec_id: spec_v2_low_rotor_train
- source molecule: 637c000f776f2a98
- source canonical SMILES: CC(=O)NC(CN)CCl
- scaffold hash: 374ee134b9503f49
- task IDs: sgchem_v1.0__bundle__00041__abstain_contradiction__04, sgchem_v1.0__bundle__00041__audit_accept__02, sgchem_v1.0__bundle__00041__audit_reject__03, sgchem_v1.0__bundle__00041__construct_feasible__01, sgchem_v1.0__bundle__00041__interrupt_resume__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, interrupt_resume
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'interrupt_resume': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'interrupt_certificate': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:9c908f8b2c541c41fa7904bbce01f6342fc6c0ead985d9239d80edf1ee233e19, sha256:7666a409935f62256f48f2232ac471d5b015479ccfe6dfce64c2ae25a95b9deb, sha256:e0eb2d2615ce257506800ab3c250aa529f92ac11d56969740b93f21f82720c2f, sha256:c08407f2a03fb6e9b86b0a3a5cd3a0ae50ad302117a4f07f2b33cd9e7ed0668d, sha256:dead0b1557dc9750872abe1e20cb75fef6e3f80ade9654e05a354368828ec1d2
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00041__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500
2. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00041__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00041__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00041__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00041__interrupt_resume__05

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00043

- split: test
- spec_id: spec_v2_ro5_balanced_test
- source molecule: 241b8eafc9e7b2f4
- source canonical SMILES: CC(=O)NC(CO)CF
- scaffold hash: e57f87ac3bce842a
- task IDs: sgchem_v1.0__bundle__00043__abstain_contradiction__04, sgchem_v1.0__bundle__00043__audit_accept__02, sgchem_v1.0__bundle__00043__audit_reject__03, sgchem_v1.0__bundle__00043__construct_feasible__01, sgchem_v1.0__bundle__00043__repair_near_miss__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:fd868aafb53cfdf67a241926f8627a6de1a7687d6125a670e47fda54dff5859c, sha256:18fb6b2a572920835fb527eda3e8899e435d477c113e60336aee4b36ecc2f2bd, sha256:4919fb6cd3685afb4427685ec066ae5c26212e350cb58f8ca644afbcc935be18, sha256:8da22fa12c3a656264c7ccf79383bdf5acf9d389fa41454cb95fc709e36d178f, sha256:c1158958f138ef423189453609ce89ce13165e2ab25714890e4e965c960a2431
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00043__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK
3. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00043__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CO)CF

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00043__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00043__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00043__repair_near_miss__05

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00044

- split: test
- spec_id: spec_v2_sa_relaxed_dev
- source molecule: df004b6818d7a63f
- source canonical SMILES: CC(=O)NC(C)CO
- scaffold hash: ed5a74145d23df89
- task IDs: sgchem_v1.0__bundle__00044__abstain_contradiction__04, sgchem_v1.0__bundle__00044__audit_accept__02, sgchem_v1.0__bundle__00044__audit_reject__03, sgchem_v1.0__bundle__00044__construct_feasible__01, sgchem_v1.0__bundle__00044__repair_multi_violation__05, sgchem_v1.0__bundle__00044__repair_near_miss__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 4, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 2}, 'expected_actions': {'ACCEPT': 4, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 4}
- agent_visible_hashes: sha256:bedb06d2fc2a578ae7cf0cc09aca98d3403f310769c0599d5ec0819b939bee42, sha256:a22adf44b1f893a9d2859ed50a622a938c4b436918b0676be79f5e57190de261, sha256:f9134fb7104f7bc8fb101282f2b9d00ae778dc19dd9752752ae9626362e70c23, sha256:e551421376089290a6a9a5a9cc34844cd0383f8d3880999906824818fb958e77, sha256:e983b207bc439a410609ac7184fbebe6b087d3d8e637190506144206f46d2b42, sha256:b8fe825a78685ffc50a6da48b62e1a6334976e06a8621947a466f6b381152642
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00044__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000
3. contradict_hba_minimum: HBA between 12.000 and 13.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00044__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CO

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00044__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00044__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00044__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(N)C(N)=O

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000
3. repair_distinct_property_guard: MW between 97.148 and 137.148

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00044__repair_near_miss__06

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00046

- split: test
- spec_id: spec_v1_basic
- source molecule: c5ac454de96fcc34
- source canonical SMILES: CC(=O)Nc1c(Cl)cccc1C(N)=O
- scaffold hash: 13cad05ca8f49c50
- task IDs: sgchem_v1.0__bundle__00046__abstain_contradiction__04, sgchem_v1.0__bundle__00046__audit_accept__02, sgchem_v1.0__bundle__00046__audit_reject__03, sgchem_v1.0__bundle__00046__construct_feasible__01, sgchem_v1.0__bundle__00046__repair_near_miss__07, sgchem_v1.0__bundle__00046__smiles_invariance__05, sgchem_v1.0__bundle__00046__smiles_invariance__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_near_miss, smiles_invariance, smiles_invariance
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, representation_invariance, representation_invariance
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 5, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'smiles_invariance': 2, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'equivalence_certificate': 2, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 5, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:31c1ea6e329a7e734c8e84187eceb272c51367f29cec261fdf3d2184935be3e1, sha256:18eb609429b34f29b20e572ec0098b287dcae9e8e6f535a0b073139182416f5c, sha256:4dd0b70481efe50ea656342410a8302946872768c4702773373b709fb701930f, sha256:fb32b04401d5b355a3aaf1e564a447ab876cbf01eeeb9acecd34ceebdd033e6a, sha256:b233e1155fe59db4ad87e75edd8b8509fa1d53e4e1f07cdceca08556fbbef8c1, sha256:7a8d824dfe304366cac739bfd54c0f1c09a147122c8b492a847bb0c0b9b7e4ab, sha256:88b6c031d31bcc945acaa71b035723ee5f6b1944cdc426c21edb61bb2dac891d
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00046__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000
2. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00046__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1c(Cl)cccc1C(N)=O

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00046__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00046__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00046__repair_near_miss__07

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00046__smiles_invariance__05

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1c(Cl)cccc1C(N)=O

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00046__smiles_invariance__06

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
c1c(Cl)c(NC(C)=O)c(C(N)=O)cc1

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

## sgchem_v1.0__bundle__00049

- split: test
- spec_id: spec_v2_amide_friendly_dev
- source molecule: 637c000f776f2a98
- source canonical SMILES: CC(=O)NC(CN)CCl
- scaffold hash: 374ee134b9503f49
- task IDs: sgchem_v1.0__bundle__00049__abstain_contradiction__04, sgchem_v1.0__bundle__00049__audit_accept__02, sgchem_v1.0__bundle__00049__audit_reject__03, sgchem_v1.0__bundle__00049__construct_feasible__01, sgchem_v1.0__bundle__00049__repair_near_miss__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:7555d68e9be95012a2ca18a89732f1d442d901e3396c2b00eab1367cf83bf657, sha256:c2978736614587aec34b185189a19f58e77ae4248b90cd78fb494ca6e7300296, sha256:fd57e0eca804796a0b866e144e623db970ba6da5fb6a4cbf1f1d804c65a3c035, sha256:288a7ac1ec1dcf5e4f45376b9e14cd1639c6badfd58ae878dbe8ba74c72fc71a, sha256:4f59cc732a0621ed9d34718fdd49d8bd13e288502fff73fe8bd2cb780d2c1c0f
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00049__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500
2. contradict_hba_minimum: HBA between 12.000 and 13.000

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00049__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00049__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00049__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00049__repair_near_miss__05

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00050

- split: test
- spec_id: spec_v2_aromatic_pref_test
- source molecule: df004b6818d7a63f
- source canonical SMILES: CC(=O)NC(C)CO
- scaffold hash: ed5a74145d23df89
- task IDs: sgchem_v1.0__bundle__00050__abstain_contradiction__04, sgchem_v1.0__bundle__00050__audit_accept__02, sgchem_v1.0__bundle__00050__audit_reject__03, sgchem_v1.0__bundle__00050__construct_feasible__01, sgchem_v1.0__bundle__00050__repair_multi_violation__05, sgchem_v1.0__bundle__00050__repair_near_miss__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 4, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 2}, 'expected_actions': {'ACCEPT': 4, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 4}
- agent_visible_hashes: sha256:53d9876eb01e60013fe1b27305ee9f3f3bbe8b9a3ef5b2c7496b61c5aee4a072, sha256:69385baf0c4d002ef97e566f944145e1f54517153538e2e7b69ffe38519eab62, sha256:52a475dfa198823f1929e07e7b64c235c972ca5aebfddcc34734ac41a97f63ff, sha256:2eff27b8926c0ebf4bd7dc5f9b04bdece7353e48978223b5d44e60c17049d2f6, sha256:b13ba796726f6cd2b81e7c6da21b8fcbbf2b82bfcd59687147897e603185a63a, sha256:87710e751a3114058953eb3b13a9fc187ee6743364764307bef374d90d8b97c9
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00050__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00050__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CO

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00050__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00050__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00050__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(N)C(N)=O

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. repair_distinct_property_guard: MW between 97.148 and 137.148

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00050__repair_near_miss__06

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00051

- split: test
- spec_id: spec_v2_cns_like_test
- source molecule: 7394906d3270b9d1
- source canonical SMILES: CC(=O)Nc1c(F)cccc1F
- scaffold hash: 13cad05ca8f49c50
- task IDs: sgchem_v1.0__bundle__00051__abstain_contradiction__04, sgchem_v1.0__bundle__00051__audit_accept__02, sgchem_v1.0__bundle__00051__audit_reject__03, sgchem_v1.0__bundle__00051__boundary_precision__05, sgchem_v1.0__bundle__00051__boundary_precision__06, sgchem_v1.0__bundle__00051__construct_feasible__01
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:39f1057932d3225856609b37888f329b8ca5aa034e506511cc64075f8f51bdb0, sha256:972fa7bc23db3a719d412a6d7286615a8a7c1cd027e188e49c3a6d068323eeb4, sha256:40e5fb0e0a285ae2ab9c98982ef72d47236909d0d72a1356c3279039dd9a0a8e, sha256:df9a29a56896496aa9cd74d0ef49434f9d924088f5ef3a4bd4d5148ac02c9b6d, sha256:c5851afcf30a8e4cebba795a309c9bc1d49c823a29cc751844d1ddf3338a3c56, sha256:ced6cecd62670c2b407731eab30fc1edd90a010093e387b81bb9d13511bd0d2b
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00051__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500
3. contradict_hba_minimum: HBA between 9.000 and 10.000

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 126.146 and 216.146; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00051__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1c(F)cccc1F

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 126.146 and 216.146; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00051__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 126.146 and 216.146; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00051__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1c(F)cccc1F

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 126.146 and 216.146; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00051__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 126.146 and 216.146; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00051__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 126.146 and 216.146; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00058

- split: test
- spec_id: spec_v2_sa_strict_test
- source molecule: 2101172a8034eec6
- source canonical SMILES: CC(=O)NC(C)Cl
- scaffold hash: f7286adb4e2e4004
- task IDs: sgchem_v1.0__bundle__00058__abstain_contradiction__04, sgchem_v1.0__bundle__00058__audit_accept__02, sgchem_v1.0__bundle__00058__audit_reject__03, sgchem_v1.0__bundle__00058__construct_feasible__01, sgchem_v1.0__bundle__00058__smiles_invariance__05, sgchem_v1.0__bundle__00058__smiles_invariance__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, smiles_invariance, smiles_invariance
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, representation_invariance, representation_invariance
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 4, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'smiles_invariance': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'equivalence_certificate': 2}, 'expected_actions': {'ACCEPT': 4, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:49d3795db150fddd26766040f09192eb8f9651229c6c71fd6864374769bfd273, sha256:0c73b93bd8adb602e06281cc3ac58f52fc7d58017a47a575b73f809b1945ff1c, sha256:c9b97b538dede232b22f9216dd027c412d059e6ec6bec6500f966e895de66b2d, sha256:e9b80566b3b72d751ac2231dac63283fce0b514a15d825c09cba32fee1b3bd5a, sha256:5f6d3dde8d9b6b60f504646c7c20215a4dd18bd2401aa1ad17b46dc0ccf2340d, sha256:ab068c32dcc01f532023adb9c42bc452d31e8300a75e99ac78c9e141684f6f54
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00058__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500
3. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00058__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00058__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)F

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00058__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00058__smiles_invariance__05

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00058__smiles_invariance__06

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
N(C(C)=O)C(C)Cl

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

## sgchem_v1.0__bundle__00060

- split: test
- spec_id: spec_v2_alert_soft_train
- source molecule: 2101172a8034eec6
- source canonical SMILES: CC(=O)NC(C)Cl
- scaffold hash: f7286adb4e2e4004
- task IDs: sgchem_v1.0__bundle__00060__abstain_contradiction__04, sgchem_v1.0__bundle__00060__audit_accept__02, sgchem_v1.0__bundle__00060__audit_reject__03, sgchem_v1.0__bundle__00060__construct_feasible__01, sgchem_v1.0__bundle__00060__tool_forced_l3__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:3a084011e3c46cba3f29e746d976d9d94f55f022645ae3d80dc0821a56e8381b, sha256:e597573ec102ae9c50c623fd4f3d6a4ad43d91902a4b25b2506bb3524a9a7c95, sha256:2800fb0acb49c73a9ceca1023145d2658dfbbf18d957faf92ee3911f440f3901, sha256:0e3fbb337d2f69a9f889c76cb3d701543a24ce66178cdcfc5ca5fdc16b085097, sha256:c7832463a7882f25c08ca8ca1af9229bf7ed15b2148b7618197acd189d74cc7e
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00060__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500
2. contradict_hba_minimum: HBA between 13.000 and 14.000

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00060__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00060__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00060__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00060__tool_forced_l3__05

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00062

- split: test
- spec_id: spec_v2_amide_friendly_dev
- source molecule: d8e2500f0c65dab2
- source canonical SMILES: CC(=O)NC(CN)CF
- scaffold hash: bdede2db5ecffff1
- task IDs: sgchem_v1.0__bundle__00062__abstain_contradiction__04, sgchem_v1.0__bundle__00062__audit_accept__02, sgchem_v1.0__bundle__00062__audit_reject__03, sgchem_v1.0__bundle__00062__construct_feasible__01, sgchem_v1.0__bundle__00062__repair_multi_violation__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:4a3f3d627f82943c348e15ae64d08aa55b3c931d2c947116b6ab44d0f40dd761, sha256:7b385ec4421a671b73d404bb83601e131083053bed538ca7d0e2a3c694b43103, sha256:65148eccececa3f38fd6f007aedd4d899dde899b3d38cd1deacbe3fe540d4a84, sha256:a6d9d502fd5e9a9443f25ec7ce42d5fc6f16e4f60137201d133b8d3877f631ce, sha256:7b9c9099ec03a4208a8fc4b285f079e7987fe930ab645ac768358853ea6ee6cc
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00062__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500
2. contradict_hba_minimum: HBA between 12.000 and 13.000

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00062__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CF

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00062__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00062__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00062__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(C)Cl

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500
2. repair_distinct_property_guard: MW between 114.154 and 154.154

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00063

- split: test
- spec_id: spec_v2_aromatic_pref_test
- source molecule: 2101172a8034eec6
- source canonical SMILES: CC(=O)NC(C)Cl
- scaffold hash: f7286adb4e2e4004
- task IDs: sgchem_v1.0__bundle__00063__abstain_contradiction__04, sgchem_v1.0__bundle__00063__audit_accept__02, sgchem_v1.0__bundle__00063__audit_reject__03, sgchem_v1.0__bundle__00063__boundary_precision__05, sgchem_v1.0__bundle__00063__boundary_precision__06, sgchem_v1.0__bundle__00063__construct_feasible__01
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:cbdbbb879f075e33895a2b57fbc2509878f4bf6702985f39d80e4c004c9150c8, sha256:ac0dc394ee54b281f1d69df953028825a82622525cfb7c84f86266e43fa9b488, sha256:1a113dbe965a35cce38e4c9bf7475dd60cf0035099f37abab4e79b6bff47cb1c, sha256:210fa166e94c3ab98eecc002b3a4461a60dfcb41d532f553d5bc944932c4fe59, sha256:4a905d99fc55ae1260a9ea851096633eeeb7faa1d6dab788e3feeaf6032cc0ea, sha256:67b85f7eee7d9157f09f79fda5d735886c458aec3f83d7d46266db8605b31dc7
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00063__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00063__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00063__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00063__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)Cl

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00063__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00063__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. contextual_property_preference: MW between 76.567 and 166.567; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00066

- split: test
- spec_id: spec_v2_fragment_ultra_dev
- source molecule: 637c000f776f2a98
- source canonical SMILES: CC(=O)NC(CN)CCl
- scaffold hash: 374ee134b9503f49
- task IDs: sgchem_v1.0__bundle__00066__abstain_contradiction__04, sgchem_v1.0__bundle__00066__audit_accept__02, sgchem_v1.0__bundle__00066__audit_reject__03, sgchem_v1.0__bundle__00066__construct_feasible__01, sgchem_v1.0__bundle__00066__tool_forced_l3__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, tool_forced_l3
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:af78c6169095d73ae467ddac093a099eda2cf22855ac362896ae428024009879, sha256:39dc5876d61256f01677c602c7fdd3cc2c2f589ee154886151afda516bbadbbe, sha256:44614b80de0514af7c9fc4321f5baf57526d1c9d6dc77c9f23718924b3a7fdf0, sha256:67f276abbd567f752dbd458eb6792c1fb7425ba60c79e3d4e0acc7368b387b13, sha256:50ac438e4bd98616834890f94ff615ddb398609e240d8aaebff789d1acbbecdc
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00066__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500
2. contradict_hba_minimum: HBA between 6.000 and 7.000

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00066__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00066__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00066__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00066__tool_forced_l3__05

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00069

- split: test
- spec_id: spec_v2_ro5_balanced_test
- source molecule: aa8ef6c646b5d536
- source canonical SMILES: CC(=O)NCC(O)CF
- scaffold hash: bee2ba1666500bd3
- task IDs: sgchem_v1.0__bundle__00069__audit_accept__02, sgchem_v1.0__bundle__00069__audit_reject__03, sgchem_v1.0__bundle__00069__boundary_precision__05, sgchem_v1.0__bundle__00069__boundary_precision__06, sgchem_v1.0__bundle__00069__construct_feasible__01
- internal task types: audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'witnesses': 2}
- agent_visible_hashes: sha256:e80003274016a078c3dcdfac4b99ec3b0e7e138af068ffb1782b18ef7ffe029a, sha256:47aea37ff912b78afb630807f7c6457b8f310bc6cdd03bf26edc39f31346446d, sha256:f0cea91bb09f0bfe30de598866de0f1003ce6f456a0bc3fb5b12dc2c81f6d010, sha256:249fd6d647d9cd8685fd808d61049399a8cff43cae788a0cf5d95a637a60d82f, sha256:2ecab965a029954bf40c77ee9735651dbb25c7b4e8a32866b6f54ef1cc955e62
- possible reviewer objection: weak_boundary_pair
- manual_grade: B
- manual_notes: Acceptable paper-safe diagnostic bundle; note that this slice is interpreted with denominator limits.
- reviewer_objection: weak_boundary_pair
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00069__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(O)CF

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00069__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00069__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(O)CF

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00069__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00069__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00070

- split: test
- spec_id: spec_v2_sa_relaxed_dev
- source molecule: d10dd842b0a3abd0
- source canonical SMILES: CC(=O)NC(C)F
- scaffold hash: 127e5e4d7bb8788c
- task IDs: sgchem_v1.0__bundle__00070__abstain_contradiction__04, sgchem_v1.0__bundle__00070__audit_accept__02, sgchem_v1.0__bundle__00070__audit_reject__03, sgchem_v1.0__bundle__00070__construct_feasible__01, sgchem_v1.0__bundle__00070__smiles_invariance__05, sgchem_v1.0__bundle__00070__smiles_invariance__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, smiles_invariance, smiles_invariance
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, representation_invariance, representation_invariance
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 4, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'smiles_invariance': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'equivalence_certificate': 2}, 'expected_actions': {'ACCEPT': 4, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:53a389e98272ee046b789ff48b5839e3d8205ef2e2dc025a87644164de989b20, sha256:0e226ce4b2f17b4acf99e9f444b4858aa7682005b7e20521de1945ed3f81d2d6, sha256:50d1da165ccf8e52bf5c7663d22a1b30f2be2d59ca9eecd24fb8c8122c98c283, sha256:0de448a5f7d79cef27f63499619c9ffc6c0aa0fba565ea3c4c23a725ce07881f, sha256:ea4baf42325b5e553dffd2cb1c7f589728aa2211503ecb81bdc840af61205a2b, sha256:1c190ac0456fa7972adb6d86dfbe27afaff8c871af09000f0df5b732f9fa2f44
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00070__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000
3. contradict_hba_minimum: HBA between 12.000 and 13.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00070__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)F

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00070__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00070__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00070__smiles_invariance__05

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)F

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00070__smiles_invariance__06

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
N(C(C)=O)C(C)F

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 60.112 and 150.112; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

## sgchem_v1.0__bundle__00075

- split: test
- spec_id: spec_v2_amide_friendly_dev
- source molecule: a70486feab53d709
- source canonical SMILES: CC(=O)NC(CO)CCl
- scaffold hash: 174e3eefd8ea1767
- task IDs: sgchem_v1.0__bundle__00075__abstain_contradiction__04, sgchem_v1.0__bundle__00075__audit_accept__02, sgchem_v1.0__bundle__00075__audit_reject__03, sgchem_v1.0__bundle__00075__boundary_precision__05, sgchem_v1.0__bundle__00075__boundary_precision__06, sgchem_v1.0__bundle__00075__construct_feasible__01
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:f35c5475ec186c8476e7f79f77c1ac465c0ea9b2cfe264e8fa6ca328ad1ceff5, sha256:1114782b5f9e823416da7133c96b4a8ce0f627facab4cb185c9b6106390b8d89, sha256:691f3a44698876a92a68609f52d5e8ed64bca726b2478873c7656ae5d14d6d26, sha256:f60ba843ad0656e89ebaf65158abfe0f8e17c241232f8b001ad2b578466c45a3, sha256:3f0485018e38d1b664b71cc1cc7d89f4722b3aa91404841fc30707a47b57c613, sha256:65f8aa78e9e93f71a2ebb146e1f8bfcb23a6bc87ddeffc09fe8bbb3c488ecfb9
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00075__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500
2. contradict_hba_minimum: HBA between 12.000 and 13.000

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 106.593 and 196.593; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00075__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CO)CCl

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 106.593 and 196.593; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00075__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 106.593 and 196.593; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00075__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CO)CCl

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 106.593 and 196.593; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00075__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 106.593 and 196.593; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00075__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 106.593 and 196.593; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00085

- split: test
- spec_id: spec_v1_basic
- source molecule: 7b4ea3caa44e9b74
- source canonical SMILES: CC(=O)Nc1c(F)cccc1Cl
- scaffold hash: 13cad05ca8f49c50
- task IDs: sgchem_v1.0__bundle__00085__abstain_contradiction__04, sgchem_v1.0__bundle__00085__audit_accept__02, sgchem_v1.0__bundle__00085__audit_reject__03, sgchem_v1.0__bundle__00085__construct_feasible__01, sgchem_v1.0__bundle__00085__repair_near_miss__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:92d62a481d7d6f49d273c59fb59b6fab18d2c6f294bea366fd0da5baae0220ee, sha256:d9f0ec6e170a2f0b1c4f6990613dd48fd30e1138a4fe8e742a3aa2aa0be0043d, sha256:cb1e94c70738bb73cefb8843665e6501b05289563f017f75b8f4aa6f58da0b7a, sha256:039c25fb04645c5b4c1fd6ca630ac5063b3ab514a6c651a111ef8b3fae184543, sha256:0de818fb52312749be7aa382681f34cd361254e1a6b8511d17ec8cec633a8478
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00085__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000
2. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00085__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1c(F)cccc1Cl

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00085__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00085__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00085__repair_near_miss__05

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)C(C)N

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between 0.000 and 5.000

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. contextual_property_preference: MW between 142.601 and 232.601; TPSA between 0.000 and 64.100 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00092

- split: test
- spec_id: spec_v2_fragment_ultra_dev
- source molecule: f8434d060e2093f4
- source canonical SMILES: CC(=O)NCC(C)O
- scaffold hash: 114e93e0ac2bca09
- task IDs: sgchem_v1.0__bundle__00092__audit_accept__02, sgchem_v1.0__bundle__00092__audit_reject__03, sgchem_v1.0__bundle__00092__construct_feasible__01, sgchem_v1.0__bundle__00092__repair_multi_violation__05
- internal task types: audit_accept, audit_reject, construct_feasible, repair_multi_violation
- visible task names: candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'repair_multi_violation': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'witnesses': 3}
- agent_visible_hashes: sha256:cbf2ed0ab71d10be6c7edaaa21b47697af718e1fabee964f3e03af06cc87195b, sha256:59b11e978e73a630d6b7fc26b141b3e8b02e3242052b948d32f9bad9ff5d7d24, sha256:1266d36270d48ba3240d1c77f891f908bc7b3077b3f7f24b9bdea14a8b8e4dd0, sha256:1b2c653654db4c86ef8742a2f56b466fd7480c01c3889562a041574a3cfa8dd7
- possible reviewer objection: too_template_like
- manual_grade: B
- manual_notes: Acceptable paper-safe bundle; primary limitation is too_template_like.
- reviewer_objection: too_template_like
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00092__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(C)O

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00092__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00092__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00092__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500
2. repair_distinct_property_guard: TPSA between 34.330 and 64.330

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 72.148 and 162.148; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00093

- split: test
- spec_id: spec_v2_low_rotor_train
- source molecule: 3c0f286881a44cbd
- source canonical SMILES: CC(=O)NCC(N)CCl
- scaffold hash: f7b097e0392c9e34
- task IDs: sgchem_v1.0__bundle__00093__audit_accept__02, sgchem_v1.0__bundle__00093__audit_reject__03, sgchem_v1.0__bundle__00093__boundary_precision__05, sgchem_v1.0__bundle__00093__boundary_precision__06, sgchem_v1.0__bundle__00093__construct_feasible__01
- internal task types: audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'witnesses': 2}
- agent_visible_hashes: sha256:8e338d952379e3674bd934308059d29c809d5b308585fa9e83fd3badba211001, sha256:256887a688792db0b58852577201c3ab7f53d292be88de9b2247927465ec6a4a, sha256:70e00a46cb872a4fa06e18857388223c3d63b63ddcd5429018342494f66949e6, sha256:53a9cff60659f3d13baac937531b262013fe6833d084bf4f0765d2890cf530c7, sha256:8499f0c467b5aef8181e9fb2b84966581ecf894090d35d1d0723f7da7b1f8046
- possible reviewer objection: weak_boundary_pair
- manual_grade: B
- manual_notes: Acceptable paper-safe diagnostic bundle; note that this slice is interpreted with denominator limits.
- reviewer_objection: weak_boundary_pair
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00093__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(N)CCl

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00093__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00093__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(N)CCl

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00093__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00093__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. low_rotor_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 480.000; ROTB between 0.000 and 4.000; logP between -1.000 and 4.500

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00095

- split: test
- spec_id: spec_v2_ro5_balanced_test
- source molecule: c5ac454de96fcc34
- source canonical SMILES: CC(=O)Nc1c(Cl)cccc1C(N)=O
- scaffold hash: 13cad05ca8f49c50
- task IDs: sgchem_v1.0__bundle__00095__abstain_contradiction__04, sgchem_v1.0__bundle__00095__audit_accept__02, sgchem_v1.0__bundle__00095__audit_reject__03, sgchem_v1.0__bundle__00095__construct_feasible__01, sgchem_v1.0__bundle__00095__interrupt_resume__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, interrupt_resume
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'interrupt_resume': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'interrupt_certificate': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:433850d5b32488fd87bac75dfac5142f9f6b3adeb03ce2c6a75f5d4d47bec754, sha256:f4a77c7faf59e0d14b4845ff93d5ad03479df27e633850979534aef86dab9dd9, sha256:a685f1ef0366153ff7c9e59251cff70f7658447ddd4ecea75eaa03749f10a725, sha256:c1db52a7f728e70c614788e58b7378a8a27cda02cb3c0481c8361467cf63e138, sha256:46cfed0abd2eb426fd719c667e6321e9d6c61e9f8322d9420401b61bcae4eef3
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00095__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK
3. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00095__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1c(Cl)cccc1C(N)=O

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00095__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00095__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00095__interrupt_resume__05

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00101

- split: test
- spec_id: spec_v2_amide_friendly_dev
- source molecule: 3c0f286881a44cbd
- source canonical SMILES: CC(=O)NCC(N)CCl
- scaffold hash: f7b097e0392c9e34
- task IDs: sgchem_v1.0__bundle__00101__audit_accept__02, sgchem_v1.0__bundle__00101__audit_reject__03, sgchem_v1.0__bundle__00101__construct_feasible__01, sgchem_v1.0__bundle__00101__interrupt_resume__05
- internal task types: audit_accept, audit_reject, construct_feasible, interrupt_resume
- visible task names: candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'interrupt_resume': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'interrupt_certificate': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'witnesses': 3}
- agent_visible_hashes: sha256:7f3538f1c9fa9cdd54fc235f1eb79b9ff2a5a82f3f81246ba8f5517d55daed10, sha256:d514e7e6b748336147942177bbf451eef28ebb7d2e2bad0b63489b174e7697b1, sha256:d897affe6c711e4118af1f99a0e8a8dd35b576e9038d1c13e0746768abedb57d, sha256:8fa69dc2c25d84ab629adcc1ba341786047d07a1a526be1ef8bbcb1980f5126e
- possible reviewer objection: too_template_like
- manual_grade: B
- manual_notes: Acceptable paper-safe diagnostic bundle; note that this slice is interpreted with denominator limits.
- reviewer_objection: too_template_like
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00101__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(N)CCl

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00101__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00101__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00101__interrupt_resume__05

- internal task type: interrupt_resume
- visible task name: repair

```text
Task: Medicinal-chemistry interrupted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00103

- split: test
- spec_id: spec_v2_cns_like_test
- source molecule: ed12cfe1d91f1962
- source canonical SMILES: CC(=O)Nc1cc(Cl)cc(C(N)=O)c1
- scaffold hash: 13cad05ca8f49c50
- task IDs: sgchem_v1.0__bundle__00103__audit_accept__02, sgchem_v1.0__bundle__00103__audit_reject__03, sgchem_v1.0__bundle__00103__construct_feasible__01, sgchem_v1.0__bundle__00103__repair_near_miss__05
- internal task types: audit_accept, audit_reject, construct_feasible, repair_near_miss
- visible task names: candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'witnesses': 3}
- agent_visible_hashes: sha256:d8a9c6179d1e862d153e19dd275e1fc90bc1e9c911c269ccc882856ee0b782b9, sha256:99692505f1ef020a2df233c7bd7c44820b67cb2a33f1b19fdd835874dc7fab35, sha256:5e3527e90ed4a810ee894c54ce7c5a41eaf75540acfaa3c438150d620810788a, sha256:af467c9dfcc0b51a324acc9aed811e14a4752a2bcbb1909f4bde9593acb0fd4b
- possible reviewer objection: too_template_like
- manual_grade: B
- manual_notes: Acceptable paper-safe bundle; primary limitation is too_template_like.
- reviewer_objection: too_template_like
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00103__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)Nc1cc(Cl)cc(C(N)=O)c1

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00103__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00103__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00103__repair_near_miss__05

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CCl

Hard constraints:
1. cns_bounds: HBA between 0.000 and 8.000; HBD between 0.000 and 2.000; MW between 150.000 and 420.000; ROTB between 0.000 and 8.000; TPSA between 20.000 and 90.000; logP between 1.000 and 4.000
2. sa_limit: SA proxy maximum 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. contextual_property_preference: MW between 167.636 and 257.636; TPSA between 37.190 and 107.190 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00105

- split: test
- spec_id: spec_v2_fragment_ultra_dev
- source molecule: 3c0f286881a44cbd
- source canonical SMILES: CC(=O)NCC(N)CCl
- scaffold hash: f7b097e0392c9e34
- task IDs: sgchem_v1.0__bundle__00105__audit_accept__02, sgchem_v1.0__bundle__00105__boundary_precision__05, sgchem_v1.0__bundle__00105__boundary_precision__06
- internal task types: audit_accept, boundary_precision, boundary_precision
- visible task names: candidate_audit, boundary_audit, boundary_audit
- expected action distribution: {'ACCEPT': 2, 'REJECT': 1}
- oracle summary: {'task_types': {'audit_accept': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 2, 'REJECT': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'witnesses': 1}
- agent_visible_hashes: sha256:32d46cf6e3ef2d6e15c4dab896b0c48adc887ba8dbd9e7abf8576d61c5fe68c0, sha256:f11ed8bde5d2efc20eb80da5bce88a1f7690b2a294ddb117cff27071a2c2d3b1, sha256:a4095147ab360b86e26aede34e7ab52ad70da343b24569a49e37f493c08dad52
- possible reviewer objection: weak_boundary_pair
- manual_grade: B
- manual_notes: Acceptable paper-safe diagnostic bundle; note that this slice is interpreted with denominator limits.
- reviewer_objection: weak_boundary_pair
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00105__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(N)CCl

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00105__boundary_precision__05

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(N)CCl

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00105__boundary_precision__06

- internal task type: boundary_precision
- visible task name: boundary_audit

```text
Task: Medicinal-chemistry boundary precision audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. fragment_ultra_bounds: HBA between 0.000 and 5.000; HBD between 0.000 and 2.000; MW between 90.000 and 220.000; ROTB between 0.000 and 4.000; logP between -0.500 and 2.500

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. contextual_property_preference: MW between 105.609 and 195.609; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

## sgchem_v1.0__bundle__00109

- split: test
- spec_id: spec_v2_sa_relaxed_dev
- source molecule: d8e2500f0c65dab2
- source canonical SMILES: CC(=O)NC(CN)CF
- scaffold hash: bdede2db5ecffff1
- task IDs: sgchem_v1.0__bundle__00109__abstain_contradiction__04, sgchem_v1.0__bundle__00109__audit_accept__02, sgchem_v1.0__bundle__00109__audit_reject__03, sgchem_v1.0__bundle__00109__construct_feasible__01, sgchem_v1.0__bundle__00109__repair_near_miss__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:ca030b1a131cccfdb412126a7aa0dc3131bfacf940560cb668772a083f1a0d86, sha256:8478bcfa9080865460d6ca691b81230910b0ea805f93f135b9f8f744aad6f371, sha256:5368b3b413bba1fec18822583a7bd2fe0f544cd67807e8817f1a10a4c2bad1cd, sha256:f86a872bf72ba08e8d146dd7a03b5398fbd72dfe9d9764d3bfe29719d62e5a4e, sha256:855e4e3bdd1cb363a393b56174eb381f44dd5b35d46c35db9651d3f4cde6b06d
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00109__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000
3. contradict_hba_minimum: HBA between 12.000 and 13.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00109__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CF

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00109__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00109__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00109__repair_near_miss__05

- internal task type: repair_near_miss
- visible task name: repair

```text
Task: Medicinal-chemistry near-miss repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00110

- split: test
- spec_id: spec_v2_sa_strict_test
- source molecule: 241b8eafc9e7b2f4
- source canonical SMILES: CC(=O)NC(CO)CF
- scaffold hash: e57f87ac3bce842a
- task IDs: sgchem_v1.0__bundle__00110__abstain_contradiction__04, sgchem_v1.0__bundle__00110__audit_accept__02, sgchem_v1.0__bundle__00110__audit_reject__03, sgchem_v1.0__bundle__00110__construct_feasible__01, sgchem_v1.0__bundle__00110__repair_multi_violation__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:5e16a39463b90c6ff77422771c846dc1b211eb78a6c3777902436f3bfc3bde51, sha256:30eb200c874cf80a38b684c2592f9c5926ff7673aef518e274e2322348b9e073, sha256:be753f5d25abe5538d173edbd18d0b9e9095573c0fe1a514697d5d3c87074038, sha256:19e1025e0bdb8aa76e54d42d425163554f31b9420b53de2dfed8d425a07f73bb, sha256:fa2b88d40ea980458a303d234d612c6a7c19ba38d7a2f4b7231e1df7f49a6dd1
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00110__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500
3. contradict_hba_minimum: HBA between 11.000 and 12.000

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00110__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CO)CF

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00110__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)F

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00110__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00110__repair_multi_violation__05

- internal task type: repair_multi_violation
- visible task name: repair

```text
Task: Medicinal-chemistry multi-violation repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(N)C(N)=O

Hard constraints:
1. sa_strict_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 110.000 and 500.000; ROTB between 0.000 and 9.000; logP between -1.000 and 5.000
2. sa_cap_strict: SA proxy maximum 4.500
3. repair_distinct_property_guard: MW between 115.138 and 155.138

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. contextual_property_preference: MW between 90.138 and 180.138; TPSA between 14.330 and 84.330 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00112

- split: test
- spec_id: spec_v2_alert_soft_train
- source molecule: d8e2500f0c65dab2
- source canonical SMILES: CC(=O)NC(CN)CF
- scaffold hash: bdede2db5ecffff1
- task IDs: sgchem_v1.0__bundle__00112__abstain_contradiction__04, sgchem_v1.0__bundle__00112__audit_accept__02, sgchem_v1.0__bundle__00112__audit_reject__03, sgchem_v1.0__bundle__00112__construct_feasible__01, sgchem_v1.0__bundle__00112__smiles_invariance__05, sgchem_v1.0__bundle__00112__smiles_invariance__06
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, smiles_invariance, smiles_invariance
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, representation_invariance, representation_invariance
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 4, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'smiles_invariance': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'equivalence_certificate': 2}, 'expected_actions': {'ACCEPT': 4, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'equivalence_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:fdf7051343d658af068a8cab9d65eb1b380c6bd5b7f1bb1d881b85e367eb7aba, sha256:c0e99b5141adb774e1e9a19afac3f2481d958a0aa5f51472371f030fe429d3b1, sha256:7d97ffa7130bac27ae22a3e9c56e68565e28c6ce46e811d07b4bd8ef6a1a164d, sha256:45ae5d831ce2134b4ea75505b652802b7c5ba3fb75857ce74b6afc4b94a57aa3, sha256:d22339643fc7f58b81ce911d74a9ca82d650bc2325ec344a6fd3d5e77fe9dbc7, sha256:76b876b916d502127acddb8b402377ff744a95bcf5abcc58c49451911aac9f7c
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00112__abstain_contradiction__04

- internal task type: abstain_contradiction
- visible task name: feasibility_check

```text
Task: Medicinal-chemistry contradiction handling.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the stated hard constraints can all be satisfied together.

Allowed actions:
- ACCEPT only if a molecule can satisfy every hard constraint.
- ABSTAIN if the hard constraints are internally contradictory.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500
2. contradict_hba_minimum: HBA between 13.000 and 14.000

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00112__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CF

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00112__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CN

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00112__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00112__smiles_invariance__05

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(CN)CF

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00112__smiles_invariance__06

- internal task type: smiles_invariance
- visible task name: representation_invariance

```text
Task: Medicinal-chemistry SMILES invariance audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
C(NC(C)=O)(CN)CF

Hard constraints:
1. broad_druglike_bounds: HBA between 0.000 and 12.000; HBD between 0.000 and 6.000; MW between 100.000 and 550.000; ROTB between 0.000 and 12.000; logP between -1.000 and 5.500

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

## sgchem_v1.0__bundle__00114

- split: test
- spec_id: spec_v2_amide_friendly_dev
- source molecule: 64653775453dfb20
- source canonical SMILES: CC(=O)NCC(N)CF
- scaffold hash: 849544041ac73707
- task IDs: sgchem_v1.0__bundle__00114__audit_accept__02, sgchem_v1.0__bundle__00114__audit_reject__03, sgchem_v1.0__bundle__00114__construct_feasible__01, sgchem_v1.0__bundle__00114__tool_forced_l3__05
- internal task types: audit_accept, audit_reject, construct_feasible, tool_forced_l3
- visible task names: candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'tool_forced_l3': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'witnesses': 3}
- agent_visible_hashes: sha256:9176967fef22c36fe2c8931d839411cf7f0025d8469c85e225f89c6241e63e48, sha256:1e85963d1e32b9829dd1d5f96f52aa8a32298d22a42be43846ad3978a38d78d0, sha256:787d70710af0b46fd61b9b5da4cd01311ea49169aec9b0fe7d3fe6085d52d6ee, sha256:2d36f5cf7cb59620dacb3433e4ee65a9490f6ed7c0e51e8825b7d95bc4f3ebd0
- possible reviewer objection: too_template_like
- manual_grade: B
- manual_notes: Acceptable paper-safe diagnostic bundle; note that this slice is interpreted with denominator limits.
- reviewer_objection: too_template_like
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00114__audit_accept__02

- internal task type: audit_accept
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NCC(N)CF

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00114__audit_reject__03

- internal task type: audit_reject
- visible task name: candidate_audit

```text
Task: Medicinal-chemistry specification audit.

You are given a machine-checkable medicinal-chemistry specification.
Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT if the provided molecule satisfies every hard constraint.
- REJECT if the provided molecule violates any hard constraint.

Candidate SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.
```

### sgchem_v1.0__bundle__00114__construct_feasible__01

- internal task type: construct_feasible
- visible task name: construct

```text
Task: Medicinal-chemistry molecule construction.

You are given a machine-checkable medicinal-chemistry specification.
Propose a valid molecule satisfying the stated medicinal-chemistry specification.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00114__tool_forced_l3__05

- internal task type: tool_forced_l3
- visible task name: repair

```text
Task: Medicinal-chemistry tool-assisted repair.

You are given a machine-checkable medicinal-chemistry specification.
Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.

Allowed actions:
- ACCEPT by returning a molecule that satisfies every hard constraint.
- ABSTAIN if the task cannot be completed under the stated rules.

Input molecule SMILES:
CC(=O)NC(C)CN

Hard constraints:
1. amide_friendly_bounds: HBA between 1.000 and 11.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 11.000; logP between -1.000 and 4.500

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. contextual_property_preference: MW between 89.154 and 179.154; TPSA between 20.120 and 90.120 (weight 0.050)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```
