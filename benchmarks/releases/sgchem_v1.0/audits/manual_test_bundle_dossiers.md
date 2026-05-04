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
- agent_visible_hashes: sha256:436706fbb1e0eb51d7126e2d8d4ab77bb5d0f93339a1c25ce7b707d0c8ef6f2c, sha256:805a6c80b83ebd8d176d5b9f93d2acd719e70c8fbadaf33a1102bfb393592bf5, sha256:f70b75285df98547ac7ccacf18a8598cd415b98204612b71c5a732fc34fe6705, sha256:1e409abd6762d41063dbfb2403f50b004798bef461201ee47da2dc96f4a10007, sha256:37ed7a9f22bf85d3d059e8649f01abab00849ab4e3f7affa1f749d3d1926d763, sha256:d2e880ac534f0816df334e959ed5daca4a5192732178fc2eab1d652e7de52330, sha256:d6af34936186058b949d873ab04acbd944dc8498efaadc01090ea56e902f838c, sha256:822ce51896cfce022edc77f7005836adda37c1974521c8dcf8d09e782b658b5d, sha256:853845c72f71f3fd0c26b856c5b1be2e21b2dbfaa83d0258515827a6f04f983f, sha256:f7e2da6550ed37c33c95f31dff5d990ba89950e6149e7f3629668644f2ac57ee, sha256:cb61b331bcc36ff3ac1eb5d47297f91f94ac803d7c3266b0685e6fdd0c2b6262, sha256:025b1acfd7a6097e44d57fe33bb050ed74da45803b2691d2c92d1adf587b926d
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 120.000 (weight 0.500)
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_74e7e615b7be81e7: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:3f4c100ec41bf598f033921f8c51c64a942198bbbb15779d5810f7932bbb0ff0, sha256:b0ebac0ee4ebfd4d6c8440446b7260c61e96b64e210b786146a6a79dcaf801ea, sha256:163b8bdabc75f7e0e81854a8d9a681bb9c10f2aa85a10d8131ed9b32be46c2b1, sha256:73ded3e3856fb0c5c12b075e885bd3989fc7328d3b7658170ead285ff52d6889, sha256:b98bb2f5605e1a9dcbede998857a94acef3fbd1a56956985cefee7659c4280eb, sha256:2d6b8742a502c5eec2f275bec08429ea691a294165a6ca5dae1bf87361908552, sha256:16c14c4d4ae89e7cfb25e5063d4d7b3d1c0f93489bd6c2b61a00ee971faed241, sha256:49994e633c2541c72d4a5865ecef404ad2c34ccc13c2792570254431d85a69df, sha256:33e3112d40d1faa09bc1098c0e41e543c3b1353e0f325a0f7a49193d5ee909a6, sha256:f0cd12f6574ec62d46ddaa39585f36e4afc2adf2f239c74bbbf60d58ea64964e, sha256:da2be609ed13c9f8ef3d188eac0727bda0f2bdd84c94cb4aeb05b702445fc3e3, sha256:d74bd6e336e49117855155f0f895dee30c1de02e796037e7b591c5cd52c13094
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_04749df0594f2d34: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:73cb7732344115241daaa668c5bf75a5f240d14def60c93c7f658ba5873b0e83, sha256:38da4ccc9073fcb400a82350e20f73b7881b6b285a0a94046de4e44ea934cda1, sha256:4b4eafdf961730a7537e2c33cfe4538fd61236b66597081e0143a9e99d00c472, sha256:936d52d104d742011a6dac0ee9b311be4e04c7eff0d68533bb5ab5e806ac5630, sha256:e2ce3a0d7d4b3e461f8d549d791ef038039f90cd351db3887acac28a8180f542, sha256:a3ee065b198ad196375a2450e46199f0ace05af52fa5f0a1eab2060168952bed, sha256:567024ff29129c74659145e67d8b1d5a6c419ba6e202a328c04909888af20f29, sha256:9bb11fd3fc9b9a2df2368dbaaac102916eecd0ec94a0b189132025c2a68973b3, sha256:70247267dc90252d84e131619f48763392ee062d2aa20ca0c82f6bd5e6d1076d, sha256:4fced2ac8bb31de2c8863f0098bf3e83f5dd3f28843fa9f4966b48ad64e4bbef
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_23678065f52d8c95: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:a1c8f7c24c9391b1729560caee2180b34b251a670a9f46ecdd35c74995f69e84, sha256:f809ee22a5eb6dbd3cf00f1125ef13b5deb4718d2ba45f17525fbeee31bcb1a0, sha256:f12acb030749a7697330b696c8f6abeb88daa4124032ed657bc82c7ca02c4bec, sha256:e9f359a4dbb6fb5fe53f25590725b3d21b0c0b04a4485dbae86f01a9116b44d8, sha256:fa6f63372005c390778ef329328edc8dc96ea67c2248fed652ff3303f2849e07, sha256:a43d65ea1c028b7df60485ef8a9febd4163aad4115899475c1b7f259f997acad, sha256:f8ec314ff541792f19f555fe809bac95ad15f2144cd78f0a248a77973f1f2941, sha256:9410b6cf0932d8cd0061d5e4fa134cbaca77343dbcb2cc01b8a4bb660b50564f, sha256:c40229778a184726f99aab0938eefcce2bc63e74795f57f78374785c32dbcaf9, sha256:6e53d93989e4792a15e269c7c70a24c88f46d8b28145eaf28e2555976433470c
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_c36bc29814d4adb9: MW between 130.190 and 130.192 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:4b21cccdfc9909ea93772da621a5d548585e8085b40dd73b632d5eed9a026338, sha256:d651bc65fee66e0f2f56479b73f4f30768a67bf3177d2a98b5d1fa3f12df1578, sha256:50cf77cc993810e5bf4335771a514503497b3d0af5f54f1529489e486655729b, sha256:24fdbcc1c502c6c89b4709a322c9322c41c64fe3c758b77550f8455ebb8ade08, sha256:25dc44e46f64395f52cab474c84490be8f227e959719cd1cdf676fe8c659c17b, sha256:50c1f0a613e55fda1d8a3e59ece0cc97dd0794de23d9eaaca5005e055f842882, sha256:8237b6c126ab9e60b3446144e42198f9a3f26e34fa4c273a0776defd99b2190a, sha256:c4cab185ce754d4f4148fd77f7f6acacd47af4e4c715717fada518caf6678159, sha256:f88134eb4df0d849ddd8466c63a9ec1ed5f21fc963acaf217f9ad5441ee10968, sha256:34c978c179b8390cc455fe12c4f9155971e063435259b2c24074b3656446bb96
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. amide_soft_preference: substructure present: amide (weight 0.800)
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c5b08741931082a5: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:61bce1e04a01266e7edb5807c5acbccc5469597042669033b954b4286eaf838d, sha256:3eba75c4b35b5b49f6c7e109a52a35a9bd32bdaee290dfcc8bd9c1f6c28a7ed5, sha256:dea948945a14af4d923d517c0151847adf3a99b828470ceae2448ef32e1e7f3c, sha256:466f97a7f617b167fa2043621246f4e25155ff3f37e280b67f3171b346a70ece, sha256:e2706d7e39aa721cd7dccb30d7dc3e6d4b5ea48c3fb63b9951b47ebefbd2dcd5, sha256:a28a626a70e2dc3cc787190dfae73abddd4e429bf338c0213983643d21745399, sha256:3dfe2f83bc937c090310b5cf520fa08367d10a14b4be423f22a09efde68a2b80, sha256:1ce11c1e84a6c4c77225613a2e9ac039ea3a6405d9c24afb1c201d14dc188b37
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
2. instance_soft_window_ee86f216c1544e58: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_ee86f216c1544e58: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_ee86f216c1544e58: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_ee86f216c1544e58: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_ee86f216c1544e58: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. instance_soft_window_ee86f216c1544e58: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_ee86f216c1544e58: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_ee86f216c1544e58: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:7166e21fcd441a04b049c99fc3570dfbdb00001cb110f5c8d8620fe48e69e9e6, sha256:b91a5954b488c808711127a139a1e33a4f121c138584a2909c8cc402230e9274, sha256:fb07ab4448e4ea5a3a91de1c3b9dd1d77db9e36eb2f7aa9ca3bcfcfb7bea8e28, sha256:0fbee4b91d730442cf39e28dc56b3936c367e4f8a5cf32b15c63c3d82ccdd402, sha256:eddb2d37451d3c14f0f56d0cb20b82d5d98fcb17cb7b0325230b38b860064ff3, sha256:1b90c4bb0ef40186a09e3ef28cfedbe437e4bb5d6a5ae9f702bfaa19915dde34, sha256:6051783052fefc4613bc55d51daa64ffc61623a76c18c0a3e61d76e9a595b5d1, sha256:4c1903ca9eb5cca4875c30dc906839e7e3121d8d537493881e3dda54b057d050
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
3. instance_soft_window_3a22a1e32a091d07: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_3a22a1e32a091d07: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_3a22a1e32a091d07: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_3a22a1e32a091d07: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_3a22a1e32a091d07: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. pains_block: alert set absent: PAINS_A (weight 1.000)
2. tpsa_pref: TPSA between 20.000 and 120.000 (weight 0.500)
3. instance_soft_window_3a22a1e32a091d07: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_3a22a1e32a091d07: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_3a22a1e32a091d07: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:fe762ee0d4c42b38ed3d7bc0db2f8a7d83ae6adb00ad70f76b3e9c818939a509, sha256:46f61ff362b9dd7d6bf97b8bd53f09a542625103747164fe425a569601baf483, sha256:4ecfc85ed71de713e73f9e08940d5873389bda4508cd591aac66f6672aa073b0, sha256:e8db41d3b8c6fcf28885752382ff2f818f2bea279de79dc13147bf9c3cf3bcfc, sha256:799fe573a6ac77398cc815ba688db620b8b3b2a081317946e1ba5267609e6c88, sha256:aa5bce6452c6917b015459c032c1d11c579c61d98167912ae96cd376d854ab98, sha256:c91ce3d308177648c6cbc7cb7e0a5d1c7418c5d539492bea2d4f9e0589292621, sha256:3afd32ba6c6963f40fa0c59fb4cfa82f3015ccbf7837572d75a32441dcda3662, sha256:937687ebdf47672cc6dd7d4bec16b96b3b2e77b21ddda4923d2071cda3dff71a
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
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 1.000)
2. pains_b_soft: alert set absent: PAINS_B (weight 1.000)
3. pains_c_soft: alert set absent: PAINS_C (weight 1.000)
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_440d2d60b8b499ac: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:cce7f9e1dc847c22202173e87eb9c94ec7bf5e839371d4b60f7ce656d159c23f, sha256:af54b48ddbf7f7ea583cb0e09dd91ed5412ecb12f179be5dc475310929ff8033, sha256:ecbed41f50a70c581e2740303d00e92601094e326fd41a925ab49a99b57199a5, sha256:ea9b10e743a8d25ba8ce4d677a5ccc4392480d1a7a5350c4867c87fd825304be, sha256:1f3144be7be9b5a835282d765f801f98a535c2e563e7211c7bc83682e6466fb7, sha256:3126b54eeb19d583ede5a65d5aa20b9e58ebdb4d2ace42c3824fab352bd25ff1, sha256:1dcce44c30506c2163fa21dbb66f08872f91360175e07d37990d9a1f59718fa1, sha256:4876cf629299777750150c44106ecc9edf873716324808d2dc3578ea101ae505
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
2. instance_soft_window_f353ae4e6c186b07: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_f353ae4e6c186b07: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_f353ae4e6c186b07: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_f353ae4e6c186b07: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. tpsa_soft_window: TPSA between 25.000 and 140.000 (weight 0.500)
2. instance_soft_window_f353ae4e6c186b07: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_f353ae4e6c186b07: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_f353ae4e6c186b07: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_f353ae4e6c186b07: MW between 131.174 and 131.176 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
- agent_visible_hashes: sha256:8415a44eb991e4347e3f55803933c4a07202b3dac1e62ac3b91420583d9c1aa0, sha256:1f41ad28b12be38b0f1e4ad5385c6b4971555a395164558493f5875198abbd89, sha256:c5b7139ad94a4c748fd0f710422a2ea45a661252ac8a61ad06184ff77c37c4ea, sha256:aeb671c29781d098da80d94f2ae3e532abc44f35459e2bed4d4be2686070141a, sha256:cc0ccb112bb245321551929361e272da07e96e1bf089f2f9258ba18ab4d4c88c, sha256:5a9944d62ae59e699173deb9c4e07f9692ce56ca243e9fc29686662ce7a81fd1, sha256:a9ccc5c335d6fa470072633abf94435ea336a5b439f2e6ca76e888e14a39ea48
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
2. instance_soft_window_00c9cc0358b42edb: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_00c9cc0358b42edb: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_00c9cc0358b42edb: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_00c9cc0358b42edb: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. instance_soft_window_00c9cc0358b42edb: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_00c9cc0358b42edb: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_00c9cc0358b42edb: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:82eeb146a15e5e5f3e1028de7b731afedcf87bc3f7e459246d66c2e0849860cc, sha256:4f06d454c080b417c251392646972c79ec7d2c974a9cf06e6dafda0b73155a3c, sha256:af48f62ae1b6d4d0698e141b9d70f48a6416dc93a65eaf403c53f6cafecfed25, sha256:17ad77267ec1cd0fe450f962b1699671c3b0d57f9a6ec583ce8260a9749e764d, sha256:c6203b836ad9c35a1e6db187ee9388534c7144480830fabb76a1d71ef2ee9704, sha256:a156f25669ef4387d7adb722991a63966b01aecc9e0f76f329cb324c3da081c9, sha256:51c18c65591556dcb10fc031f5c7ed9ab09e14f4ba755e5c70c3746d3ea1d488
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
3. instance_soft_window_2d08adea46639b68: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_2d08adea46639b68: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_2d08adea46639b68: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_2d08adea46639b68: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. instance_soft_window_2d08adea46639b68: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_2d08adea46639b68: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_2d08adea46639b68: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:407a35a33892f32eccd7b5be7f3ae829e7da96f85ed6affd71c7dfbff71096d7, sha256:79afb60a77b491ca621b8e3c74f574f50458f4d9e91c6e0d339d962edc1344cf, sha256:e4f8bd8c6077fb721ea6fc4990820139b56d7a9db61ec76db28d3e43cf3e5c24, sha256:bfbb5de94132b96009e6456ac78c00bdbcb667cafd29b23513ce6ddba5305559, sha256:ca572ff838fbe15ed31b85fe64ad29b0034e0e33c4f8bb2fd31e1fa5262d3944, sha256:f874d50b58a13b95637ce2acbbf16a0f08d8a59d02ca8b9eb82c70fe8a7a08cc
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
2. instance_soft_window_93ea0ccb68029b4c: MW between 116.163 and 116.165 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_93ea0ccb68029b4c: MW between 116.163 and 116.165 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_93ea0ccb68029b4c: MW between 116.163 and 116.165 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_93ea0ccb68029b4c: MW between 116.163 and 116.165 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. instance_soft_window_93ea0ccb68029b4c: MW between 116.163 and 116.165 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_93ea0ccb68029b4c: MW between 116.163 and 116.165 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:76252898ebae4133864d6519ea41fc166136cf9c12099d665b05fab5612bdc2d, sha256:71366940596072909cf5cce667ce744d17ca0c453dc184520789681226a25f0d, sha256:7b3b73dbbbc377ff0958983ea5e6749935b8bba96566701f78066ffaa87f0926, sha256:e2814289dd73b20a6e4cd16578521ddd54237e1d365fc7dba0d8836b2b8d96e8, sha256:e7abb2c3ff5e4465042426a90423dfe3eee252859e1cceb68626cbbf2a453780, sha256:59dd15065f75f2e918aed03d04781346aa90b5f1ee2efe89ca8cc13c23d9bf63
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
2. instance_soft_window_ea5101a8d1030fc2: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_ea5101a8d1030fc2: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_ea5101a8d1030fc2: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_ea5101a8d1030fc2: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
2. instance_soft_window_ea5101a8d1030fc2: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_ea5101a8d1030fc2: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:fe473a7a1d49a021dd54bed69ca554a557d7c73ec1eedfa7525c9123a16f7c19, sha256:d6f410ee04f9dc9df4dc0eb38144bdaba8378e35346f9d48efbe5fb8c1141639, sha256:9e9df2eabc9af0045b1c4da42a8155f83c63a2502946ef077d5cf0648cd40406, sha256:7064f132a44c61256eb21e95d8e5d47c122aaf61acc79a6681155576b912baf8, sha256:c9f3859d7abf56a7a4f6c244d99be8e3157bb847e5714caa5076ac6bf7d470b8, sha256:02022a97802e5367bc6af07a6377f26fec0ab9f65a48471d11870c0793e02f44, sha256:a4d03b849dc8305f8664198c5f9fe6bca700ce68028146cee3b55aa5a03a75e0, sha256:ab838f9f6bce4353e5e96b212deb6bb9359e77c3d4884225a29bd5b3575d7f9e
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
2. instance_soft_window_11704d15417f76a5: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_11704d15417f76a5: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_11704d15417f76a5: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_11704d15417f76a5: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. instance_soft_window_11704d15417f76a5: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_11704d15417f76a5: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_11704d15417f76a5: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_11704d15417f76a5: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:4f16c378ac53cb8a3f4fbcf00a9f66ddce46bfcfed1651ab34f1762af31007d4, sha256:1056a875cc88e5785a14172cb9fb2a006e841ecdc4bdab43f771c906b262a379, sha256:8ec1c7ebd63872ac95081240c6d7b42908f3ee7d602a958e6989993d6eda24a3, sha256:263a059007d2a3b410a7468828826cbd5301e47d5e1bd319b28cbcafd6a5daa9, sha256:62787485bb8eee54e1f8093577f132f5a83fcd39b79ea9e13f6bfad4d21257ca, sha256:55979bb6abaf3d10b5db6f6935a4903abebb49f4b3d9021317c446293c6c3315
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
2. instance_soft_window_acbea93d40d47f30: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_acbea93d40d47f30: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_acbea93d40d47f30: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_acbea93d40d47f30: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
2. instance_soft_window_acbea93d40d47f30: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_acbea93d40d47f30: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:694786d00e8456c6b5aed55a07be96856a89b93b7af87b6a4ca92e28b46c92fe, sha256:fc6abca4b5ee40c2ff83fa93b1c254124c4087c8481740ec3f73a635f2acba5d, sha256:b53835ad892297e5b0bb14bcee88e3d98b5a5c95460b1b969950727550374d85, sha256:72ac04c4b356e9c7c3b1e3436dad1e7f53b04595cc70f196778c155b60b4058c, sha256:97d8a556b14deb6f363bbeb7f216a07d23864f496a3306dc8031c7f50233c6e0, sha256:369e22cf4cf3c09ccb96ff448b199fb1a6b71a1157519c7d79af6d8b225498fb, sha256:4db5c08b734f1e6379e52b4a0d025cddbd9031fb995e9b7dff67640318282030
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
3. instance_soft_window_6036d0f3c0dd4c16: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_6036d0f3c0dd4c16: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_6036d0f3c0dd4c16: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_6036d0f3c0dd4c16: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_6036d0f3c0dd4c16: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_6036d0f3c0dd4c16: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_6036d0f3c0dd4c16: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:4b7f77e6d4304fef947a4f4d6229ae56ea323b26f597eb2547ee9f956b4a2723, sha256:567e03f9b2c8b2894cd6cfb8eb9e1720fd0acbdc04a01c98cd2d60ca3c45bbf9, sha256:5021f15e6bb6a97df749032db71f1a03f09ed08fc874ec651b79de80747a229d, sha256:689cd3e0b7681193846859425bc9cce3bb2d8e5bf5fb18ef48facb79c9a9029e, sha256:905c424bcd31e5d42ceee6e94742312fe8e688ed621e0dd5763e745b857afe2f, sha256:9ac5d9086e65b2240f3ea2ac3295c0e8dcd3bfff46f54592a0f654c83e5cff04
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
2. instance_soft_window_351e4f55104e0cd7: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_351e4f55104e0cd7: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_351e4f55104e0cd7: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_351e4f55104e0cd7: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
2. instance_soft_window_351e4f55104e0cd7: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_351e4f55104e0cd7: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:93d60df48e529464d0d25ed31bf85a673752456b42370da01a655692b8abb5b1, sha256:a6e0ef3f8f46a0dd868ba834f08e0ee140e04058ce8bc580ca5deccc2ae566c1, sha256:571d02f1f0701e2067810b685350ebdcc6d63b0e9ec8718443380e5d8f5454ee, sha256:cafaff438c5f0ac1697d5aa568f5abdbc1de5d147ff5dcc533b0506d071ed404, sha256:e915bde8f3bdc8d381f6a99febe6d7de9c1cfa12417f6413ec25a15965971311, sha256:abd29592ad2791dd14db459eff4020e24a1f8783bc18c063c0f81cd5b96153e5
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
2. instance_soft_window_f18b6b776576be40: MW between 171.145 and 171.147 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_f18b6b776576be40: MW between 171.145 and 171.147 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_f18b6b776576be40: MW between 171.145 and 171.147 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_f18b6b776576be40: MW between 171.145 and 171.147 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_f18b6b776576be40: MW between 171.145 and 171.147 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_f18b6b776576be40: MW between 171.145 and 171.147 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

## sgchem_v1.0__bundle__00056

- split: test
- spec_id: spec_v2_ro5_balanced_test
- source molecule: 64653775453dfb20
- source canonical SMILES: CC(=O)NCC(N)CF
- scaffold hash: 849544041ac73707
- task IDs: sgchem_v1.0__bundle__00056__abstain_contradiction__04, sgchem_v1.0__bundle__00056__audit_accept__02, sgchem_v1.0__bundle__00056__audit_reject__03, sgchem_v1.0__bundle__00056__construct_feasible__01, sgchem_v1.0__bundle__00056__repair_multi_violation__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:ac584552b0d62f3eddcc5a629756df397a56940b771a6f72488656f7eb5307e0, sha256:a60baccb84db00877a698cf6d88867c7a08146a80596551f632eb44d0d0322aa, sha256:79c73b4ac4c3aabc93645bac216d7f1416e1a414cdef31664260c7e8b1fbe696, sha256:58989491a693cc06a3595d5641002b0edcea855642521b58e172538ffb9c8c3f, sha256:0d5969899356d8510e30c8b37f6fcdddd324918ff8c8af14198b7e6d1175ba59
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00056__abstain_contradiction__04

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
2. instance_soft_window_df184201a7ac326b: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00056__audit_accept__02

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
1. ro5_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 120.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. brenk_hard_block: alert set absent: BRENK

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. instance_soft_window_df184201a7ac326b: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
```

### sgchem_v1.0__bundle__00056__audit_reject__03

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
2. instance_soft_window_df184201a7ac326b: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00056__construct_feasible__01

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
2. instance_soft_window_df184201a7ac326b: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00056__repair_multi_violation__05

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

Soft preferences:
1. tpsa_soft_window: TPSA between 20.000 and 130.000 (weight 0.500)
2. instance_soft_window_df184201a7ac326b: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:b53d37e1f0e3bd3bc3bdb3efe1eaa9b69d68a77291727cb3fcc5c830eb10d69f, sha256:1ea7771987ba47e5d220d44aefd5932e1ad9bf37d6bbe5243d05814bfa2ada7b, sha256:b111dc57f4cbb13ec89f9d6d85ed1e2f4a9148e8650434e5a4c9ae3434d35e7b, sha256:62caba9ad8206779ab92165c018f8b0adeaf91c59362285b83aa21339856c672, sha256:b8c2ccc9731dfe1357a9da7136aaccf290adf9c7d04d235edcf47f565a47123b, sha256:ca4eebf181ed012a0df4144bf94a6109f662ee61a9e546d85a6fa7567927a5c0
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
2. instance_soft_window_81c6d67c9bdfdce1: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_81c6d67c9bdfdce1: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_81c6d67c9bdfdce1: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_81c6d67c9bdfdce1: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_81c6d67c9bdfdce1: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_81c6d67c9bdfdce1: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
- agent_visible_hashes: sha256:f7fbf0773ead939756ea50c5808c6410ab73b37a9ba308eb80041ecb550fdf6d, sha256:ea98879a1cdcc8ce795dab4dcb4258e684618f2e8c80edf28977c08eae9f3091, sha256:56bc4638834911b506531eb6c360aa7e6954cb536b17687ff333b92243c17e6b, sha256:67d3399c69f6529583a44c57abc32a40d3930035034319cb65a65c723fa14bc2, sha256:9a54be968e04e7462414a3a0d7699144d76109bf918036cf6156e9997cdbd183
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
4. instance_soft_window_75c2f168ee53e98b: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_75c2f168ee53e98b: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_75c2f168ee53e98b: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_75c2f168ee53e98b: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_75c2f168ee53e98b: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:7a0835c82c1c4a5c39a3188cbb143c1a4a08fb0c2721409535392f5f22f58c04, sha256:197e1f73a5b7310fb7ae8178737b72d9d12882c35dadbb1c54232f0a59292167, sha256:b71bab49e8099d4f159fc1ba1e7a57c350123d228625c3ca49ae67abf7d8e923, sha256:68ff7f5caaab1860673177cbb9b3c54783d648f0e6aec97a348b7b8218ac0ca8, sha256:85f594884113d234c055fe1e0f107374041377ee466d5c8bfe56d20435a821db
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
3. instance_soft_window_e82df8671954427a: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_e82df8671954427a: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_e82df8671954427a: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_e82df8671954427a: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. amide_preference: substructure present: amide (weight 1.000)
2. pains_a_soft: alert set absent: PAINS_A (weight 0.500)
3. instance_soft_window_e82df8671954427a: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:d64c52f921db1522d5ad8b2e9de5f816220571c3acc1a757ad35655127a9dddb, sha256:e6f810120c276f6c51109b8b8487914e2101863d3fbbff4974f2e3f8ce7ed8e9, sha256:a0dc4222886b80ea89db0875bea2ecc8e2cb7e87e18969327e1a0adf95f8afc7, sha256:a8d5419d108b22944d579653e407f1a7939ed853378eccc08a5f81818ad5ad3a, sha256:eacd2e68664719f6bd9036befc30b9e3f47f6541dffe713052fb1f7226b49b73, sha256:8e83cf60f212cab357a7cb93103341c9a55f1a755739a99d8c5ef59074883301
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
2. instance_soft_window_77078bedffcf4c53: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_77078bedffcf4c53: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_77078bedffcf4c53: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_77078bedffcf4c53: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_77078bedffcf4c53: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_77078bedffcf4c53: MW between 121.566 and 121.568 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- task IDs: sgchem_v1.0__bundle__00069__abstain_contradiction__04, sgchem_v1.0__bundle__00069__audit_accept__02, sgchem_v1.0__bundle__00069__audit_reject__03, sgchem_v1.0__bundle__00069__boundary_precision__05, sgchem_v1.0__bundle__00069__boundary_precision__06, sgchem_v1.0__bundle__00069__construct_feasible__01
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:41cfa0ef7e3d3f039541dd4a7d4722051e8edd1931394095634c41bcd3fbe6f1, sha256:9a9b6683849147e0f6e0a846a21f2225371a5281fb4b7cda6bf318278c7183f1, sha256:0ff6cdce34c52a17785c6f08c3a0c62126fb91d280c10005ee8c330b341ee7af, sha256:1d40cf14c7fdeb1a587714c09ae07ed8bf70ef9211396146b4438a27f27283d3, sha256:578efc997be227ab2a5d1494a5e4c439efaef6da5d8ac5f32163354402c8a96b, sha256:5e4be9d2872e881c62a89bb97a7fbd620ad0618fb1fbbbbbd81368734178bf77
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00069__abstain_contradiction__04

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
2. instance_soft_window_877b5c8c766d1e69: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

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
2. instance_soft_window_877b5c8c766d1e69: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_877b5c8c766d1e69: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_877b5c8c766d1e69: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_877b5c8c766d1e69: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_877b5c8c766d1e69: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:3c41912a0fd8962c3d78217e2049fb9998564992a255aa305f8ae814a02ea055, sha256:f97ab24ab8d280bb067fbc9e8061955284ef2e621385e5a4141c73245911b2a0, sha256:c7ea1eeedae459cab54e124f3d6a13441ef81028bd26624dbd27448e3d7fccd7, sha256:aaf09a6774b5eecf827d3769b1ea5ee78330e42a36436480c50eae1dcb293607, sha256:cf8109b0497fa61b3011951c76078282f9d66fc1c1b115fdf8a6c9e37ccd903d, sha256:c04a819b7379f3c29b3ca344fa7b516155995d0514ae09e56c8c50b073b17396
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
2. instance_soft_window_c10614889820dd70: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_c10614889820dd70: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c10614889820dd70: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c10614889820dd70: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_c10614889820dd70: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c10614889820dd70: MW between 105.111 and 105.113 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
- agent_visible_hashes: sha256:f563361daa523bbadd83135250e8d0ed0ba5a822def0db531ef71dcb3803e209, sha256:c121d16724a7a9af106d3e57c3c2ef1a857a85b7f458364d8d148e66f2fac5a1, sha256:2fa837f3a8c9975d77f8ac50219fa16bb46a5d63f0b6a7e00be4056eb07a2631, sha256:4d07322792a3d8062349999f27f26d884742767d367cacbb647a6476cedd3dcd, sha256:942d4f1d1d45daaece05c54f781dd1553370471b88488f22dd9186570e71a8ec, sha256:50c86c3cb0c5c3eb69655bca6ff1464c0e36bafa3191b678088a79681a8ed5a3
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
3. instance_soft_window_d5df2d65f990935e: MW between 151.592 and 151.594 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_d5df2d65f990935e: MW between 151.592 and 151.594 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_d5df2d65f990935e: MW between 151.592 and 151.594 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_d5df2d65f990935e: MW between 151.592 and 151.594 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_d5df2d65f990935e: MW between 151.592 and 151.594 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_d5df2d65f990935e: MW between 151.592 and 151.594 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:bf016aec458711cfeeaf3d13198727b28fda2955bcef716d9340050f591d697f, sha256:8e39fea3f2c08b3dd2852ff91b114cbf5b9dbae569204fd53ef4dff3d4387bc8, sha256:a50c9e8074106b6884a0fc41aecd716ffdc084bf704f982d0a219932ed10da11, sha256:1ee1719e558bc980fad635ad9db69e214bf12bbcce9af4e16c70134a105607f9, sha256:b6e3fe841a79ca63fd9c2e4788966fb245c4645947ad0aa9a1c56e4779152def
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
3. instance_soft_window_68322b411ff65ec9: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_68322b411ff65ec9: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_68322b411ff65ec9: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_68322b411ff65ec9: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_68322b411ff65ec9: MW between 187.600 and 187.602 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00092

- split: test
- spec_id: spec_v2_fragment_ultra_dev
- source molecule: f8434d060e2093f4
- source canonical SMILES: CC(=O)NCC(C)O
- scaffold hash: 114e93e0ac2bca09
- task IDs: sgchem_v1.0__bundle__00092__abstain_contradiction__04, sgchem_v1.0__bundle__00092__audit_accept__02, sgchem_v1.0__bundle__00092__audit_reject__03, sgchem_v1.0__bundle__00092__construct_feasible__01, sgchem_v1.0__bundle__00092__repair_multi_violation__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_multi_violation
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_multi_violation': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:bc90c6b3e9d02cf25e7c3c17e35e804ce96ba22b14d94832171dc570410d5bb1, sha256:20e3d11a7fdde470aa3a8c3c8a4bc423146e2679ed378bd9e36d0401ed4b5f96, sha256:532b4f1f86299df1f9e256713c3d9f4221678af5085a2199766493cb0dda182b, sha256:1b3f4d2a4f41f207b421c6096b63ba5d270caf89577082c5e8cc51bad1c5cedd, sha256:e2062fd09449de7043036765bb68b5fc34e4f4140fb54a3d2e6ebb036b12ac1b
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00092__abstain_contradiction__04

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
2. instance_soft_window_c0b407cb5cfd70f0: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

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
2. instance_soft_window_c0b407cb5cfd70f0: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c0b407cb5cfd70f0: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_c0b407cb5cfd70f0: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. sa_soft: SA proxy maximum 4.800 (weight 0.700)
2. instance_soft_window_c0b407cb5cfd70f0: MW between 117.147 and 117.149 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- task IDs: sgchem_v1.0__bundle__00093__abstain_contradiction__04, sgchem_v1.0__bundle__00093__audit_accept__02, sgchem_v1.0__bundle__00093__audit_reject__03, sgchem_v1.0__bundle__00093__boundary_precision__05, sgchem_v1.0__bundle__00093__boundary_precision__06, sgchem_v1.0__bundle__00093__construct_feasible__01
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:58ae0f4e5191fa9b1677351e1ae1fbee66cffeb0c9138f1a0e314e8cfb6c7f29, sha256:c223515b1195769132a50bdc5f273f53804bf7336ebbe3599daee0d70c2a8b0b, sha256:2f8755f4b8f7374633b7c97c26e2712db85f78a02e44be4846ae589e9b595025, sha256:bb63e04cf529aea883be20767e1f82a234a4f219ccbe3e35db773b051205718d, sha256:873bb9371f6f88f00fc5863e833494fc9e13eba0baf69a663ad49c0db24d71f9, sha256:8d836394b619f04a3ee71358b8e0d87e80ac8b4cac4c49118d80854656571f3f
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00093__abstain_contradiction__04

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
2. instance_soft_window_9ad849b3210c2d42: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

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
2. instance_soft_window_9ad849b3210c2d42: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_9ad849b3210c2d42: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_9ad849b3210c2d42: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_9ad849b3210c2d42: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_9ad849b3210c2d42: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:4e59bd0944627aa3861beb3be4d7fbdeb372602841513ae2953215bace607ece, sha256:999e9fffc9b4fde6cf2a75ac43452aac6f0cc7e682a4777eb0797859a734ab7a, sha256:13f275997527b842ef9f88a1c8946756abe492a6a8034dff421aa93a379dbe30, sha256:1c5d3c2c6e5e8624f291a2dc2362837f359aa0bd859d23ac105096dd86413db3, sha256:a60bf6f0feeeab400dc14012e11f346a555e0b7743f1752dbbc32efbcbbda1b1
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
2. instance_soft_window_dbd1d51fa832fb75: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_dbd1d51fa832fb75: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_dbd1d51fa832fb75: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_dbd1d51fa832fb75: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_dbd1d51fa832fb75: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- task IDs: sgchem_v1.0__bundle__00101__abstain_contradiction__04, sgchem_v1.0__bundle__00101__audit_accept__02, sgchem_v1.0__bundle__00101__audit_reject__03, sgchem_v1.0__bundle__00101__construct_feasible__01, sgchem_v1.0__bundle__00101__interrupt_resume__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, interrupt_resume
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'interrupt_resume': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'interrupt_certificate': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:2e520a36e7da0ce08a86c57a36e1188c46af5f5c65008596ca00e5c83ebe5b9b, sha256:d3e709234465ab9679158e21905c7a16ab8e5e4a6d7bd7fd020380eab3d96da8, sha256:843e336177b8e37af75ed4d4c5d83c9fe52120414a2badb456e5b11d23bb692a, sha256:e3bf1afcc7727b1a4727d1d173aea7b8baf9622081b27565b532d09c04a178a0, sha256:d1c0961e5a7f7686a619202ffb190070292c5c4ddfe75d71e37006ccb5834fa6
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00101__abstain_contradiction__04

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
3. instance_soft_window_c0f1c4b6f3875ecd: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

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
3. instance_soft_window_c0f1c4b6f3875ecd: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_c0f1c4b6f3875ecd: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
3. instance_soft_window_c0f1c4b6f3875ecd: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
3. instance_soft_window_c0f1c4b6f3875ecd: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 3
- max_proposals: 2
- max_verify_calls: 1

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- task IDs: sgchem_v1.0__bundle__00103__abstain_contradiction__04, sgchem_v1.0__bundle__00103__audit_accept__02, sgchem_v1.0__bundle__00103__audit_reject__03, sgchem_v1.0__bundle__00103__construct_feasible__01, sgchem_v1.0__bundle__00103__repair_near_miss__05
- internal task types: abstain_contradiction, audit_accept, audit_reject, construct_feasible, repair_near_miss
- visible task names: feasibility_check, candidate_audit, candidate_audit, construct, repair
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 1}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'repair_near_miss': 1}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'repair_witness': 1}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 1, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'unsat_certificates': 1, 'witnesses': 3}
- agent_visible_hashes: sha256:998cf017b6cadcbe177068593c329aab4bf2a050d0d966dc101a18ac7355ee63, sha256:213236b5b4451547aaaa92e143cfcbdb5780e185c24950defc9f0338ecd064ba, sha256:8ca7675d896f3c9613ab72fdfd6dd032b1975054efa0cacca6ec479218768d11, sha256:f7456e8a7b01ecbbc21dc841557b8792de093b4e40ccd1c872dd3a5a6801d905, sha256:03b799aea3e2148561b02344f76d76d492245a89593bec174212e0494b175529
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00103__abstain_contradiction__04

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
2. instance_soft_window_20cf02d6bb50075b: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

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
2. instance_soft_window_20cf02d6bb50075b: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_20cf02d6bb50075b: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_20cf02d6bb50075b: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_20cf02d6bb50075b: MW between 212.635 and 212.637 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

## sgchem_v1.0__bundle__00105

- split: test
- spec_id: spec_v2_fragment_ultra_dev
- source molecule: 3c0f286881a44cbd
- source canonical SMILES: CC(=O)NCC(N)CCl
- scaffold hash: f7b097e0392c9e34
- task IDs: sgchem_v1.0__bundle__00105__abstain_contradiction__04, sgchem_v1.0__bundle__00105__audit_accept__02, sgchem_v1.0__bundle__00105__audit_reject__03, sgchem_v1.0__bundle__00105__boundary_precision__05, sgchem_v1.0__bundle__00105__boundary_precision__06, sgchem_v1.0__bundle__00105__construct_feasible__01
- internal task types: abstain_contradiction, audit_accept, audit_reject, boundary_precision, boundary_precision, construct_feasible
- visible task names: feasibility_check, candidate_audit, candidate_audit, boundary_audit, boundary_audit, construct
- expected action distribution: {'ABSTAIN': 1, 'ACCEPT': 3, 'REJECT': 2}
- oracle summary: {'task_types': {'construct_feasible': 1, 'audit_accept': 1, 'audit_reject': 1, 'abstain_contradiction': 1, 'boundary_precision': 2}, 'oracle_types': {'feasible_witness': 2, 'violation_certificate': 1, 'unsat_certificate': 1, 'boundary_certificate': 2}, 'expected_actions': {'ACCEPT': 3, 'REJECT': 2, 'ABSTAIN': 1}}
- verifier recomputation summary: see oracle_validation_report.md
- hidden witness/certificate summary: {'boundary_certificates': 2, 'unsat_certificates': 1, 'witnesses': 2}
- agent_visible_hashes: sha256:eb757e8514fd3c6bd814075a7c2d63d9471704007a47d0694c9fdfdaad2ce062, sha256:08a1a9324b2c36b6263a28b3682f80d44c58d40f9bb51ffd8e810f962fc4bad5, sha256:57fe6c47a45e03d05daf69672dad45a96457613a971312cdc790fe1c0a4518a6, sha256:48fba2d1b8132b23ba80338182c22827123ffd925a085d3a2b702214c9b82851, sha256:2900567be57d8ab0cf5196b7b7da7bfd1be9b94edad5b56b86e31c0e889fe591, sha256:318e259d93c510f0ca90c76b8b317a6faaa98c270bc12c96ce0c0493e7af2898
- possible reviewer objection: weak_unsat_certificate
- manual_grade: A
- manual_notes: Strong multi-view bundle with explicit contradiction, accept/reject audit contrast, and oracle-backed visible constraints.
- reviewer_objection: weak_unsat_certificate
- decision: keep
- paper_safe: true

### sgchem_v1.0__bundle__00105__abstain_contradiction__04

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
2. instance_soft_window_9a83d0501c9dd304: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

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
2. instance_soft_window_9a83d0501c9dd304: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00105__audit_reject__03

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
2. instance_soft_window_9a83d0501c9dd304: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_9a83d0501c9dd304: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_9a83d0501c9dd304: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```

### sgchem_v1.0__bundle__00105__construct_feasible__01

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
2. instance_soft_window_9a83d0501c9dd304: MW between 150.608 and 150.610 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
Verifier-tool availability: verify(smiles) may be used within the verify-call budget.
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
- agent_visible_hashes: sha256:53a69fbc04d6924696095d6948018c969e6d321e9f5ce91b43fa23eb7f85511d, sha256:a42d972c74daf66999877794497caa0c8a6514f7f96a1fcecdc5310ac0fcc6d6, sha256:98a52fe4734772e8ea53c5e034b543f5c32bda90944660533e03ce3be9a68889, sha256:66c2706c6960ed1ab39aaffe3a88bc6dd2f461e76bcb4941cad2d0817be8a2ef, sha256:731b4c1f5cd6b51146d1e24658978fe52b36ae8b61d9d3106e3e4927cb68daa2
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
2. instance_soft_window_1db14d7de063e3d2: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_1db14d7de063e3d2: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_1db14d7de063e3d2: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_1db14d7de063e3d2: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_1db14d7de063e3d2: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:9e1152a9e1b94681fd0405f6bb41a48a2d9ab68564b15b3c178ec80b16226b40, sha256:b25d81fd54798985e5cbb6522713f9b3cb9f52c0196cc5627c370cdd6a254aed, sha256:15a58a2a2238de8c43234a3ab1d2d33f938bb5868d48f46ff5087b0645227636, sha256:c28a3663e0c4bf721a244a54b5a67a2ee60d4d7775ec0c89fa2da4a51040f361, sha256:84e60f0bfc5a104b17e3da0b3828151151cd6d5c53d55218499fb9399d2077c2
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
2. instance_soft_window_7b676fe36a4fb9c3: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
2. instance_soft_window_7b676fe36a4fb9c3: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_7b676fe36a4fb9c3: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
2. instance_soft_window_7b676fe36a4fb9c3: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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

Soft preferences:
1. brenk_soft_absent: alert set absent: BRENK (weight 0.600)
2. instance_soft_window_7b676fe36a4fb9c3: MW between 135.137 and 135.139 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
- agent_visible_hashes: sha256:0de174ba776c32a4b1f4bbd46273f9e37798de7399ed59c10770392a24f8b209, sha256:af47aa59342d9cd531b6e2404d587dfb39e9fb82f3c3cc53b1e0f3a038e3bfd8, sha256:fff9682347b4051afd218e9c78076b6d15fb2daedc4dfa389137a537ccc2df9b, sha256:2eeac24c31a0c348a94c41bb746a42786f1da2a3c31ed2c7223475873fc7e4ca, sha256:5dd279ef1961ba30237b9e9c1bbe8d06645ecdb027e38dc3d3e145aed1905496, sha256:81d9d671887d01b8140753cadecb1dedb0b44dc05c932157c7e49f4619c7c229
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
4. instance_soft_window_6122ac99941ecf12: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_6122ac99941ecf12: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L2.
Budget:
- max_steps: 3
- max_proposals: 3
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_6122ac99941ecf12: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L3.
Budget:
- max_steps: 4
- max_proposals: 4
- max_verify_calls: 4

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_6122ac99941ecf12: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
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
4. instance_soft_window_6122ac99941ecf12: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
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
4. instance_soft_window_6122ac99941ecf12: MW between 134.153 and 134.155 (weight 0.010)

Protocol: L1.
Budget:
- max_steps: 1
- max_proposals: 1
- max_verify_calls: 0

Output schema:
{"action": "ACCEPT|REJECT|ABSTAIN", "rationale": "...", "smiles": "..."}
Include a SMILES value only when returning or accepting a molecule.
Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.
```
