# Public Paper Examples

These examples are safe for paper-facing use. They omit internal task IDs, bundle IDs, hidden expected actions, oracle evidence, witnesses, proofs, and certificates.

## Example 1: construction

- visible_task_name: construct
- protocol: L2
- allowed_actions: ["ACCEPT by returning a molecule that satisfies every hard constraint.", "ABSTAIN if the task cannot be completed under the stated rules."]

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

## Example 2: construction

- visible_task_name: construct
- protocol: L3
- allowed_actions: ["ACCEPT by returning a molecule that satisfies every hard constraint.", "ABSTAIN if the task cannot be completed under the stated rules."]

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

## Example 3: candidate audit

- visible_task_name: candidate_audit
- protocol: L3
- allowed_actions: ["ACCEPT if the provided molecule satisfies every hard constraint.", "REJECT if the provided molecule violates any hard constraint."]

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

## Example 4: candidate audit

- visible_task_name: candidate_audit
- protocol: L3
- allowed_actions: ["ACCEPT if the provided molecule satisfies every hard constraint.", "REJECT if the provided molecule violates any hard constraint."]

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

## Example 5: repair

- visible_task_name: repair
- protocol: L2
- allowed_actions: ["ACCEPT by returning a molecule that satisfies every hard constraint.", "ABSTAIN if the task cannot be completed under the stated rules."]

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

## Example 6: repair

- visible_task_name: repair
- protocol: L3
- allowed_actions: ["ACCEPT by returning a molecule that satisfies every hard constraint.", "ABSTAIN if the task cannot be completed under the stated rules."]

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

## Example 7: feasibility check

- visible_task_name: feasibility_check
- protocol: L1
- allowed_actions: ["ACCEPT only if a molecule can satisfy every hard constraint.", "ABSTAIN if the hard constraints are internally contradictory."]

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

## Example 8: feasibility check

- visible_task_name: feasibility_check
- protocol: L1
- allowed_actions: ["ACCEPT only if a molecule can satisfy every hard constraint.", "ABSTAIN if the hard constraints are internally contradictory."]

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

## Example 9: boundary audit

- visible_task_name: boundary_audit
- protocol: L2
- allowed_actions: ["ACCEPT if the provided molecule satisfies every hard constraint.", "REJECT if the provided molecule violates any hard constraint."]

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

## Example 10: boundary audit

- visible_task_name: boundary_audit
- protocol: L2
- allowed_actions: ["ACCEPT if the provided molecule satisfies every hard constraint.", "REJECT if the provided molecule violates any hard constraint."]

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

## Example 11: representation invariance

- visible_task_name: representation_invariance
- protocol: L1
- allowed_actions: ["ACCEPT if the provided molecule satisfies every hard constraint.", "REJECT if the provided molecule violates any hard constraint."]

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

## Example 12: representation invariance

- visible_task_name: representation_invariance
- protocol: L1
- allowed_actions: ["ACCEPT if the provided molecule satisfies every hard constraint.", "REJECT if the provided molecule violates any hard constraint."]

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

## Example 13: interrupt/resume repair

- visible_task_name: repair
- protocol: L3
- allowed_actions: ["ACCEPT by returning a molecule that satisfies every hard constraint.", "ABSTAIN if the task cannot be completed under the stated rules."]

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
1. sa_relaxed_bounds: HBA between 0.000 and 11.000; HBD between 0.000 and 5.000; MW between 100.000 and 520.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000
2. sa_cap_relaxed: SA proxy maximum 6.000

Soft preferences:
1. acid_soft_absent: substructure absent: carboxylic_acid (weight 0.500)
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

## Example 14: tool-assisted repair

- visible_task_name: repair
- protocol: L3
- allowed_actions: ["ACCEPT by returning a molecule that satisfies every hard constraint.", "ABSTAIN if the task cannot be completed under the stated rules."]

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
1. aromatic_pref_bounds: HBA between 0.000 and 10.000; HBD between 0.000 and 5.000; MW between 100.000 and 500.000; ROTB between 0.000 and 10.000; logP between -1.000 and 5.000

Soft preferences:
1. phenol_soft_present: substructure present: phenol (weight 0.800)
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
