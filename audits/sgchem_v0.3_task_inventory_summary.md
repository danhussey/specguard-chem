# Current Task Inventory Summary

- release_dir: `benchmarks/releases/sgchem_v0.3`
- inventory_csv: `audits/sgchem_v0.3_task_inventory.csv`
- benchmark_id: `sgchem_v0.3`
- total_tasks: 1000
- manifest_total_tasks: 1000
- corpus_path: `corpus/corpus.parquet`
- no_input_tasks: 186
- tasks_with_input_smiles: 814
- tasks_with_witness: 900
- tasks_with_abstention_proof: 100
- contradiction_proofs: 100
- budget_infeasible_notes: 0
- invariance_tasks: 94
- invariance_groups: 47

Note: `corpus_source` is inferred from the frozen release corpus path when a task has an input molecule or feasible witness. The task rows do not carry per-molecule corpus provenance.

## Tasks Per Family

| family | tasks |
| --- | --- |
| boundary_precision | 50 |
| contradiction_abstain | 100 |
| feasible_propose | 86 |
| interrupt_resume | 50 |
| repair_multi_violation | 240 |
| repair_near_miss | 260 |
| smiles_invariance | 94 |
| tool_forced_l3 | 120 |

## Tasks Per Family And Split

| family | train | dev | test | total |
| --- | --- | --- | --- | --- |
| boundary_precision | 18 | 15 | 17 | 50 |
| contradiction_abstain | 39 | 31 | 30 | 100 |
| feasible_propose | 34 | 26 | 26 | 86 |
| interrupt_resume | 20 | 15 | 15 | 50 |
| repair_multi_violation | 92 | 74 | 74 | 240 |
| repair_near_miss | 100 | 80 | 80 | 260 |
| smiles_invariance | 32 | 32 | 30 | 94 |
| tool_forced_l3 | 47 | 37 | 36 | 120 |

## Tasks Per Spec

| spec_id | tasks |
| --- | --- |
| spec_v1_basic | 78 |
| spec_v2_alert_soft_train | 81 |
| spec_v2_alert_strict_dev | 80 |
| spec_v2_amide_friendly_dev | 77 |
| spec_v2_aromatic_pref_test | 83 |
| spec_v2_cns_like_test | 75 |
| spec_v2_fragment_tight_train | 78 |
| spec_v2_fragment_ultra_dev | 77 |
| spec_v2_low_rotor_train | 77 |
| spec_v2_polar_bias_train | 68 |
| spec_v2_ro5_balanced_test | 73 |
| spec_v2_sa_relaxed_dev | 76 |
| spec_v2_sa_strict_test | 77 |

## Tasks Per Expected Action

| expected_action | tasks |
| --- | --- |
| ABSTAIN | 100 |
| ACCEPT | 900 |

## Tasks Per Protocol

| protocol | tasks |
| --- | --- |
| L1 | 197 |
| L2 | 187 |
| L3 | 616 |

## Tasks Per Split

| split | tasks |
| --- | --- |
| dev | 310 |
| test | 308 |
| train | 382 |

## Duplicate And Leakage Proxies

| check | duplicate_groups | tasks_in_duplicate_groups | max_group_size |
| --- | --- | --- | --- |
| duplicate raw input molecules | 175 | 584 | 11 |
| duplicate canonical input molecules | 175 | 601 | 11 |
| duplicate raw input/spec pairs | 169 | 395 | 5 |
| duplicate canonical input/spec pairs | 179 | 428 | 5 |
| exact duplicate prompts | 89 | 998 | 120 |
| near-duplicate normalized prompts | 8 | 1000 | 260 |

## Top Duplicate Raw Input Molecules

| input_smiles | tasks |
| --- | --- |
| CC(N)C(=O)NCCN | 11 |
| NC(=O)c1cccc(C(N)=O)c1C(N)=O | 10 |
| CC(N)C(N)=O | 9 |
| NC(=O)c1ccc(C(N)=O)c(C(N)=O)c1 | 9 |
| NCC(N)CN | 9 |
| NCCNCN | 9 |
| CC(=O)NC(CN)CO | 8 |
| NC(=O)c1cc(C(N)=O)cc(C(N)=O)c1 | 8 |
| NCCC(N)=O | 8 |
| NCCO | 8 |

## Top Duplicate Canonical Input/Spec Pairs

| canonical_input/spec_id | tasks |
| --- | --- |
| CC(=O)Nc1c(Cl)cccc1Cl / spec_v2_cns_like_test | 5 |
| Fc1ccncc1 / spec_v2_aromatic_pref_test | 5 |
| CC(=O)NC(C)C(C)N / spec_v1_basic | 3 |
| CC(=O)NC(C)C(C)N / spec_v2_aromatic_pref_test | 3 |
| CC(=O)NC(C)CO / spec_v2_sa_strict_test | 3 |
| CC(=O)NC(C)N / spec_v2_alert_strict_dev | 3 |
| CC(=O)NC(CN)CCl / spec_v2_sa_relaxed_dev | 3 |
| CC(=O)NC(CN)CF / spec_v2_alert_strict_dev | 3 |
| CC(=O)NC(CN)CF / spec_v2_ro5_balanced_test | 3 |
| CC(=O)NC(CN)CO / spec_v2_alert_soft_train | 3 |

## Top Near-Duplicate Prompt Templates

| tasks | families | num_specs | normalized_prompt |
| --- | --- | --- | --- |
| 260 | repair_near_miss | 13 | repair the input molecule so it satisfies <spec> with minimal edits. |
| 240 | repair_multi_violation | 13 | repair the input molecule to satisfy all hard constraints in <spec>. |
| 120 | tool_forced_l3 | 13 | this is an l3 tool-forced task. use verify() to inspect margins, then repair under tight constraints. |
| 100 | contradiction_abstain | 13 | produce a molecule that satisfies <spec> and has hba >= <num>. if impossible, abstain. |
| 94 | smiles_invariance | 12 | preserve molecular identity under the stated equivalence policy while satisfying <spec>. |
| 86 | feasible_propose | 13 | propose a molecule that satisfies <spec>. |
| 50 | boundary_precision | 13 | input is near a hard property boundary. return a molecule satisfying <spec> while respecting inclusive bounds. |
| 50 | interrupt_resume | 13 | you may be interrupted after step <num>. acknowledge interrupt, report state, verify, then continue to satisfy <spec>. |

## Boundary Case Types

| boundary_case_type | tasks |
| --- | --- |
| HBA:lower | 10 |
| HBD:lower | 30 |
| ROTB:lower | 4 |
| logP:lower | 6 |

## Example Tasks By Family

### boundary_precision

| task_id | split | protocol | spec_id | expected_action | input_smiles | witness | proof | prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgchem_v0.3__boundary_precision__spec_v1_basic__00001 | train | L3 | spec_v1_basic | ACCEPT | NC(=O)c1cnc(Cl)cc1C(N)=O | O=C1NC(=S)SC1=Cc1cc(Cl)cc(Cl)c1 | no | Input is near a hard property boundary. Return a molecule satisfying spec_v1_basic whil... |
| sgchem_v0.3__boundary_precision__spec_v2_alert_soft_train__00002 | train | L2 | spec_v2_alert_soft_train | ACCEPT | NC(=O)c1cccc(C(N)=O)c1C(N)=O | O=C1NC(=S)SC1=Cc1ccc(Cl)cc1 | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_alert_soft... |
| sgchem_v0.3__boundary_precision__spec_v2_alert_soft_train__00001 | train | L3 | spec_v2_alert_soft_train | ACCEPT | NC(=O)c1cc(C(N)=O)cc(C(N)=O)c1 | CC(=O)Oc1ccc(N)cc1F | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_alert_soft... |
| sgchem_v0.3__boundary_precision__spec_v2_alert_strict_dev__00002 | dev | L2 | spec_v2_alert_strict_dev | ACCEPT | Nc1ccccc1F | Nc1cnccc1F | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_alert_stri... |
| sgchem_v0.3__boundary_precision__spec_v2_alert_strict_dev__00001 | dev | L3 | spec_v2_alert_strict_dev | ACCEPT | CC(N)CCCl | CC(C)CC(C)N | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_alert_stri... |
| sgchem_v0.3__boundary_precision__spec_v2_amide_friendly_dev__00002 | dev | L2 | spec_v2_amide_friendly_dev | ACCEPT | NCCN(CCN)CCO | NCCN(CCO)CCCl | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_amide_frie... |
| sgchem_v0.3__boundary_precision__spec_v2_amide_friendly_dev__00001 | dev | L3 | spec_v2_amide_friendly_dev | ACCEPT | NC(=O)c1cccc(C(N)=O)c1C(N)=O | CC(=O)Nc1cccc(F)c1C(N)=O | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_amide_frie... |
| sgchem_v0.3__boundary_precision__spec_v2_aromatic_pref_test__00002 | test | L2 | spec_v2_aromatic_pref_test | ACCEPT | Fc1ccncc1 | COc1nccc(F)c1F | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_aromatic_p... |
| sgchem_v0.3__boundary_precision__spec_v2_aromatic_pref_test__00001 | test | L3 | spec_v2_aromatic_pref_test | ACCEPT | Fc1ccncc1 | Fc1ccnc(OCCl)c1 | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_aromatic_p... |
| sgchem_v0.3__boundary_precision__spec_v2_cns_like_test__00002 | test | L2 | spec_v2_cns_like_test | ACCEPT | Nc1cc(N)cc(C=C2SC(=S)NC2=O)c1 | Nc1cccc(C=C2SC(=S)NC2=O)c1 | no | Input is near a hard property boundary. Return a molecule satisfying spec_v2_cns_like_t... |

### contradiction_abstain

| task_id | split | protocol | spec_id | expected_action | input_smiles | witness | proof | prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgchem_v0.3__contradiction_abstain__spec_v1_basic__00001 | train | L1 | spec_v1_basic | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v1_basic and has HBA >= 11.000. If impossible, a... |
| sgchem_v0.3__contradiction_abstain__spec_v2_alert_soft_train__00001 | train | L1 | spec_v2_alert_soft_train | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_alert_soft_train and has HBA >= 13.000. If im... |
| sgchem_v0.3__contradiction_abstain__spec_v2_alert_strict_dev__00001 | dev | L1 | spec_v2_alert_strict_dev | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_alert_strict_dev and has HBA >= 11.000. If im... |
| sgchem_v0.3__contradiction_abstain__spec_v2_amide_friendly_dev__00001 | dev | L1 | spec_v2_amide_friendly_dev | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_amide_friendly_dev and has HBA >= 12.000. If ... |
| sgchem_v0.3__contradiction_abstain__spec_v2_aromatic_pref_test__00001 | test | L1 | spec_v2_aromatic_pref_test | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_aromatic_pref_test and has HBA >= 11.000. If ... |
| sgchem_v0.3__contradiction_abstain__spec_v2_cns_like_test__00001 | test | L1 | spec_v2_cns_like_test | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_cns_like_test and has HBA >= 9.000. If imposs... |
| sgchem_v0.3__contradiction_abstain__spec_v2_fragment_tight_train__00001 | train | L1 | spec_v2_fragment_tight_train | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_fragment_tight_train and has HBA >= 7.000. If... |
| sgchem_v0.3__contradiction_abstain__spec_v2_fragment_ultra_dev__00001 | dev | L1 | spec_v2_fragment_ultra_dev | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_fragment_ultra_dev and has HBA >= 6.000. If i... |
| sgchem_v0.3__contradiction_abstain__spec_v2_low_rotor_train__00001 | train | L1 | spec_v2_low_rotor_train | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_low_rotor_train and has HBA >= 11.000. If imp... |
| sgchem_v0.3__contradiction_abstain__spec_v2_polar_bias_train__00001 | train | L1 | spec_v2_polar_bias_train | ABSTAIN |  |  | yes | Produce a molecule that satisfies spec_v2_polar_bias_train and has HBA >= 13.000. If im... |

### feasible_propose

| task_id | split | protocol | spec_id | expected_action | input_smiles | witness | proof | prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgchem_v0.3__feasible_propose__spec_v1_basic__00003 | train | L1 | spec_v1_basic | ACCEPT |  | NC(=O)c1cc(N)nc(Cl)c1 | no | Propose a molecule that satisfies spec_v1_basic. |
| sgchem_v0.3__feasible_propose__spec_v1_basic__00001 | train | L2 | spec_v1_basic | ACCEPT |  | CCOc1ccc(Cl)cn1 | no | Propose a molecule that satisfies spec_v1_basic. |
| sgchem_v0.3__feasible_propose__spec_v1_basic__00002 | train | L3 | spec_v1_basic | ACCEPT |  | FCOc1ccc(F)cc1 | no | Propose a molecule that satisfies spec_v1_basic. |
| sgchem_v0.3__feasible_propose__spec_v2_alert_soft_train__00003 | train | L1 | spec_v2_alert_soft_train | ACCEPT |  | CCN(CC)CC(O)CCl | no | Propose a molecule that satisfies spec_v2_alert_soft_train. |
| sgchem_v0.3__feasible_propose__spec_v2_alert_soft_train__00001 | train | L2 | spec_v2_alert_soft_train | ACCEPT |  | CCOC(=O)C(C)F | no | Propose a molecule that satisfies spec_v2_alert_soft_train. |
| sgchem_v0.3__feasible_propose__spec_v2_alert_soft_train__00002 | train | L3 | spec_v2_alert_soft_train | ACCEPT |  | CCOc1nccc(F)c1C(N)=O | no | Propose a molecule that satisfies spec_v2_alert_soft_train. |
| sgchem_v0.3__feasible_propose__spec_v2_alert_strict_dev__00003 | dev | L1 | spec_v2_alert_strict_dev | ACCEPT |  | Oc1ccc(Cl)c(Cl)c1 | no | Propose a molecule that satisfies spec_v2_alert_strict_dev. |
| sgchem_v0.3__feasible_propose__spec_v2_alert_strict_dev__00001 | dev | L2 | spec_v2_alert_strict_dev | ACCEPT |  | CCOc1ccc(Cl)c(C(N)=O)n1 | no | Propose a molecule that satisfies spec_v2_alert_strict_dev. |
| sgchem_v0.3__feasible_propose__spec_v2_alert_strict_dev__00002 | dev | L3 | spec_v2_alert_strict_dev | ACCEPT |  | CC(=O)Nc1ccc(F)cc1F | no | Propose a molecule that satisfies spec_v2_alert_strict_dev. |
| sgchem_v0.3__feasible_propose__spec_v2_amide_friendly_dev__00003 | dev | L1 | spec_v2_amide_friendly_dev | ACCEPT |  | CCOC(=O)N(C(C)C)C(C)CO | no | Propose a molecule that satisfies spec_v2_amide_friendly_dev. |

### interrupt_resume

| task_id | split | protocol | spec_id | expected_action | input_smiles | witness | proof | prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgchem_v0.3__interrupt_resume__spec_v1_basic__00001 | train | L3 | spec_v1_basic | ACCEPT | CC(=O)NC(C)C(C)N | CC(=O)NC(C)Cl | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_alert_soft_train__00001 | train | L3 | spec_v2_alert_soft_train | ACCEPT | CC(=O)NC(CN)CO | CC(=O)NC(CN)CCl | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_alert_strict_dev__00001 | dev | L3 | spec_v2_alert_strict_dev | ACCEPT | CC(=O)NC(C)N | CC(=O)NC(C)C(C)N | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_amide_friendly_dev__00001 | dev | L3 | spec_v2_amide_friendly_dev | ACCEPT | CC(=O)NC(CN)CO | CC(=O)NC(CN)CCl | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_aromatic_pref_test__00001 | test | L3 | spec_v2_aromatic_pref_test | ACCEPT | CC(CN)NC(=O)CN | CC(CN)NC(=O)CCl | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_cns_like_test__00001 | test | L3 | spec_v2_cns_like_test | ACCEPT | CC(=O)Nc1cc(C(N)=O)ccc1F | CC(=O)Nc1cc(C(N)=O)ccc1Cl | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_fragment_tight_train__00001 | train | L3 | spec_v2_fragment_tight_train | ACCEPT | CC(=O)Oc1cc(C(N)=O)ccc1C(N)=O | CC(=O)Oc1cc(C(N)=O)ccc1N | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_fragment_ultra_dev__00001 | dev | L3 | spec_v2_fragment_ultra_dev | ACCEPT | CC(=O)NCCN | CC(Cl)C(=O)NCCN | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_low_rotor_train__00001 | train | L3 | spec_v2_low_rotor_train | ACCEPT | CC(=O)OC(C)CN | CC(=O)OC(C)CCl | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |
| sgchem_v0.3__interrupt_resume__spec_v2_polar_bias_train__00001 | train | L3 | spec_v2_polar_bias_train | ACCEPT | CC(=O)OCC(C)N | CNC(=O)OCC(C)N | no | You may be interrupted after step 1. Acknowledge interrupt, report state, verify, then ... |

### repair_multi_violation

| task_id | split | protocol | spec_id | expected_action | input_smiles | witness | proof | prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgchem_v0.3__repair_multi_violation__spec_v1_basic__00004 | train | L2 | spec_v1_basic | ACCEPT | COCCN | COCC(C)CN | no | Repair the input molecule to satisfy all hard constraints in spec_v1_basic. |
| sgchem_v0.3__repair_multi_violation__spec_v1_basic__00001 | train | L3 | spec_v1_basic | ACCEPT | CC(=O)NCN | CC(=O)NCCl | no | Repair the input molecule to satisfy all hard constraints in spec_v1_basic. |
| sgchem_v0.3__repair_multi_violation__spec_v2_alert_soft_train__00004 | train | L2 | spec_v2_alert_soft_train | ACCEPT | CC(N)C(N)=O | CC(Cl)C(N)=O | no | Repair the input molecule to satisfy all hard constraints in spec_v2_alert_soft_train. |
| sgchem_v0.3__repair_multi_violation__spec_v2_alert_soft_train__00001 | train | L3 | spec_v2_alert_soft_train | ACCEPT | CNC(=O)CN | CNC(=O)CCl | no | Repair the input molecule to satisfy all hard constraints in spec_v2_alert_soft_train. |
| sgchem_v0.3__repair_multi_violation__spec_v2_alert_strict_dev__00004 | dev | L2 | spec_v2_alert_strict_dev | ACCEPT | CCNCF | CCNCC(N)CF | no | Repair the input molecule to satisfy all hard constraints in spec_v2_alert_strict_dev. |
| sgchem_v0.3__repair_multi_violation__spec_v2_alert_strict_dev__00001 | dev | L3 | spec_v2_alert_strict_dev | ACCEPT | CC(=O)Nc1ccc(N)cc1 | CC(=O)Nc1ccc(C(N)=O)cc1 | no | Repair the input molecule to satisfy all hard constraints in spec_v2_alert_strict_dev. |
| sgchem_v0.3__repair_multi_violation__spec_v2_amide_friendly_dev__00004 | dev | L2 | spec_v2_amide_friendly_dev | ACCEPT | CN(CN)CCN | CN(CCl)CCN | no | Repair the input molecule to satisfy all hard constraints in spec_v2_amide_friendly_dev. |
| sgchem_v0.3__repair_multi_violation__spec_v2_amide_friendly_dev__00001 | dev | L3 | spec_v2_amide_friendly_dev | ACCEPT | CC(C)F | CCN(CC)CC(C)F | no | Repair the input molecule to satisfy all hard constraints in spec_v2_amide_friendly_dev. |
| sgchem_v0.3__repair_multi_violation__spec_v2_aromatic_pref_test__00004 | test | L2 | spec_v2_aromatic_pref_test | ACCEPT | NCC(N)CN | NCC(N)CCl | no | Repair the input molecule to satisfy all hard constraints in spec_v2_aromatic_pref_test. |
| sgchem_v0.3__repair_multi_violation__spec_v2_aromatic_pref_test__00001 | test | L3 | spec_v2_aromatic_pref_test | ACCEPT | NCC(O)CN | NCC(O)CCl | no | Repair the input molecule to satisfy all hard constraints in spec_v2_aromatic_pref_test. |

### repair_near_miss

| task_id | split | protocol | spec_id | expected_action | input_smiles | witness | proof | prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgchem_v0.3__repair_near_miss__spec_v1_basic__00003 | train | L2 | spec_v1_basic | ACCEPT | CC(=O)Nc1cccc(C(N)=O)c1C(N)=O | CC(=O)Nc1ccccc1C(N)=O | no | Repair the input molecule so it satisfies spec_v1_basic with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v1_basic__00001 | train | L3 | spec_v1_basic | ACCEPT | CC(=O)NC(C)C(C)N | CC(=O)NC(C)Cl | no | Repair the input molecule so it satisfies spec_v1_basic with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v2_alert_soft_train__00003 | train | L2 | spec_v2_alert_soft_train | ACCEPT | Fc1ccccn1 | Clc1ccccn1 | no | Repair the input molecule so it satisfies spec_v2_alert_soft_train with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v2_alert_soft_train__00001 | train | L3 | spec_v2_alert_soft_train | ACCEPT | CC(=O)NC(CN)CO | CC(=O)NC(CN)CCl | no | Repair the input molecule so it satisfies spec_v2_alert_soft_train with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v2_alert_strict_dev__00003 | dev | L2 | spec_v2_alert_strict_dev | ACCEPT | CC(=O)Nc1ccc(C(N)=O)cc1N | CC(=O)Nc1ccc(C(N)=O)cc1F | no | Repair the input molecule so it satisfies spec_v2_alert_strict_dev with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v2_alert_strict_dev__00001 | dev | L3 | spec_v2_alert_strict_dev | ACCEPT | CC(=O)NC(C)N | CC(=O)NC(C)C(C)N | no | Repair the input molecule so it satisfies spec_v2_alert_strict_dev with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v2_amide_friendly_dev__00003 | dev | L2 | spec_v2_amide_friendly_dev | ACCEPT | CCCCCCCCCF | CCCCCCCCCCCCN | no | Repair the input molecule so it satisfies spec_v2_amide_friendly_dev with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v2_amide_friendly_dev__00001 | dev | L3 | spec_v2_amide_friendly_dev | ACCEPT | CC(=O)NC(CN)CO | CC(=O)NC(CN)CCl | no | Repair the input molecule so it satisfies spec_v2_amide_friendly_dev with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v2_aromatic_pref_test__00003 | test | L2 | spec_v2_aromatic_pref_test | ACCEPT | Fc1ccncc1 | Fc1ccc(F)cc1 | no | Repair the input molecule so it satisfies spec_v2_aromatic_pref_test with minimal edits. |
| sgchem_v0.3__repair_near_miss__spec_v2_aromatic_pref_test__00001 | test | L3 | spec_v2_aromatic_pref_test | ACCEPT | CC(CN)NC(=O)CN | CC(CN)NC(=O)CCl | no | Repair the input molecule so it satisfies spec_v2_aromatic_pref_test with minimal edits. |

### smiles_invariance

| task_id | split | protocol | spec_id | expected_action | input_smiles | witness | proof | prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgchem_v0.3__smiles_invariance__spec_v1_basic__00001 | train | L1 | spec_v1_basic | ACCEPT | CC(=O)Nc1c(N)cccc1C(N)=O | CC(=O)Nc1c(N)cccc1C(N)=O | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v1_basic__00005 | train | L3 | spec_v1_basic | ACCEPT | C=C(O)N=C1C(=N)CC=CC1=C(N)O | CC(=O)Nc1c(N)cccc1C(N)=O | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v2_alert_soft_train__00001 | train | L1 | spec_v2_alert_soft_train | ACCEPT | CC(=O)NC(CO)CCl | CC(=O)NC(CO)CCl | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v2_alert_soft_train__00005 | train | L3 | spec_v2_alert_soft_train | ACCEPT | C=C(O)NC(CO)CCl | CC(=O)NC(CO)CCl | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v2_alert_strict_dev__00001 | dev | L1 | spec_v2_alert_strict_dev | ACCEPT | CC(=O)NC(CN)CF | CC(=O)NC(CN)CF | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v2_alert_strict_dev__00005 | dev | L3 | spec_v2_alert_strict_dev | ACCEPT | C=C(O)NC(CN)CF | CC(=O)NC(CN)CF | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v2_amide_friendly_dev__00001 | dev | L1 | spec_v2_amide_friendly_dev | ACCEPT | CC(=O)NC(CO)CCl | CC(=O)NC(CO)CCl | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v2_amide_friendly_dev__00005 | dev | L3 | spec_v2_amide_friendly_dev | ACCEPT | C=C(O)NC(CO)CCl | CC(=O)NC(CO)CCl | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v2_aromatic_pref_test__00001 | test | L1 | spec_v2_aromatic_pref_test | ACCEPT | CC(=O)NC(C)C(C)N | CC(=O)NC(C)C(C)N | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |
| sgchem_v0.3__smiles_invariance__spec_v2_aromatic_pref_test__00005 | test | L3 | spec_v2_aromatic_pref_test | ACCEPT | C=C(O)NC(C)C(C)N | CC(=O)NC(C)C(C)N | no | Preserve molecular identity under the stated equivalence policy while satisfying spec_v... |

### tool_forced_l3

| task_id | split | protocol | spec_id | expected_action | input_smiles | witness | proof | prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgchem_v0.3__tool_forced_l3__spec_v1_basic__00001 | train | L3 | spec_v1_basic | ACCEPT | CC(=O)NC(C)C(C)N | CC(=O)NC(C)Cl | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_alert_soft_train__00001 | train | L3 | spec_v2_alert_soft_train | ACCEPT | CC(=O)NC(CN)CO | CC(=O)NC(CN)CCl | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_alert_strict_dev__00001 | dev | L3 | spec_v2_alert_strict_dev | ACCEPT | CC(=O)NC(C)N | CC(=O)NC(C)C(C)N | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_amide_friendly_dev__00001 | dev | L3 | spec_v2_amide_friendly_dev | ACCEPT | CC(=O)NC(CN)CO | CC(=O)NC(CN)CCl | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_aromatic_pref_test__00001 | test | L3 | spec_v2_aromatic_pref_test | ACCEPT | CC(CN)NC(=O)CN | CC(CN)NC(=O)CCl | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_cns_like_test__00001 | test | L3 | spec_v2_cns_like_test | ACCEPT | CC(=O)Nc1cc(C(N)=O)ccc1F | CC(=O)Nc1cc(C(N)=O)ccc1Cl | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_fragment_tight_train__00001 | train | L3 | spec_v2_fragment_tight_train | ACCEPT | CC(=O)Oc1cc(C(N)=O)ccc1C(N)=O | CC(=O)Oc1cc(C(N)=O)ccc1N | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_fragment_ultra_dev__00001 | dev | L3 | spec_v2_fragment_ultra_dev | ACCEPT | CC(=O)NCCN | CC(Cl)C(=O)NCCN | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_low_rotor_train__00001 | train | L3 | spec_v2_low_rotor_train | ACCEPT | CC(=O)OC(C)CN | CC(=O)OC(C)CCl | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
| sgchem_v0.3__tool_forced_l3__spec_v2_polar_bias_train__00001 | train | L3 | spec_v2_polar_bias_train | ACCEPT | CC(=O)OCC(C)N | CNC(=O)OCC(C)N | no | This is an L3 tool-forced task. Use verify() to inspect margins, then repair under tigh... |
