# sgchem_v1.0 Benchmark Card

Name: SpecGuard-Chem
Version: sgchem_v1.0
Intended use: offline evaluation of model compliance with machine-checkable medicinal-chemistry constraints as an oracle-compiled evaluation contract.
Out-of-scope use: drug discovery claims, biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, dosing, disease relevance, or target-binding claims.
Medicinal chemistry scope: property ranges, substructure requirements, alert filters, representation equivalence, boundary behavior, abstention, and protocol compliance.

Dataset composition:
- tasks: 656
- bundles: 118
- task types: {"abstain_contradiction": 100, "audit_accept": 118, "audit_reject": 116, "boundary_precision": 44, "construct_feasible": 116, "interrupt_resume": 27, "repair_multi_violation": 29, "repair_near_miss": 36, "smiles_invariance": 44, "tool_forced_l3": 26}
- expected actions: {"ABSTAIN": 100, "ACCEPT": 418, "REJECT": 138}
- protocols: {"L1": 261, "L2": 197, "L3": 198}

Generation process: deterministic bundle compiler `bundle_compiler_v1` from offline corpus molecules and local specs.
Public specification instances use broad contextual soft preferences when needed; `instance_soft_window_*` micro ranges are forbidden by strict validation.
Oracle/certificate policy: each task carries a feasible witness, violation certificate, explicit contradiction certificate, equivalence certificate, boundary certificate, or interrupt certificate.
Repair semantics: `repair_near_miss` has one hard violation unit or one configured failing constraint; `repair_multi_violation` requires at least two distinct hard constraint IDs.
Split policy: {"duplicate_public_view_policy": "drop later exact agent-visible duplicates before release writing; prefer test, then dev, then train when retaining one copy", "name": "bundle_hash_seeded_50_20_30_with_duplicate_public_view_pruning", "proportions": {"dev": 0.2, "test": 0.3, "train": 0.5}, "seed": 7, "unit": "bundle"}
Validation policy: strict schema, oracle, split, protocol, and safety-scope validation must pass before reporting results.
Curation policy: generated tasks are retained only when oracle checks and bundle minimums pass; shortfalls are reported rather than clone-filled.
Metrics: paper-facing tables report action accuracy, task-inconsistent acceptance, reject/abstain recall, diagnostic denominators, verifier-wrapper saturation, and molecule_acceptance_rate. The internal accept_rate is only molecule acceptance, not overall task success.
Challenge slice: structural difficulty_tags are assigned from task/spec/oracle metadata before any baseline run.
Limitations: this benchmark measures rule compliance and protocol behavior, not real-world molecular quality or intrinsic chemistry task hardness. A deterministic verifier wrapper can solve the current test split.
Safety and misuse considerations: outputs must not be interpreted as therapeutic candidates or biological claims.

Reproducibility commands:
```bash
uv run specguard-chem compile-benchmark --benchmark-id sgchem_v1.0 --out benchmarks/releases/sgchem_v1.0 --seed 7 --target-bundles 120 --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```
