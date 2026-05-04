# sgchem_v1.0 Benchmark Card

Name: SpecGuard-Chem
Version: sgchem_v1.0
Intended use: offline evaluation of model compliance with machine-checkable medicinal-chemistry constraints.
Out-of-scope use: drug discovery claims, biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, dosing, disease relevance, or target-binding claims.
Medicinal chemistry scope: property ranges, substructure requirements, alert filters, representation equivalence, boundary behavior, abstention, and protocol compliance.

Dataset composition:
- tasks: 426
- bundles: 80
- task types: {"abstain_contradiction": 80, "audit_accept": 80, "audit_reject": 80, "boundary_precision": 26, "construct_feasible": 80, "interrupt_resume": 13, "repair_multi_violation": 14, "repair_near_miss": 14, "smiles_invariance": 26, "tool_forced_l3": 13}
- expected actions: {"ABSTAIN": 80, "ACCEPT": 253, "REJECT": 93}
- protocols: {"L1": 186, "L2": 120, "L3": 120}

Generation process: deterministic bundle compiler `bundle_compiler_v1` from offline corpus molecules and local specs.
Oracle/certificate policy: each task carries a feasible witness, violation certificate, explicit contradiction certificate, equivalence certificate, boundary certificate, or interrupt certificate.
Split policy: {"name": "bundle_hash_seeded_60_20_20", "proportions": {"dev": 0.2, "test": 0.2, "train": 0.6}, "seed": 7, "unit": "bundle"}
Validation policy: strict schema, oracle, split, protocol, and safety-scope validation must pass before reporting results.
Curation policy: generated tasks are retained only when oracle checks and bundle minimums pass; shortfalls are reported rather than clone-filled.
Limitations: this benchmark measures rule compliance and protocol behavior, not real-world molecular quality.
Safety and misuse considerations: outputs must not be interpreted as therapeutic candidates or biological claims.

Reproducibility commands:
```bash
uv run specguard-chem compile-benchmark --benchmark-id sgchem_v1.0 --out benchmarks/releases/sgchem_v1.0 --seed 7 --target-bundles 80 --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```
