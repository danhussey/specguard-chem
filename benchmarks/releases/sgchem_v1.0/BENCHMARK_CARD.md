# sgchem_v1.0 Benchmark Card

Name: SpecGuard-Chem
Version: sgchem_v1.0
Intended use: offline evaluation of model compliance with machine-checkable medicinal-chemistry constraints.
Out-of-scope use: drug discovery claims, biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, dosing, disease relevance, or target-binding claims.
Medicinal chemistry scope: property ranges, substructure requirements, alert filters, representation equivalence, boundary behavior, abstention, and protocol compliance.

Dataset composition:
- tasks: 688
- bundles: 120
- task types: {"abstain_contradiction": 120, "audit_accept": 120, "audit_reject": 120, "boundary_precision": 44, "construct_feasible": 120, "interrupt_resume": 27, "repair_multi_violation": 30, "repair_near_miss": 36, "smiles_invariance": 44, "tool_forced_l3": 27}
- expected actions: {"ABSTAIN": 120, "ACCEPT": 426, "REJECT": 142}
- protocols: {"L1": 284, "L2": 200, "L3": 204}

Generation process: deterministic bundle compiler `bundle_compiler_v1` from offline corpus molecules and local specs.
Oracle/certificate policy: each task carries a feasible witness, violation certificate, explicit contradiction certificate, equivalence certificate, boundary certificate, or interrupt certificate.
Split policy: {"name": "bundle_hash_seeded_50_20_30", "proportions": {"dev": 0.2, "test": 0.3, "train": 0.5}, "seed": 7, "unit": "bundle"}
Validation policy: strict schema, oracle, split, protocol, and safety-scope validation must pass before reporting results.
Curation policy: generated tasks are retained only when oracle checks and bundle minimums pass; shortfalls are reported rather than clone-filled.
Metrics: paper-facing tables report action accuracy, unsafe acceptance, reject/abstain recall, diagnostic denominators, and molecule_acceptance_rate. The internal accept_rate is only molecule acceptance, not overall task success.
Challenge slice: structural difficulty_tags are assigned from task/spec/oracle metadata before any baseline run.
Limitations: this benchmark measures rule compliance and protocol behavior, not real-world molecular quality.
Safety and misuse considerations: outputs must not be interpreted as therapeutic candidates or biological claims.

Reproducibility commands:
```bash
uv run specguard-chem compile-benchmark --benchmark-id sgchem_v1.0 --out benchmarks/releases/sgchem_v1.0 --seed 7 --target-bundles 120 --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```
