# Generator Design v1

## Design Goals
`sgchem_v1.0` is an oracle-first benchmark compiler for medicinal-chemistry evaluation test cases. It emits bundles, not template-filled independent tasks. Each bundle represents one underlying specification scenario and contains controlled task views.

## Non-Goals
The generator does not perform synthesis planning, docking, biological activity prediction, toxicity prediction, target-binding prediction, disease modeling, therapeutic selection, or clinical evaluation.

## Why Medicinal Chemistry
The benchmark uses medicinal-chemistry rule checks because they are concrete, deterministic, and locally verifiable: property ranges, substructure requirements, alert filters, similarity guards, and SMILES equivalence.

## Why Not Drug Discovery
The benchmark evaluates rule following and protocol compliance. It does not claim that generated molecules are useful, safe, effective, synthesizable, or clinically relevant.

## Bundle Structure
Each bundle has one `bundle_id`, one `spec_id`, one split, one source molecule context when available, oracle summary metadata, and a list of task IDs. All tasks from a bundle remain in the same split.

## Task Types
`construct_feasible`, `repair_near_miss`, `repair_multi_violation`, `audit_accept`, `audit_reject`, `abstain_contradiction`, `boundary_precision`, `smiles_invariance`, `interrupt_resume`, and `tool_forced_l3`.

## Oracle Types
`feasible_witness`, `repair_witness`, `violation_certificate`, `unsat_certificate`, `equivalence_certificate`, `boundary_certificate`, and `interrupt_certificate`.

## Split Policy
Bundles are sorted by a deterministic seed-keyed hash and assigned to train/dev/test with a 60/20/20 target. Group IDs for invariance, boundary, and interrupt tasks are bundle-local and cannot cross splits.

## Validation Gates
Strict validation checks schema fields, oracle evidence, bundle references, split leakage, protocol budgets, safety-scope text, duplicate agent-visible hashes, and minimum task composition.

## Audit Reports
The compiler writes task inventory, duplicate, split leakage, oracle validation, safety scope, and curation reports under `benchmarks/releases/sgchem_v1.0/audits/`.

## Known Limitations
Scaffold overlap can occur across splits because the minimum viable split policy is bundle-hash based. The leakage report makes this visible. The benchmark remains a rule-compliance benchmark and does not establish real-world molecular utility.

## Reproducibility
```bash
uv run specguard-chem compile-benchmark --benchmark-id sgchem_v1.0 --out benchmarks/releases/sgchem_v1.0 --seed 7 --target-bundles 80 --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```
