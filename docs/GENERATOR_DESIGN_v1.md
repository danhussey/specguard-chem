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

`repair_multi_violation` is validated as a distinct-constraint repair case: the input must fail at least two different hard constraint IDs in the effective task specification. Near-miss repair tasks are limited to exactly one hard violation unit or one configured failing hard constraint.

## Oracle Types
`feasible_witness`, `repair_witness`, `violation_certificate`, `unsat_certificate`, `equivalence_certificate`, `boundary_certificate`, and `interrupt_certificate`.

## Split Policy
Bundles are sorted by a deterministic seed-keyed hash and assigned to train/dev/test with a 50/20/30 target. Group IDs for invariance, boundary, and interrupt tasks are bundle-local and cannot cross splits. After task rendering, bundles linked by exact public input/spec keys are coalesced into one split, and exact duplicate public views are pruned rather than padded.

The compiler no longer emits `instance_soft_window_*` micro preferences. When a public spec needs scenario context, it uses a broad `contextual_property_preference` soft range with human-readable property bounds; strict validation rejects micro soft ranges in public task text.

## Public/Private Task Views
Raw task records contain hidden oracle fields for validation and scoring. Normal adapters receive only `PublicTaskView`: rendered agent input, public molecule input, public effective spec, protocol, budgets, allowed actions/tools, round index, interrupt signal, and permitted feedback. `PublicTaskView` excludes `expected_action`, oracle type, evidence, witnesses, certificates, task IDs, bundle IDs, split labels, and internal task labels such as `audit_accept` or `audit_reject`.

## Validation Gates
Strict validation checks schema fields, oracle evidence, bundle references, split leakage, protocol budgets, safety-scope text, prompt visibility, duplicate agent-visible hashes, and minimum task composition. Negative-control tests corrupt witnesses, certificates, split assignments, prompt text, tool budgets, and boundary/invariance groups to prove strict validation fails with clear invariant errors.

## Audit Reports
The compiler and hardening scripts write task inventory, duplicate, split leakage, oracle validation, safety scope, prompt-leakage, oracle-scrambling, claim-readiness, manual test-bundle dossier, artifact-preflight, and reviewer-attack reports under `benchmarks/releases/sgchem_v1.0/audits/`.

## Baseline Tracks
Primary claims use `primary_closed_book` baselines. Tool-enabled baselines, retrieval upper bounds, oracle upper bounds, and external model snapshots are reported in separate tracks so retrieval or oracle assistance cannot be confused with ordinary model performance.

## Metric Denominators
`paper_v1/tables/evaluation_denominators.md` classifies each paper-facing metric as `primary_reportable`, `diagnostic_only`, `appendix_only`, or `not_reportable` based on test denominator size.

## Known Limitations
Scaffold overlap can occur across splits because the minimum viable split policy is bundle-hash based. The leakage report makes this visible. The benchmark remains a rule-compliance benchmark and does not establish real-world molecular utility.

## Reproducibility
```bash
uv run specguard-chem compile-benchmark --benchmark-id sgchem_v1.0 --out benchmarks/releases/sgchem_v1.0 --seed 7 --target-bundles 120 --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
uv run python scripts/build_and_check_sgchem_v1.py
```
