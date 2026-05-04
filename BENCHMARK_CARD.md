# BENCHMARK_CARD — SpecGuard-Chem sgchem_v1.0

## Name
SpecGuard-Chem sgchem_v1.0.

## Intended Use
Offline evaluation of whether agents follow machine-checkable medicinal-chemistry specifications under bounded protocols.

## Out-of-Scope Use
SpecGuard-Chem must not be used to claim biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, dosing, disease relevance, or target-binding behavior.

## Medicinal Chemistry Scope
The benchmark covers molecular property ranges, substructure requirements, alert filters, similarity guards, representation equivalence, boundary behavior, abstention, and interrupt/tool protocol handling.

## Dataset Composition
The primary release is generated at `benchmarks/releases/sgchem_v1.0`. It contains split task JSONL files, split bundle JSONL files, frozen specs, corpus metadata, audits, checksums, Croissant metadata, release notes, and this benchmark card.

## Task Types
`construct_feasible`, `repair_near_miss`, `repair_multi_violation`, `audit_accept`, `audit_reject`, `abstain_contradiction`, `boundary_precision`, `smiles_invariance`, `interrupt_resume`, and `tool_forced_l3`.

## Expected Actions
Tasks use task-level `expected_action` values: `ACCEPT`, `REJECT`, and `ABSTAIN`. Invalid model outputs are tracked separately in metrics rather than silently treated as correct abstention.

## Protocols
`L1` is one-shot without verifier tools. `L2` allows multi-step coarse feedback without direct verifier access. `L3` allows bounded `verify(smiles)` calls.

## Metrics
Reports include pass@budget where appropriate, expected-action confusion matrix, hard violation rate, unsafe accept rate, correct reject rate, correct abstain rate, abstention utility, tool-call economy, edit economy, calibration metrics, boundary precision, invariance consistency, and interrupt/resume success.

## Generation Process
`sgchem_v1.0` is compiled by `bundle_compiler_v1`. Each bundle represents one underlying specification scenario and emits controlled task views. There is no clone-fill padding; shortfalls are recorded in `MANIFEST.json`.

## Oracle Policy
Every task includes one oracle or certificate: feasible witness, repair witness, violation certificate, explicit contradiction certificate, equivalence certificate, boundary certificate, or interrupt certificate.

## Split Policy
Splits are assigned by deterministic bundle hash using the release seed. All tasks from a bundle stay in the same split. Leakage audits check bundle, group, agent-visible hash, canonical input/spec, witness/spec, and scaffold overlap.

## Validation Policy
`uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict` must pass with zero errors before reporting benchmark results.

## Curation Policy
Generated tasks are retained only if schema, oracle, protocol, split, safety-scope, and duplicate checks pass. Baseline performance is not used to select tasks.

## Limitations
SpecGuard-Chem evaluates rule compliance, not real-world molecular quality. Passing a task does not imply usefulness, safety, efficacy, synthesizability, or developability.

## Safety and Misuse
Agent-visible tasks use scoped medicinal-chemistry language and avoid out-of-scope claims. Audit scans block forbidden claim terms only in agent-visible task text.

## Reproducibility
```bash
uv run specguard-chem compile-benchmark \
  --benchmark-id sgchem_v1.0 \
  --out benchmarks/releases/sgchem_v1.0 \
  --seed 7 \
  --target-bundles 80 \
  --min-tasks 400 \
  --max-tasks 700 \
  --anonymous

uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```
