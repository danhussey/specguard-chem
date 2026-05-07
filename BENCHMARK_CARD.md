# BENCHMARK_CARD — SpecGuard-Chem sgchem_v1.0

## Name
SpecGuard-Chem sgchem_v1.0.

## Intended Use
Offline evaluation of whether agents follow machine-checkable, chemically typed specifications under bounded protocols. The benchmark is intended as an evaluation contract and audit harness, not as a measure of real-world chemistry capability.

## Out-of-Scope Use
SpecGuard-Chem must not be used to claim biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, dosing, disease relevance, or target-binding behavior.

## Medicinal Chemistry Scope
The benchmark covers molecular property ranges, substructure requirements, alert filters, similarity guards, representation equivalence, boundary behavior, abstention, and interrupt/tool protocol handling.

## Dataset Composition
The primary release is generated at `benchmarks/releases/sgchem_v1.0`. It contains split task JSONL files, split bundle JSONL files, frozen specs, corpus metadata, audits, checksums, Croissant metadata, release notes, and this benchmark card.

## Task Types
`construct_feasible`, `repair_near_miss`, `repair_multi_violation`, `audit_accept`, `audit_reject`, `abstain_contradiction`, `boundary_precision`, `smiles_invariance`, `interrupt_resume`, and `tool_forced_l3`.

`repair_multi_violation` means the input molecule fails at least two distinct hard constraint IDs under the task's effective specification. `repair_near_miss` is reserved for one hard violation unit or one configured failing constraint.

## Expected Actions
Tasks use task-level `expected_action` values: `ACCEPT`, `REJECT`, and `ABSTAIN`. Invalid model outputs are tracked separately in metrics rather than silently treated as correct abstention.

## Protocols
`L1` is one-shot without verifier tools. `L2` allows multi-step coarse feedback without direct verifier access. `L3` allows bounded `verify(smiles)` calls.

## Metrics
Reports include pass@budget where appropriate, expected-action confusion matrix, hard violation rate, task-inconsistent accept rate, correct reject rate, correct abstain rate, abstention utility, tool-call economy, edit economy, calibration metrics, boundary precision, invariance consistency, and interrupt/resume success.

The internal sweep field `accept_rate` is reported in paper tables as `molecule_acceptance_rate`. It is the share of tasks ending in a verifier-accepted molecule, not overall task success. Paper-facing empirical claims should use action accuracy, task-inconsistent acceptance, reject/abstain recall, denominators, and diagnostic-slice reports rather than aggregate molecule acceptance alone.

## Generation Process
`sgchem_v1.0` is compiled by `bundle_compiler_v1`. Each bundle represents one underlying specification scenario and emits controlled task views. There is no clone-fill padding; shortfalls are recorded in `MANIFEST.json`.

Public prompts may include a broad `contextual_property_preference` soft range to keep specification instances concrete. These ranges are intentionally wide medicinal-chemistry preferences, not witness-level micro windows, and strict validation rejects `instance_soft_window_*` prompts or micro soft ranges.

## Oracle Policy
Every task includes one oracle or certificate: feasible witness, repair witness, violation certificate, explicit contradiction certificate, equivalence certificate, boundary certificate, or interrupt certificate.

## Split Policy
Splits are assigned by deterministic bundle hash using the release seed with a 50/20/30 train/dev/test target. All tasks from a bundle stay in the same split. Bundles linked by exact public input/spec keys are coalesced into one split, and exact duplicate public views are pruned rather than padded. Leakage audits check bundle, group, agent-visible hash, canonical input/spec, witness/spec, and scaffold overlap.

## Public/Private Task Views
Adapters and non-oracle baselines consume `PublicTaskView`, not raw task records. Public views contain rendered task text, public molecule input, public effective spec, protocol, budgets, allowed actions/tools, round index, and permitted feedback. They exclude expected answers, oracle types, evidence, witnesses, proofs, certificates, task IDs, bundle IDs, split labels, and internal answer-encoding task labels.

## Validation Policy
`uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict` must pass with zero errors before reporting benchmark results.

Strict validation is paired with negative controls that intentionally corrupt audit, repair, construct, abstain, boundary, invariance, split, protocol, and prompt-scope invariants. Prompt-leakage and oracle-scrambling audits prove public model inputs are invariant to hidden oracle fields.

## Curation Policy
Generated tasks are retained only if schema, oracle, protocol, split, safety-scope, and duplicate checks pass. Baseline performance is not used to select tasks.

## Limitations
SpecGuard-Chem evaluates rule compliance, not real-world molecular quality. Passing a task does not imply usefulness, safety, efficacy, synthesizability, or developability.

A deterministic engineered verifier wrapper can solve the current test split when it directly implements the public specification contract. This is a saturation baseline, not a failure of validation: sgchem_v1.0 should be used to audit public/private isolation, action semantics, verifier-tool policies, and reporting discipline rather than to claim intrinsic benchmark hardness.

Metric denominator reports classify paper claims as primary, diagnostic, appendix-only, or not reportable. The corpus retrieval row is a molecule-retrieval upper bound, not an upper bound on action-correct task performance. Retrieval, tool-enabled, oracle, external snapshot, and wrapper-guarded rows are separated from the primary closed-book leaderboard.

The structurally defined challenge slice uses `difficulty_tags` assigned from task/spec/oracle metadata before any baseline run. Boundary, invariance, and interrupt slices remain diagnostic in sgchem_v1.0: boundary_precision has 22 test tasks, smiles_invariance has 20, and interrupt_resume has 11.

## Safety and Misuse
Agent-visible tasks use scoped medicinal-chemistry language and avoid out-of-scope claims. Audit scans block forbidden claim terms only in agent-visible task text.

## Reproducibility
```bash
uv run specguard-chem compile-benchmark \
  --benchmark-id sgchem_v1.0 \
  --out benchmarks/releases/sgchem_v1.0 \
  --seed 7 \
  --target-bundles 120 \
  --min-tasks 650 \
  --max-tasks 900 \
  --anonymous

uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
uv run python scripts/build_and_check_sgchem_v1.py
```
