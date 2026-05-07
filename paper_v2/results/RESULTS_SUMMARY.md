# SpecGuard-Chem paper_v2 result summary

## Release and environment
- commit: ae17cfd26c8fb176de0f35a5e1a4dc258874ab4c
- release path: benchmarks/releases/sgchem_v1.0
- validation status: see `validation/`

## Main findings
1. Molecule acceptance versus action accuracy: representative baseline tables separate molecule acceptance from action accuracy and task-inconsistent acceptance.
2. Per-family failure modes: `per_family_metrics_test.csv` and the heatmap show family-specific behavior.
3. Wrapper saturation and ablations: wrapper action accuracy is 1.000 under the public verifier/search access model.
4. Protocol ladder: `protocol_ladder_test.csv` separates L1/L2/L3 behavior and verifier-call use.
5. Metric ranking sensitivity: `metric_winners_by_objective.md` shows that headline metric choice changes apparent winners.
6. External diagnostics, if available: external diagnostics are marked secondary; skipped runs are documented in notes.
7. Audit status: validation and audit logs are stored under `validation/`.

## Recommended paper changes
- main text changes: emphasize action-aware evaluation, access-model separation, and wrapper ceiling interpretation.
- appendix additions: full matrix, confusion matrices, protocol ladder, CIs, and audit gates.
- figures/tables to replace existing ones: representative baseline matrix, per-family heatmap, wrapper ablation/budget curve, and metric winners table.

## Caveats
- synthetic rule-based release
- wrapper-solvable under public verifier/search threat model
- no drug-discovery claims
- external results are diagnostic only
