# SpecGuard-Chem paper_v2 result summary
## Release and environment
- commit: ae17cfd26c8fb176de0f35a5e1a4dc258874ab4c
- release path: benchmarks/releases/sgchem_v1.0
- validation status: pass
## Main findings
1. Molecule acceptance versus action accuracy: accept-biased and retrieval systems can produce high molecule_acceptance_rate while action_accuracy and reject/abstain recalls expose failures.
2. Per-family failure modes: see `tables/per_family_metrics_test.csv` and the heatmap for family-specific denominators and accuracies.
3. Wrapper saturation and ablations: the full wrapper measured action_accuracy=1.000; ablations are reported separately under the verifier/search wrapper access model.
4. Protocol-slice analysis: L1/L2/L3 rows in `tables/protocol_ladder_test.csv` are grouped by the native task protocol, not by a forced same-task protocol intervention.
5. Metric ranking sensitivity: `tables/metric_winners_by_objective.md` shows that headline metric choice changes the apparent winner.
6. External diagnostics, if available: skipped in this run because no v1-compatible replay cache or explicit live configuration was present.
7. Audit status: strict validation, prompt leakage, oracle scrambling, Croissant validation, and consistency outputs are under `validation/`.
## Recommended paper changes
- main text changes: replace acceptance-only toplines with action-aware metrics and access-model separation.
- appendix additions: full matrix, confusion matrices, protocol-slice analysis, external skip/snapshot, bootstrap CIs, and audit gates.
- figures/tables to replace existing ones: representative baseline table, per-family heatmap, wrapper budget curve, metric winners table.
## Caveats
- synthetic rule-based release
- wrapper-solvable under public verifier/search threat model
- no drug-discovery claims
- external results are diagnostic only
