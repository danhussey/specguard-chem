# Safety & Scope

SpecGuard-Chem uses medicinal-chemistry constraints such as molecular property ranges, substructure requirements, alert filters, similarity guards, and representation equivalence checks. It does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, dosing, or disease relevance.

## Non-Claims
- No biological activity, potency, toxicity, efficacy, therapeutic, clinical, patient, dosing, disease, target-binding, or synthesis-route claims.
- No synthesis planning, retrosynthesis, route optimization, docking, toxicity prediction, activity prediction, or target selection.
- No claim that a hard-passing molecule is useful, safe, effective, synthesizable, or developable.

## Allowed Benchmark Content
- Medicinal-chemistry property ranges such as MW, logP, TPSA, HBD, HBA, and ROTB.
- Substructure-present and substructure-absent requirements.
- Alert-family filters such as PAINS and BRENK as rule checks only.
- SA proxy constraints as benchmark heuristics, not synthesis-feasibility claims.
- Representation equivalence, boundary precision, abstention, interrupt/resume, and verifier-tool protocol tests.

## Operational Safeguards
- Verifiers and scoring are deterministic and offline.
- No external web calls are required for dataset generation, validation, or scoring.
- Agent-visible task text is scanned for out-of-scope claim terms.
- Baseline performance is run only after strict release validation and is not used for task selection.

## sgchem_v1.0 Gate
Before reporting results:

```bash
uv run specguard-chem compile-benchmark --benchmark-id sgchem_v1.0 --out benchmarks/releases/sgchem_v1.0 --seed 7 --target-bundles 80 --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```
