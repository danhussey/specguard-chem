# Safety & Scope

SpecGuard-Chem is a rule-following benchmark, not a drug-design system.

## Non-negotiable boundaries
- No claims about biological activity, efficacy, toxicity, or clinical utility.
- No synthesis planning, retrosynthesis, or route optimization.
- No target-specific or disease-specific optimization objectives.

## Allowed content
- Generic descriptor constraints (MW, logP, TPSA, HBD/HBA, ROTB, SA proxy).
- Structural alert checks (PAINS/BRENK families) as rule filters.
- Synthetic benchmark tasks and deterministic verifier feedback.

## Operational safeguards
- Benchmark correctness must be offline-capable and deterministic.
- No external web/service dependency for verifier correctness or scoring.
- Dataset generation must be seeded and reproducible.
- Invalid adapter outputs are logged explicitly (schema/invalid-action/tool-call rates).

## Data policy
- Tasks are synthetic and non-clinical.
- No proprietary datasets required.
- No patient data or sensitive biomedical records.

## Review rule
If a proposed feature materially shifts the project toward practical compound discovery or misuse-enabling guidance, reject it.
