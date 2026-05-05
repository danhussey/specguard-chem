# Limitations And Safety

SpecGuard-Chem is a synthetic, rule-based benchmark for specification compliance. It does not establish real-world molecular value, safety, efficacy, or developability.

SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.

The boundary, invariance, and interrupt/resume slices are diagnostic in sgchem_v1.0 because their denominators are 10 to 20 test tasks. The challenge slice is larger and structurally defined, but challenge membership does not guarantee every individual task is hard for every model or baseline.

A deterministic verifier wrapper solves the current test split. This means sgchem_v1.0 is not an intrinsic model-capability benchmark for systems that are allowed to engineer directly against the public specification contract. The artifact remains useful as a compiler, audit harness, and evaluation-validity case study.

The public/private isolation guarantee applies to adapters that use the runner's PublicTaskView boundary. New adapters should be audited with the model prompt leakage and oracle scrambling scripts before being reported.

Retrieval and oracle-like upper bounds must remain separated from primary closed-book results.
