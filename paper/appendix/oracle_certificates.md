# Appendix: Oracle Certificates

Oracle evidence types include feasible witnesses, repair witnesses, violation certificates, explicit contradiction certificates, equivalence certificates, boundary certificates, and interrupt certificates.

Construct and audit-accept tasks require hard-passing witnesses or candidates. Audit-reject and repair tasks require failing inputs or candidates with verifier-visible hard-constraint violations. Abstention tasks require explicit contradictions rather than search failure. Boundary tasks are paired pass/fail contrasts. Invariance tasks require equivalent representations with the same hard-pass decision.

See `src/specguard_chem/dataset/oracles.py` and `benchmarks/releases/sgchem_v1.0/audits/oracle_validation_report.md`.
