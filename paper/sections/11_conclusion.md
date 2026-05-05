# Conclusion

SpecGuard-Chem sgchem_v1.0 provides a validated oracle-compiled artifact for evaluating specification compliance under chemically typed constraints. Its main contribution is not only the task set, but the compiler and audit harness: bundle-aware splitting, hidden-oracle isolation, deterministic validation, negative controls, prompt leakage audits, oracle scrambling, and action-aware metrics.

The deterministic wrapper result is central to the paper's interpretation. It shows that a well-engineered verifier wrapper can saturate this release, so sgchem_v1.0 should not be used as evidence that systems are better at real-world chemistry or that raw agents face an intrinsically hard challenge. The defensible claim is narrower: formalizable scientific-agent benchmarks should publish the oracle contract, leakage audits, negative controls, metric denominators, and verifier-wrapper saturation baselines before making capability claims.
