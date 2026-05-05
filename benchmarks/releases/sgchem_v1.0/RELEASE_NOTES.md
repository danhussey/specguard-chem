# sgchem_v1.0 Release Notes

This release is a hard cutover to oracle-first bundle compilation.

- removed clone-fill task padding
- split by bundle rather than spec identity
- added task-level REJECT and explicit ABSTAIN cases
- added rendered agent-visible inputs and stable agent-visible hashes
- generated strict audits, checksums, benchmark card, and Croissant metadata

Generated tasks: 656
Generated bundles: 118
Strict validation: {'valid': True, 'num_errors': 0}

Reproducibility:
```bash
uv run specguard-chem compile-benchmark --benchmark-id sgchem_v1.0 --out benchmarks/releases/sgchem_v1.0 --seed 7 --target-bundles 120 --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```

<!-- sgchem-hosted-url:start -->
## Anonymous Hosted Artifact

- Dataset URL: https://huggingface.co/datasets/anon2389434/specguard-chem-sgchem-v1-anonymous
- Review access: anonymous reviewer-accessible dataset page.
- Archive: `sgchem_v1.0_anonymous_artifact.zip`
- Archive SHA256: `c0e248e9a4e1b5f7be41082bfdbf186fb29f421aaf2878fe56c93f845fd70568`
- Croissant metadata: `benchmarks/releases/sgchem_v1.0/croissant.json`
<!-- sgchem-hosted-url:end -->
