# sgchem_v1.0 Release Notes

This release is a hard cutover to oracle-first bundle compilation.

- removed clone-fill task padding
- split by bundle rather than spec identity
- added task-level REJECT and explicit ABSTAIN cases
- added rendered agent-visible inputs and stable agent-visible hashes
- generated strict audits, checksums, benchmark card, and Croissant metadata

Generated tasks: 426
Generated bundles: 80
Strict validation: {'valid': True, 'num_errors': 0}

Reproducibility:
```bash
uv run specguard-chem compile-benchmark --benchmark-id sgchem_v1.0 --out benchmarks/releases/sgchem_v1.0 --seed 7 --target-bundles 80 --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict
```
