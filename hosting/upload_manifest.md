# Upload Manifest

## Primary Artifact

| file | sha256 | purpose |
| --- | --- | --- |
| `sgchem_v1.0_anonymous_artifact.zip` | `c0e248e9a4e1b5f7be41082bfdbf186fb29f421aaf2878fe56c93f845fd70568` | anonymous review artifact |

## Included Content Summary

- `benchmarks/releases/sgchem_v1.0/`: compiled dataset release
- `src/`: implementation
- `scripts/`: build, audit, packaging, URL finalization, and consistency checks
- `baselines/`: deterministic baseline configuration
- `paper_v1/`: generated paper tables, figures, audits, and claim ledger
- `paper/`: manuscript skeleton and appendix notes
- `tests/`: local test suite
- root documentation, lockfile, and license

## Host Metadata

- Dataset title: SpecGuard-Chem sgchem_v1.0
- Version: sgchem_v1.0
- License: MIT
- Authorship: anonymous for double-blind review
- Contact: omitted for double-blind review
- Visibility: reviewer-accessible anonymous URL

## Required Post-Upload Verification

1. Download the uploaded archive from a clean browser session.
2. Verify its SHA256 equals `c0e248e9a4e1b5f7be41082bfdbf186fb29f421aaf2878fe56c93f845fd70568`.
3. Run `scripts/finalize_hosted_url.py` with the hosted URL.
4. Run `scripts/check_paper_consistency.py --mode final`.
5. Confirm `dataset_url_accessible=passed` in `benchmarks/releases/sgchem_v1.0/MANIFEST.json`.
