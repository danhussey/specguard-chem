# SpecGuard-Chem sgchem_v1.0

SpecGuard-Chem sgchem_v1.0 is an oracle-first benchmark compiler release and evaluation package for agentic language models under chemically typed medicinal-chemistry-inspired rule cards.

Authorship: anonymous for double-blind review.

## Description

The release contains 118 bundles, 656 tasks, and 266 test tasks. Each bundle represents one underlying specification scenario and contains controlled task views such as construction, repair, candidate audit, rejection, abstention, boundary precision, representation invariance, and interrupt/resume behavior.

Tasks are generated from machine-checkable witnesses and certificates. The benchmark evaluates whether systems follow visible specifications and protocols while hidden oracle fields remain isolated from model prompts.

## Intended Use

Use this dataset for offline evaluation of specification compliance under medicinal-chemistry rule constraints. The intended comparisons include closed-book model behavior, verifier-tool-enabled behavior, molecule-retrieval baselines, verifier/search-wrapper saturation, rejection, abstention, task-inconsistent acceptance, boundary precision, representation invariance, and interrupt/resume behavior. The dataset is an evaluation contract and audit artifact, not an intrinsic chemistry-capability leaderboard.

## Out-of-Scope Use

SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.

Do not use this artifact to rank molecules for real-world safety, efficacy, clinical relevance, target engagement, or developability.

## Medicinal Chemistry Scope

The specifications use molecular property ranges, substructure constraints, alert filters, similarity guards, and representation-equivalence checks. Medicinal-chemistry wording is used only for scoped rule-compliance evaluation.

## Release Contents

- `benchmarks/releases/sgchem_v1.0/`: release corpus, specs, tasks, bundles, audits, checksums, manifest, benchmark card, release notes, and Croissant metadata
- `src/`: package source
- `scripts/`: build, audit, packaging, and consistency scripts
- `baselines/`: deterministic baseline configuration
- `paper_v1/`: generated tables, figures, claim ledger, and artifact links
- `paper/`: paper skeleton and appendix drafts
- `tests/`: unit and artifact tests
- root documentation and lockfile

## Checksums

Archive: `sgchem_v1.0_anonymous_artifact.zip`

SHA256: `c0e248e9a4e1b5f7be41082bfdbf186fb29f421aaf2878fe56c93f845fd70568`

Release-level checksums are in `benchmarks/releases/sgchem_v1.0/checksums/sha256sums.txt`.

## Reproducibility

```bash
uv run python scripts/build_and_check_sgchem_v1.py
uv run python scripts/test_clean_reviewer_reproduction.py
uv run python scripts/preflight_neurips_ed_artifact.py --release benchmarks/releases/sgchem_v1.0
uv run python scripts/check_paper_consistency.py --mode rc
```

After a hosted anonymous URL is available:

```bash
uv run python scripts/finalize_hosted_url.py \
  --release benchmarks/releases/sgchem_v1.0 \
  --dataset-url "<ANONYMOUS_HOSTED_DATASET_URL>"
uv run python scripts/check_paper_consistency.py --mode final
```

## Croissant Metadata

Croissant metadata is provided at `benchmarks/releases/sgchem_v1.0/croissant.json`. Local validation passes. External hosted URL validation should be rerun after upload.

## License

MIT.

## Citation

Citation will be added after double-blind review.
