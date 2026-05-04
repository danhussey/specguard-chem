# SpecGuard-Chem Anonymous Upload Instructions

This directory contains double-blind-safe text for uploading the SpecGuard-Chem sgchem_v1.0 release candidate to an anonymous dataset host.

Preferred host: Hugging Face Dataset under an anonymous account.

Acceptable alternatives: Harvard Dataverse, Kaggle, or OpenML if they can provide an anonymous, reviewer-accessible URL and preserve the release archive checksum.

## Files To Upload

Upload the existing archive:

- `sgchem_v1.0_anonymous_artifact.zip`
- SHA256: `c0e248e9a4e1b5f7be41082bfdbf186fb29f421aaf2878fe56c93f845fd70568`

The archive contains the release, source code, scripts, baselines, paper artifacts, tests, documentation, lockfile, and license required for review reproduction.

## Hugging Face Dataset Steps

1. Create or use an anonymous review account.
2. Create a private or unlisted Dataset repository.
3. Use the dataset name `specguard-chem-sgchem-v1-anonymous`.
4. Upload `sgchem_v1.0_anonymous_artifact.zip`.
5. Copy `hosting/HUGGINGFACE_DATASET_CARD.md` into the dataset README.
6. Confirm the dataset page does not show personal names, institution names, emails, named GitHub accounts, or local paths.
7. Confirm anonymous reviewers can access the page and archive without requesting access from an identifying account.
8. Record the hosted dataset URL.
9. Run:

```bash
uv run python scripts/finalize_hosted_url.py \
  --release benchmarks/releases/sgchem_v1.0 \
  --dataset-url "<ANONYMOUS_HOSTED_DATASET_URL>"
```

10. Run the final rc checks:

```bash
uv run python scripts/build_and_check_sgchem_v1.py
uv run python scripts/test_clean_reviewer_reproduction.py
uv run python scripts/preflight_neurips_ed_artifact.py --release benchmarks/releases/sgchem_v1.0
uv run python scripts/check_paper_consistency.py --mode final
```

## Post-Upload Fields To Update

After the URL is available, update only through `scripts/finalize_hosted_url.py`. The script updates:

- `benchmarks/releases/sgchem_v1.0/MANIFEST.json`
- `benchmarks/releases/sgchem_v1.0/croissant.json`
- `benchmarks/releases/sgchem_v1.0/BENCHMARK_CARD.md`
- `benchmarks/releases/sgchem_v1.0/RELEASE_NOTES.md`
- `README.md`
- `paper_v1/artifact_links.md`
- release checksums
- NeurIPS artifact preflight report
- reviewer attack report

Do not manually replace generated release fields unless the finalizer fails.

## Double-Blind Safety

The upload should not include personal names, email addresses, institution names, local machine paths, named GitHub accounts, or a repository URL tied to an author identity.

The dataset card intentionally uses anonymous authorship wording and omits contact information.
