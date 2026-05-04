# Post-Upload Checklist

Use this checklist after the anonymous dataset URL is available.

Benchmark: SpecGuard-Chem sgchem_v1.0.

## URL Finalization

- [ ] Hosted URL opens from a clean browser session without personal approval.
- [ ] Hosted page does not show personal names, institution names, emails, named GitHub accounts, or local paths.
- [ ] Uploaded archive checksum matches `c0e248e9a4e1b5f7be41082bfdbf186fb29f421aaf2878fe56c93f845fd70568`.
- [ ] Run:

```bash
uv run python scripts/finalize_hosted_url.py \
  --release benchmarks/releases/sgchem_v1.0 \
  --dataset-url "<ANONYMOUS_HOSTED_DATASET_URL>"
```

## Files Updated By Finalizer

- [ ] `benchmarks/releases/sgchem_v1.0/MANIFEST.json`
- [ ] `benchmarks/releases/sgchem_v1.0/croissant.json`
- [ ] `benchmarks/releases/sgchem_v1.0/BENCHMARK_CARD.md`
- [ ] `benchmarks/releases/sgchem_v1.0/RELEASE_NOTES.md`
- [ ] `README.md`
- [ ] `paper_v1/artifact_links.md`
- [ ] `benchmarks/releases/sgchem_v1.0/checksums/sha256sums.txt`
- [ ] `benchmarks/releases/sgchem_v1.0/audits/neurips_ed_preflight_report.md`
- [ ] `benchmarks/releases/sgchem_v1.0/audits/reviewer_attack_report.md`

## Final Checks

- [ ] `uv run specguard-chem validate-croissant benchmarks/releases/sgchem_v1.0/croissant.json --anonymous`
- [ ] `uv run python scripts/preflight_neurips_ed_artifact.py --release benchmarks/releases/sgchem_v1.0 --dataset-url "<ANONYMOUS_HOSTED_DATASET_URL>"`
- [ ] `uv run python scripts/check_paper_consistency.py --mode final`
- [ ] `uv run python scripts/test_clean_reviewer_reproduction.py`
- [ ] Reviewer attack report has no red flags.
- [ ] Hosted artifact URL is no longer the only yellow flag.
