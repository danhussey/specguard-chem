---
license: mit
pretty_name: SpecGuard-Chem sgchem_v1.0
task_categories:
  - text-generation
  - question-answering
language:
  - en
size_categories:
  - n<1K
---

# SpecGuard-Chem sgchem_v1.0

SpecGuard-Chem sgchem_v1.0 is an oracle-first compiler release and evaluation artifact for agentic language models under medicinal-chemistry specifications.

Authorship is anonymous for double-blind review.

## Dataset Description

The artifact contains a deterministic sgchem_v1.0 release with 118 bundles, 656 tasks, and 266 test tasks. Each bundle is one underlying specification scenario with controlled public task views for construction, repair, candidate audit, rejection, abstention, boundary precision, representation invariance, and interrupt/resume behavior.

Tasks include hidden oracle evidence such as witnesses and certificates, but normal model adapters consume sanitized public task views. Prompt leakage, oracle scrambling, strict validation, clean-clone reproduction, and anonymous artifact scans are included in the release audits.

## Intended Uses

- Offline benchmark evaluation of medicinal-chemistry specification compliance.
- Measuring ACCEPT, REJECT, and ABSTAIN behavior.
- Measuring unsafe acceptance, rejection recall, abstention recall, verifier-tool use, boundary precision, representation invariance, and interrupt/resume behavior.
- Reproducing the paper tables, figures, audits, and deterministic baseline runs.

## Out-of-Scope Uses

SpecGuard-Chem does not evaluate biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.

The artifact must not be used to make claims about real-world molecular safety, efficacy, clinical relevance, target engagement, synthesis feasibility, or developability.

## Scope

The benchmark uses medicinal-chemistry rule constraints: molecular property ranges, substructure requirements, alert filters, similarity guards, and representation-equivalence checks. The scope is specification compliance, not molecule discovery.

## Files

- `sgchem_v1.0_anonymous_artifact.zip`: full anonymous review artifact

Archive SHA256:

```text
c0e248e9a4e1b5f7be41082bfdbf186fb29f421aaf2878fe56c93f845fd70568
```

Inside the archive:

- `benchmarks/releases/sgchem_v1.0/`
- `src/`
- `scripts/`
- `baselines/`
- `paper_v1/`
- `paper/`
- `tests/`
- root documentation, lockfile, and license

## Reproducibility

```bash
uv run python scripts/build_and_check_sgchem_v1.py
uv run python scripts/test_clean_reviewer_reproduction.py
uv run python scripts/preflight_neurips_ed_artifact.py --release benchmarks/releases/sgchem_v1.0
uv run python scripts/check_paper_consistency.py --mode rc
```

## Croissant

Croissant metadata is included at `benchmarks/releases/sgchem_v1.0/croissant.json`. Local Croissant validation passes in the artifact pipeline. After this hosted URL is recorded in the release, run:

```bash
uv run python scripts/finalize_hosted_url.py \
  --release benchmarks/releases/sgchem_v1.0 \
  --dataset-url "<THIS_DATASET_URL>"
```

## License

MIT.

## Citation

Citation placeholder for double-blind review. Full citation will be provided after review.
