# SpecGuard-Chem Revival Review Package

This directory consolidates the paper-facing artifacts for the revived
SpecGuard-Chem manuscript.

## Contents

- `main.tex`: self-contained LaTeX review manuscript using the frozen held-out
  offline results and the strict v3 external-baseline snapshot as diagnostic
  evidence.
- `figures/`: selected paper-facing figures copied from `paper_v2/results` and
  `md_research/results`.
- `tables/`: selected source Markdown tables used for checking the manuscript
  values.
- `reports/`: result summaries, insertion memos, integrity gates, and external
  interpretation notes.
- `build/`: compiled review artifacts.

## Canonical Result Sources

- Primary offline package: `paper_final/results_offline_full_2026_05_20`
- Legacy paper-v2 package: `paper_v2/results`
- Strict external package: `external_baselines/results_full_2026_05_20_strict_v3`
- Frozen benchmark release: `benchmarks/releases/sgchem_v1.0`

The external rows should remain diagnostic even after the strict v3 rerun. They
are useful for interface and agent-control analysis, but the primary
SpecGuard-Chem result set is the offline benchmark package above. The old
`md_research` path is a historical directory name, not a claim that these are a
separate medicinal-chemistry project result.

## Build

This machine currently has `pandoc` but no TeX engine (`pdflatex`, `xelatex`,
`lualatex`, or `tectonic`). The reproducible local build therefore produces HTML
and DOCX from the TeX source:

```bash
pandoc main.tex --standalone --embed-resources --mathjax --resource-path=. -o build/specguard_chem_review.html
pandoc main.tex --resource-path=. -o build/specguard_chem_review.docx
```

For this review handoff, `build/specguard_chem_review.pdf` was also generated
from the self-contained HTML using local Chrome headless print-to-PDF. Treat that
PDF as a review rendering, not as the canonical LaTeX build.

On a machine with LaTeX installed, compile the TeX source normally:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
```
