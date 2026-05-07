# paper_v2 Results Package

This directory commits the compact, reviewable result package for the paper_v2
analysis. Large raw runner traces under `raw_runs/` are intentionally not part
of the committed artifact.

Committed recovery sources:

- `summaries/normalized_task_results.jsonl`: scorer-side task rows used to
  regenerate action metrics, family slices, confusion matrices, protocol slices,
  wrapper ablations, and bootstrap summaries.
- `summaries/normalized_run_metrics.csv`: run-level metrics by split, adapter,
  access model, and protocol slice.
- `tables/`, `figures/`, and `notes/`: paper-ready derived outputs.
- `commands.log`, `environment.txt`, and `validation/`: exact commands,
  environment, validation, and consistency evidence.

The files named `protocol_ladder_*` are kept under their original generated
filenames for compatibility with the run checklist, but the analysis is a
native protocol slice, not a forced same-task L1/L2/L3 comparison.

To regenerate local raw traces and derived outputs, run:

```bash
bash scripts/run_paper_v2_results.sh
```

By default the script regenerates the mandatory held-out test package. Set
`SGCHEM_PAPER_V2_SPLITS="train dev test"` to regenerate best-effort all-split
offline raw runs locally.
