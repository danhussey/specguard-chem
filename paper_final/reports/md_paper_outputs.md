# MD Research Paper Outputs

Use this directory as the canonical repo-output package for the MD Research paper.

## Status

- Dataset: `benchmarks/releases/sgchem_v1.0`
- Subset manifest: `md_research/config/external_subset.json`
- Offline deterministic rows: 8
- External diagnostic rows: 12
- External providers: OpenAI, Anthropic, DeepSeek
- Replay status: cached live responses replayed into `raw_runs/external_replay/aggregate.json`

## Canonical Tables

- Interpretation notes and caveats: `INTERPRETATION_NOTES.md`
- Metric definitions: `tables/metric_definitions.md`
- Baseline groups: `tables/baseline_groups.md`
- External diagnostic snapshot: `tables/external_diagnostic_snapshot.md`
- Molecule acceptance versus action accuracy: `tables/molecule_acceptance_vs_action_accuracy.md`
- Reject/abstain/action-collapse analysis: `tables/reject_abstain_action_collapse.md`
- Wrapper saturation: `tables/wrapper_saturation.md`
- Protocol-slice caveat: `tables/protocol_caveat.md`

CSV versions of the same tables are in `tables/`.

## Canonical Figures

- Figure 1: `figures/figure1_valid_molecule_not_valid_decision.{png,pdf}`
- Figure 2: `figures/figure2_specguard_architecture.{png,pdf}`
- Figure 3: `figures/figure3_molecule_acceptance_vs_action_accuracy.{png,pdf}`
- Figure 4: `figures/figure4_action_recall_by_type.{png,pdf}`
- Figure 5: `figures/figure5_wrapper_saturation.{png,pdf}`

## Raw And Replay Artifacts

- Offline subset run: `raw_runs/subset_offline/aggregate.json`
- Anthropic live run: `raw_runs/external_live_anthropic/aggregate.json`
- Previous OpenAI live row: `raw_runs/external_live/openai_fast_closed/`
- Previous OpenAI/DeepSeek live continuation: `raw_runs/external_live_remaining/aggregate.json`
- Canonical all-provider replay: `raw_runs/external_replay/aggregate.json`
- External live cache root: `cache/external_live/`
- Final preflight record: `validation/external_model_preflight.json`

The paper-facing tables and figures should cite the replay outputs, not the split live-run directories.
