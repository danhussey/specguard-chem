# Paper Insertion Memo

## Proposed Results Section

### Finding 1: Molecule acceptance overstates specification success.
The expanded baselines show that high molecule acceptance can be achieved without satisfying task-level action semantics. For example, `always_accept` has molecule acceptance 0.707, but its task-inconsistent acceptance rate is 0.517.

### Finding 2: Failures are action- and family-specific.
Per-family tables and action-confusion matrices show which task families drive each failure mode, including reject collapse, abstention collapse, boundary precision failures, and interrupt/resume behavior.

### Finding 3: Public verifier/search access saturates the contract.
The wrapper result should be presented as a declared-access ceiling: action accuracy 1.000, reject recall 1.000, and abstain recall 1.000. This supports the interpretation of SpecGuard-Chem as an evaluation-contract artifact rather than a chemistry-capability leaderboard.

### Finding 4: Metric choice changes apparent conclusions.
The ranking-sensitivity table shows that molecule acceptance, action accuracy, reject recall, abstain recall, and cost-adjusted scores select different winners.

### Finding 5: Expanded audits preserve public/private isolation.
The public adapter boundary remains separate from scorer-side labels and oracle certificates; audit logs are included with the paper_v2 result package.

## Recommended Main-Paper Figures/Tables

- Table: representative baseline matrix with action metrics.
- Figure: per-family action accuracy heatmap.
- Figure: wrapper ablation / verifier-budget curve.
- Table: metric winners by objective.

## Appendix Tables

- Full offline baseline matrix.
- All confusion matrices.
- Protocol ladder.
- External diagnostic snapshot.
- Bootstrap confidence intervals.
- Audit logs and integrity gates.

## Key Wording

The expanded baselines show that high molecule acceptance is easy to obtain with accept-biased or retrieval-heavy systems, but those systems fail reject and abstain semantics. The wrapper ablations show that saturation is driven by public verifier/search access rather than hidden oracle leakage. Therefore, SpecGuard-Chem should be interpreted as an evaluation-contract artifact with explicit access-model ceilings, not as a chemistry capability leaderboard.
