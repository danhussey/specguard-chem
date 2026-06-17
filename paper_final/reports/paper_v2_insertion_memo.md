# Paper Insertion Memo

## Proposed new Results section

### Finding 1: Molecule acceptance overstates specification success.
`always_accept` reached molecule_acceptance_rate=0.707 but action_accuracy=0.564, illustrating why acceptance alone is not a task-success metric.

### Finding 2: Failures are action- and family-specific.
Use the per-family heatmap and confusion matrices to show which systems collapse reject or abstain tasks into accept decisions.

### Finding 3: Public verifier/search access saturates the contract.
The wrapper row measured action_accuracy=1.000 under the public verifier/search access model. Present it as an access-model ceiling, not a closed-book baseline.

### Finding 4: Metric choice changes apparent conclusions.
Report metric winners by objective to show how molecule acceptance, reject recall, abstain recall, and cost-adjusted action scores select different systems.

### Finding 5: Expanded audits preserve public/private isolation.
Cite strict validation, prompt leakage, oracle scrambling, and the v2 consistency check as result-generation gates.

## Recommended main-paper figures/tables

- Table: representative baseline matrix with action metrics
- Figure: per-family action accuracy heatmap
- Figure: wrapper ablation / verifier-budget curve
- Table: metric winners by objective

## Appendix tables

- full offline baseline matrix
- all confusion matrices
- protocol-slice analysis
- external diagnostic snapshot
- bootstrap confidence intervals
- audit logs and integrity gates

## Key wording

The expanded baselines show that high molecule acceptance is easy to obtain with accept-biased or retrieval-heavy systems, but those systems fail reject and abstain semantics. The wrapper ablations show that saturation is driven by public verifier/search access rather than hidden oracle leakage.

Therefore, SpecGuard-Chem should be interpreted as an evaluation-contract artifact with explicit access-model ceilings, not as a chemistry capability leaderboard.
