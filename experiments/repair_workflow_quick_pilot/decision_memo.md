# Repair Workflow Quick Pilot Decision Memo

## Verdict

**Recommendation: `revise_pilot_constraints`.**

This pilot does not yet show a strong nontrivial repair-workflow signal. The basic sanity checks worked: no-op fails all repair tasks, and random corpus molecules almost always fail preservation. But the nearest-passing-corpus baseline solves 82.5% of tasks and the local mutation/search baselines solve 100%. That triggers the stop condition: this construction is too easy for deterministic search and should not be scaled as-is.

The result is useful because it says the constrained analogue-repair direction may still be viable, but only if the task generator stops selecting witnesses that are directly recoverable by the same local mutation neighborhood used by the cheap baselines, and if corpus retrieval is made materially weaker through holdout design or stronger preservation/novelty constraints.

## Task Counts

| family | n |
| --- | ---: |
| boundary_repair | 4 |
| construct_under_rule_card | 4 |
| repair_multi_violation | 12 |
| repair_near_miss | 12 |
| scaffold_or_similarity_preserving_repair | 8 |

## Validation

Pilot validation passed with `valid=true`, `num_errors=0`, `num_tasks=40`.

## Cheap Baseline Results

| baseline | n | success@1 | hard_pass_rate | similarity_preservation_rate | scaffold_preservation_rate | valid_smiles_rate | no_op_rate | unrelated_valid_molecule_rate | median_similarity_on_successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_op_starting_molecule | 40 | 0.000 | 0.000 | 1.000 | 1.000 | 0.900 | 1.000 | 0.000 | NA |
| random_valid_corpus | 40 | 0.050 | 0.050 | 0.000 | 0.825 | 1.000 | 0.000 | 0.875 | NA |
| nearest_passing_corpus_by_similarity | 40 | 0.825 | 0.825 | 0.806 | 0.975 | 0.825 | 0.000 | 0.000 | 0.654 |
| local_mutation_search | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.783 |
| verifier_guided_greedy | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.783 |

## Stop Conditions

- no_op_starting_molecule success@1 = 0.000: good; starts are actually failing.
- random_valid_corpus success@1 = 0.050: good; random valid molecules do not generally satisfy analogue preservation.
- nearest_passing_corpus_by_similarity success@1 = 0.825: bad; this exceeds the >80% trivialization threshold.
- local_mutation_search success@1 = 1.000 and verifier_guided_greedy success@1 = 1.000: bad; current tasks are exactly recoverable by simple deterministic mutation/search.

## Model Calls

No LLM calls were run. The cheap-baseline stop condition fired before Phase 4. Running OpenAI/Anthropic models here would mostly measure whether they rediscover an already deterministic search-solvable neighborhood, not whether the repair workflow is intrinsically useful.

## Representative Public Task Examples

### pilot_001_repair_near_miss

family: `repair_near_miss`
spec: `spec_v2_polar_bias_train`
starting_smiles: `CCN(CCN)C(C)CCl`
failure_constraint_ids: `['polar_bounds']`

```text
Task: Repair the starting molecule while preserving analogue similarity.

You are given a machine-checkable medicinal-chemistry rule card.

Starting SMILES:
CCN(CCN)C(C)CCl

Allowed actions:
- REPAIR with a candidate SMILES satisfying all hard constraints.
- ABSTAIN only if no compliant analogue can be found under the stated preservation constraints.

Hard constraints:
1. polar_bounds: HBA between 1.0 and 12.0, HBD between 0.0 and 6.0, MW between 120.0 and 520.0, ROTB between 0.0 and 10.0, TPSA between 40.0 and 180.0, logP between -1.0 and 3.0.
2. pilot_similarity_to_start: Morgan Tanimoto similarity to the starting molecule at least 0.45.

Analogue preservation:
- Candidate Morgan Tanimoto similarity to the starting molecule must be at least 0.45.

```

### pilot_002_repair_near_miss

family: `repair_near_miss`
spec: `spec_v2_polar_bias_train`
starting_smiles: `CCN(CCN)C(C)CF`
failure_constraint_ids: `['polar_bounds']`

```text
Task: Repair the starting molecule while preserving analogue similarity.

You are given a machine-checkable medicinal-chemistry rule card.

Starting SMILES:
CCN(CCN)C(C)CF

Allowed actions:
- REPAIR with a candidate SMILES satisfying all hard constraints.
- ABSTAIN only if no compliant analogue can be found under the stated preservation constraints.

Hard constraints:
1. polar_bounds: HBA between 1.0 and 12.0, HBD between 0.0 and 6.0, MW between 120.0 and 520.0, ROTB between 0.0 and 10.0, TPSA between 40.0 and 180.0, logP between -1.0 and 3.0.
2. pilot_similarity_to_start: Morgan Tanimoto similarity to the starting molecule at least 0.45.

Analogue preservation:
- Candidate Morgan Tanimoto similarity to the starting molecule must be at least 0.45.

```

### pilot_003_repair_near_miss

family: `repair_near_miss`
spec: `spec_v2_polar_bias_train`
starting_smiles: `CCN(CCN)CC(C)Cl`
failure_constraint_ids: `['polar_bounds']`

```text
Task: Repair the starting molecule while preserving analogue similarity.

You are given a machine-checkable medicinal-chemistry rule card.

Starting SMILES:
CCN(CCN)CC(C)Cl

Allowed actions:
- REPAIR with a candidate SMILES satisfying all hard constraints.
- ABSTAIN only if no compliant analogue can be found under the stated preservation constraints.

Hard constraints:
1. polar_bounds: HBA between 1.0 and 12.0, HBD between 0.0 and 6.0, MW between 120.0 and 520.0, ROTB between 0.0 and 10.0, TPSA between 40.0 and 180.0, logP between -1.0 and 3.0.
2. pilot_similarity_to_start: Morgan Tanimoto similarity to the starting molecule at least 0.45.

Analogue preservation:
- Candidate Morgan Tanimoto similarity to the starting molecule must be at least 0.45.

```

### pilot_004_repair_near_miss

family: `repair_near_miss`
spec: `spec_v2_polar_bias_train`
starting_smiles: `CCN(CCN)CC(C)F`
failure_constraint_ids: `['polar_bounds']`

```text
Task: Repair the starting molecule while preserving analogue similarity.

You are given a machine-checkable medicinal-chemistry rule card.

Starting SMILES:
CCN(CCN)CC(C)F

Allowed actions:
- REPAIR with a candidate SMILES satisfying all hard constraints.
- ABSTAIN only if no compliant analogue can be found under the stated preservation constraints.

Hard constraints:
1. polar_bounds: HBA between 1.0 and 12.0, HBD between 0.0 and 6.0, MW between 120.0 and 520.0, ROTB between 0.0 and 10.0, TPSA between 40.0 and 180.0, logP between -1.0 and 3.0.
2. pilot_similarity_to_start: Morgan Tanimoto similarity to the starting molecule at least 0.45.

Analogue preservation:
- Candidate Morgan Tanimoto similarity to the starting molecule must be at least 0.45.

```

### pilot_005_repair_near_miss

family: `repair_near_miss`
spec: `spec_v2_polar_bias_train`
starting_smiles: `CCN(CC)C(CN)CCl`
failure_constraint_ids: `['polar_bounds']`

```text
Task: Repair the starting molecule while preserving analogue similarity.

You are given a machine-checkable medicinal-chemistry rule card.

Starting SMILES:
CCN(CC)C(CN)CCl

Allowed actions:
- REPAIR with a candidate SMILES satisfying all hard constraints.
- ABSTAIN only if no compliant analogue can be found under the stated preservation constraints.

Hard constraints:
1. polar_bounds: HBA between 1.0 and 12.0, HBD between 0.0 and 6.0, MW between 120.0 and 520.0, ROTB between 0.0 and 10.0, TPSA between 40.0 and 180.0, logP between -1.0 and 3.0.
2. pilot_similarity_to_start: Morgan Tanimoto similarity to the starting molecule at least 0.45.

Analogue preservation:
- Candidate Morgan Tanimoto similarity to the starting molecule must be at least 0.45.

```

## Best 3 Potentially Useful Tasks

These are tasks where nearest corpus retrieval failed but local mutation succeeded, which is closest to the desired signal.

- `pilot_016_repair_multi_violation` `repair_multi_violation` `spec_v2_cns_like_test` start `CCN(CCCl)C(C)CN` failures `['cns_bounds', 'pilot_secondary_property_repair']`; nearest failed, local mutation succeeded.
- `pilot_018_repair_multi_violation` `repair_multi_violation` `spec_v2_cns_like_test` start `CCN(CCCl)CC(C)N` failures `['cns_bounds', 'pilot_secondary_property_repair']`; nearest failed, local mutation succeeded.
- `pilot_020_repair_multi_violation` `repair_multi_violation` `spec_v2_polar_bias_train` start `CCN(CCN)C(C)CCl` failures `['polar_bounds', 'pilot_secondary_property_repair']`; nearest failed, local mutation succeeded.

## Weakest 3 Tasks

These are likely too easy because nearest corpus retrieval succeeds with high similarity.

- `pilot_013_repair_multi_violation` `repair_multi_violation` `spec_v2_polar_bias_train` start `CCN(CC)CCN`; nearest corpus candidate `CCN(CCN)CCN` succeeds at similarity 0.867.
- `pilot_014_repair_multi_violation` `repair_multi_violation` `spec_v2_ro5_balanced_test` start `CCN(CC)CCF`; nearest corpus candidate `CCN(CCF)CCF` succeeds at similarity 0.867.
- `pilot_010_repair_near_miss` `repair_near_miss` `spec_v2_polar_bias_train` start `CCN(CCN)CCCl`; nearest corpus candidate `CCN(CCN)CCN` succeeds at similarity 0.737.

## Failure Taxonomy

### no_op_starting_molecule

```json
{
  "abstained": 4,
  "hard_constraint_failure": 36
}
```

### random_valid_corpus

```json
{
  "hard_constraint_failure": 38,
  "success": 2
}
```

### nearest_passing_corpus_by_similarity

```json
{
  "abstained": 7,
  "success": 33
}
```

### local_mutation_search

```json
{
  "success": 40
}
```

### verifier_guided_greedy

```json
{
  "success": 40
}
```

## Answers To Core Questions

1. Does constrained analogue repair look viable? Maybe, but not with this construction. The direction has a plausible signal in the nearest-corpus failures, but most tasks are too searchable.
2. Are tasks nontrivial for cheap deterministic baselines? No. Nearest corpus solves 82.5%; local mutation and verifier-guided greedy solve 100%.
3. Does nearest-corpus retrieval solve it? Mostly yes, and above the predefined trivialization threshold.
4. Does local mutation/search solve it? Yes, completely.
5. Do LLMs show any meaningful signal? Not tested because the cheap-baseline gate failed.
6. Are failures chemically/workflow meaningful, or just invalid SMILES/schema noise? Cheap-baseline failures are meaningful preservation/search failures, not schema noise. But there are too few of them.
7. Should we scale this into the main paper? Not yet.
8. What should change before scaling? Use scaffold/series holdout, exclude local-neighborhood witnesses from retrieval/search baselines, raise similarity/scaffold constraints selectively, require multi-step transformations not in the local mutation operator, and build tasks from paired analogue transformations rather than from the same mutation operator used as a baseline.

## Paths

- `experiments/repair_workflow_quick_pilot/tasks.jsonl`
- `experiments/repair_workflow_quick_pilot/validation_result.json`
- `experiments/repair_workflow_quick_pilot/baseline_results.json`
- `experiments/repair_workflow_quick_pilot/baseline_results.md`
- `experiments/repair_workflow_quick_pilot/*_rows.jsonl`
