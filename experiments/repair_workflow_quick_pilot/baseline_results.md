# Repair Workflow Pilot Baseline Results

| baseline | n | success@1 | hard_pass_rate | similarity_preservation_rate | scaffold_preservation_rate | valid_smiles_rate | no_op_rate | unrelated_valid_molecule_rate | median_similarity_on_successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_op_starting_molecule | 40 | 0.000 | 0.000 | 1.000 | 1.000 | 0.900 | 1.000 | 0.000 | NA |
| random_valid_corpus | 40 | 0.050 | 0.050 | 0.000 | 0.825 | 1.000 | 0.000 | 0.875 | NA |
| nearest_passing_corpus_by_similarity | 40 | 0.825 | 0.825 | 0.806 | 0.975 | 0.825 | 0.000 | 0.000 | 0.654 |
| local_mutation_search | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.783 |
| verifier_guided_greedy | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.783 |

## Failure Counts

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
