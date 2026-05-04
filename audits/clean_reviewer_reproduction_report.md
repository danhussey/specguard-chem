# Clean Reviewer Reproduction Report

valid: true
archive: sgchem_v1.0_anonymous_artifact.zip
temp_dir: /var/folders/cf/b2x7gmhd3p75s_swg_s896tm0000gn/T/sgchem_clean_repro_mg77d4ha

| command | returncode |
| --- | ---: |
| `uv run --extra dev pytest` | 0 |
| `uv run specguard-chem validate-dataset benchmarks/releases/sgchem_v1.0 --strict` | 0 |
| `uv run python scripts/audit_model_prompt_leakage.py --release benchmarks/releases/sgchem_v1.0` | 0 |
| `uv run python scripts/audit_oracle_scrambling.py --release benchmarks/releases/sgchem_v1.0` | 0 |
| `uv run specguard-chem run-benchmark --benchmark benchmarks/releases/sgchem_v1.0 --split test --baselines baselines/paper_baselines.yaml --out /tmp/sgchem_repro_run --seed 7` | 0 |
| `uv run specguard-chem paper-figures --runs /tmp/sgchem_repro_run --out /tmp/sgchem_repro_paper` | 0 |

Command output tails:

## Command 1

```text
warning: `VIRTUAL_ENV=/Users/danielhussey/.codex/worktrees/03db/specguard-chem/.venv` does not match the project environment path `/var/folders/cf/b2x7gmhd3p75s_swg_s896tm0000gn/T/sgchem_clean_repro_mg77d4ha/.venv` and will be ignored
Using CPython 3.11.11
Creating virtual environment at: /var/folders/cf/b2x7gmhd3p75s_swg_s896tm0000gn/T/sgchem_clean_repro_mg77d4ha/.venv
Installed 48 packages in 897ms
........................................................................ [ 56%]
.......................................................                  [100%]
127 passed in 277.76s (0:04:37)
```

## Command 2

```text
        "test": 244,
        "train": 319
      }
    }
  },
  "errors": [],
  "num_bundles": 120,
  "num_errors": 0,
  "num_tasks": 688,
  "num_warnings": 0,
  "valid": true
}
```

## Command 3

```text
warning: `VIRTUAL_ENV=/Users/danielhussey/.codex/worktrees/03db/specguard-chem/.venv` does not match the project environment path `/var/folders/cf/b2x7gmhd3p75s_swg_s896tm0000gn/T/sgchem_clean_repro_mg77d4ha/.venv` and will be ignored
{
  "actual_model_prompts_checked": 688,
  "audit_accept_reject_name_leaks": 0,
  "label_leaks": 0,
  "leaks": [],
  "literal_leaks": [],
  "literal_witness_leaks": 0,
  "oracle_field_leaks": 0,
  "valid": true
}
```

## Command 4

```text
warning: `VIRTUAL_ENV=/Users/danielhussey/.codex/worktrees/03db/specguard-chem/.venv` does not match the project environment path `/var/folders/cf/b2x7gmhd3p75s_swg_s896tm0000gn/T/sgchem_clean_repro_mg77d4ha/.venv` and will be ignored
{
  "baseline_output_mismatches": [],
  "non_oracle_baseline_outputs_identical": true,
  "oracle_dependent_baselines": [],
  "prompt_mismatches": [],
  "public_views_identical_under_oracle_scrambling": true,
  "tasks_checked": 688,
  "valid": true
}
```

## Command 5

```text
│ always_… │ always_… │ primary_… │ mixed    │      244 │     0.000 │    1.000 │
│ always_… │ always_… │ primary_… │ mixed    │      244 │     0.000 │    0.000 │
│ random_… │ random_… │ primary_… │ mixed    │      244 │     0.119 │    0.832 │
│ schema_… │ schema_… │ primary_… │ mixed    │      244 │     0.717 │    0.283 │
│ heurist… │ heurist… │ primary_… │ mixed    │      244 │     0.393 │    0.590 │
│ abstent… │ abstent… │ primary_… │ mixed    │      244 │     0.717 │    0.252 │
│ local_m… │ local_m… │ primary_… │ mixed    │      244 │     0.852 │    0.148 │
│ verify_… │ verify_… │ tool_ena… │ mixed    │      244 │     0.852 │    0.148 │
│ verifie… │ verifie… │ tool_ena… │ mixed    │      244 │     0.713 │    0.287 │
│ corpus_… │ corpus_… │ retrieva… │ mixed    │      244 │     0.852 │    0.148 │
└──────────┴──────────┴───────────┴──────────┴──────────┴───────────┴──────────┘
Aggregate written to /tmp/sgchem_repro_run/aggregate.json
```

## Command 6

```text
warning: `VIRTUAL_ENV=/Users/danielhussey/.codex/worktrees/03db/specguard-chem/.venv` does not match the project environment path `/var/folders/cf/b2x7gmhd3p75s_swg_s896tm0000gn/T/sgchem_clean_repro_mg77d4ha/.venv` and will be ignored
Figures: /tmp/sgchem_repro_paper/figures
Tables: /tmp/sgchem_repro_paper/tables
Summary: /tmp/sgchem_repro_paper/metrics_summary.md
```
