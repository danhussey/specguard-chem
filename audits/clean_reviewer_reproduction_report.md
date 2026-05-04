# Clean Reviewer Reproduction Report

valid: true
archive: <ANONYMOUS_ARTIFACT_ARCHIVE>
temp_dir: <TEMP_REPRO_DIR>

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
warning: `VIRTUAL_ENV=<SOURCE_VENV>` does not match the project environment path `<TEMP_REPRO_DIR>/.venv` and will be ignored
Using CPython 3.11.11
Creating virtual environment at: <TEMP_REPRO_DIR>/.venv
Installed 48 packages in 983ms
........................................................................ [ 54%]
..............................................s.s...........             [100%]
130 passed, 2 skipped in 282.39s (0:04:42)
```

## Command 2

```text
        "test": 266,
        "train": 260
      }
    }
  },
  "errors": [],
  "num_bundles": 118,
  "num_errors": 0,
  "num_tasks": 656,
  "num_warnings": 0,
  "valid": true
}
```

## Command 3

```text
warning: `VIRTUAL_ENV=<SOURCE_VENV>` does not match the project environment path `<TEMP_REPRO_DIR>/.venv` and will be ignored
{
  "actual_model_prompts_checked": 656,
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
warning: `VIRTUAL_ENV=<SOURCE_VENV>` does not match the project environment path `<TEMP_REPRO_DIR>/.venv` and will be ignored
{
  "baseline_output_mismatches": [],
  "non_oracle_baseline_outputs_identical": true,
  "oracle_dependent_baselines": [],
  "prompt_mismatches": [],
  "public_views_identical_under_oracle_scrambling": true,
  "tasks_checked": 656,
  "valid": true
}
```

## Command 5

```text
│ always_… │ always_… │ primary_… │ mixed    │      266 │     0.000 │    1.000 │
│ always_… │ always_… │ primary_… │ mixed    │      266 │     0.000 │    0.000 │
│ random_… │ random_… │ primary_… │ mixed    │      266 │     0.147 │    0.796 │
│ schema_… │ schema_… │ primary_… │ mixed    │      266 │     0.707 │    0.293 │
│ heurist… │ heurist… │ primary_… │ mixed    │      266 │     0.406 │    0.576 │
│ abstent… │ abstent… │ primary_… │ mixed    │      266 │     0.680 │    0.290 │
│ local_m… │ local_m… │ primary_… │ mixed    │      266 │     0.846 │    0.154 │
│ verify_… │ verify_… │ tool_ena… │ mixed    │      266 │     0.846 │    0.154 │
│ verifie… │ verifie… │ tool_ena… │ mixed    │      266 │     0.699 │    0.301 │
│ corpus_… │ corpus_… │ retrieva… │ mixed    │      266 │     0.868 │    0.132 │
└──────────┴──────────┴───────────┴──────────┴──────────┴───────────┴──────────┘
Aggregate written to <LOCAL_TEMP_PATH>
```

## Command 6

```text
warning: `VIRTUAL_ENV=<SOURCE_VENV>` does not match the project environment path `<TEMP_REPRO_DIR>/.venv` and will be ignored
Figures: <LOCAL_TEMP_PATH>
Tables: <LOCAL_TEMP_PATH>
Summary: <LOCAL_TEMP_PATH>
```
