#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-.venv/bin/python}"
CLI="${CLI:-.venv/bin/specguard-chem}"
RESULTS="paper_v2/results"
RELEASE="benchmarks/releases/sgchem_v1.0"
COMMANDS="$RESULTS/commands.log"

mkdir -p "$RESULTS/validation" "$RESULTS/raw_runs" "$RESULTS/summaries" "$RESULTS/tables" "$RESULTS/figures" "$RESULTS/notes"
: > "$COMMANDS"

log_cmd() {
  printf '%s\n' "$*" >> "$COMMANDS"
}

run_logged() {
  local logfile="$1"
  shift
  log_cmd "$* > $logfile 2>&1"
  "$@" > "$logfile" 2>&1
}

run_if_missing() {
  local sentinel="$1"
  local logfile="$2"
  shift 2
  if [[ -f "$sentinel" ]]; then
    log_cmd "SKIP existing $sentinel"
    return 0
  fi
  run_logged "$logfile" "$@"
}

log_cmd "git status --short"
git status --short > "$RESULTS/validation/git_status_short.log" 2>&1
log_cmd "git rev-parse HEAD"
git rev-parse HEAD > "$RESULTS/validation/git_rev_parse_HEAD.log" 2>&1
log_cmd "$PY --version"
"$PY" --version > "$RESULTS/validation/python_version.log" 2>&1
log_cmd "$PY -m pip show specguard-chem"
"$PY" -m pip show specguard-chem > "$RESULTS/validation/pip_show_specguard_chem.log" 2>&1 || true
log_cmd "$CLI --help"
"$CLI" --help > "$RESULTS/validation/specguard_chem_help.log" 2>&1

{
  printf 'commit: '
  git rev-parse HEAD
  printf 'date_utc: '
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  printf 'python: '
  "$PY" --version 2>&1
  printf '\n'
  "$PY" -m pip freeze
} > "$RESULTS/environment.txt"
log_cmd "record environment -> $RESULTS/environment.txt"

run_logged "$RESULTS/validation/validate_dataset_strict.log" \
  "$CLI" validate-dataset "$RELEASE" --strict
run_logged "$RESULTS/validation/validate_croissant.log" \
  "$CLI" validate-croissant "$RELEASE/croissant.json"

for split in train dev test; do
  full_reports=$(find "$RESULTS/raw_runs/full_offline_${split}" -maxdepth 2 -name report.json 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$full_reports" -ge 13 ]]; then
    log_cmd "SKIP existing full_offline_${split} report set ($full_reports reports)"
  else
    run_if_missing "$RESULTS/raw_runs/full_offline_${split}/aggregate.json" \
      "$RESULTS/validation/run_full_offline_${split}.log" \
      "$CLI" run-benchmark \
        --benchmark "$RELEASE" \
        --split "$split" \
        --baselines baselines/paper_v2_full_offline_baselines.yaml \
        --out "$RESULTS/raw_runs/full_offline_${split}" \
        --seed 7 \
        --n-bootstrap 400
  fi
done

wrapper_reports=$(find "$RESULTS/raw_runs/wrapper_ablation_test" -maxdepth 2 -name report.json 2>/dev/null | wc -l | tr -d ' ')
if [[ "$wrapper_reports" -ge 10 ]]; then
  log_cmd "SKIP existing wrapper_ablation_test report set ($wrapper_reports reports)"
else
  run_if_missing "$RESULTS/raw_runs/wrapper_ablation_test/aggregate.json" \
    "$RESULTS/validation/run_wrapper_ablation_test.log" \
    "$CLI" run-benchmark \
      --benchmark "$RELEASE" \
      --split test \
      --baselines baselines/paper_v2_wrapper_ablation_baselines.yaml \
      --out "$RESULTS/raw_runs/wrapper_ablation_test" \
      --seed 7 \
      --n-bootstrap 400
fi

run_if_missing "$RESULTS/raw_runs/protocol_ladder_test/aggregate.json" \
  "$RESULTS/validation/run_protocol_ladder_test.log" \
  "$CLI" run-benchmark \
    --benchmark "$RELEASE" \
    --split test \
    --baselines baselines/paper_v2_protocol_ladder_baselines.yaml \
    --out "$RESULTS/raw_runs/protocol_ladder_test" \
    --seed 7 \
    --n-bootstrap 400

run_logged "$RESULTS/validation/task_inventory_audit.log" \
  "$PY" scripts/audit_task_inventory.py --release "$RELEASE"
run_logged "$RESULTS/validation/oracle_scrambling_audit.log" \
  "$PY" scripts/audit_oracle_scrambling.py --release "$RELEASE" --baselines baselines/paper_v2_full_offline_baselines.yaml
run_logged "$RESULTS/validation/model_prompt_leakage_audit.log" \
  "$PY" scripts/audit_model_prompt_leakage.py --release "$RELEASE"
run_logged "$RESULTS/validation/neurips_ed_preflight.log" \
  "$PY" scripts/preflight_neurips_ed_artifact.py --release "$RELEASE"

run_logged "$RESULTS/validation/make_paper_v2_tables_and_figures.log" \
  "$PY" scripts/make_paper_v2_tables_and_figures.py \
    --release "$RELEASE" \
    --results "$RESULTS" \
    --n-bootstrap 2000

run_logged "$RESULTS/validation/paper_v2_consistency_check.log" \
  "$PY" scripts/check_paper_v2_consistency.py \
    --release "$RELEASE" \
    --results "$RESULTS"
