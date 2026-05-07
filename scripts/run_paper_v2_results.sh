#!/usr/bin/env bash
set -euo pipefail

RELEASE="${RELEASE:-benchmarks/releases/sgchem_v1.0}"
RESULTS="${RESULTS:-paper_v2/results}"
SEED="${SEED:-7}"
SPLITS="${SGCHEM_PAPER_V2_SPLITS:-test}"
N_BOOTSTRAP_SWEEP="${N_BOOTSTRAP_SWEEP:-400}"
N_BOOTSTRAP_TABLES="${N_BOOTSTRAP_TABLES:-2000}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
if [[ -x ".venv/bin/specguard-chem" ]]; then
  SGCHEM_BIN="${SGCHEM_BIN:-.venv/bin/specguard-chem}"
else
  SGCHEM_BIN="${SGCHEM_BIN:-specguard-chem}"
fi

mkdir -p "$RESULTS"/{validation,raw_runs,summaries,tables,figures,notes}
export MPLCONFIGDIR="$RESULTS/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
: > "$RESULTS/commands.log"

log_cmd() {
  printf '%s\n' "$*" >> "$RESULTS/commands.log"
}

run_logged() {
  log_cmd "$*"
  "$@"
}

run_logged_capture() {
  local output="$1"
  shift
  log_cmd "$* > $output 2>&1"
  "$@" > "$output" 2>&1
}

run_logged_capture_allow_failure() {
  local output="$1"
  shift
  log_cmd "$* > $output 2>&1 || true"
  "$@" > "$output" 2>&1 || true
}

run_logged git status --short
run_logged git rev-parse HEAD
run_logged "$PYTHON_BIN" --version
run_logged_capture_allow_failure "$RESULTS/validation/pip_show_specguard_chem.log" "$PYTHON_BIN" -m pip show specguard-chem
run_logged_capture "$RESULTS/validation/specguard_chem_help.log" "$SGCHEM_BIN" --help

{
  echo "commit: $(git rev-parse HEAD)"
  echo "date_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "python: $("$PYTHON_BIN" --version 2>&1)"
  echo
  "$PYTHON_BIN" -m pip freeze
} > "$RESULTS/environment.txt"
log_cmd "record environment > $RESULTS/environment.txt"

run_logged_capture "$RESULTS/validation/validate_dataset_strict.log" \
  "$SGCHEM_BIN" validate-dataset --strict "$RELEASE"
run_logged_capture "$RESULTS/validation/validate_croissant.log" \
  "$SGCHEM_BIN" validate-croissant "$RELEASE/croissant.json"

for split in $SPLITS; do
  run_logged_capture "$RESULTS/validation/run_benchmark_full_offline_${split}.log" \
    "$SGCHEM_BIN" run-benchmark \
      --benchmark "$RELEASE" \
      --split "$split" \
      --baselines baselines/paper_v2_full_offline_baselines.yaml \
      --out "$RESULTS/raw_runs/full_offline_${split}" \
      --seed "$SEED" \
      --n-bootstrap "$N_BOOTSTRAP_SWEEP"
done

run_logged_capture "$RESULTS/validation/run_benchmark_wrapper_ablation_test.log" \
  "$SGCHEM_BIN" run-benchmark \
    --benchmark "$RELEASE" \
    --split test \
    --baselines baselines/paper_v2_wrapper_ablation_baselines.yaml \
    --out "$RESULTS/raw_runs/wrapper_ablation_test" \
    --seed "$SEED" \
    --n-bootstrap "$N_BOOTSTRAP_SWEEP"

run_logged_capture "$RESULTS/validation/run_benchmark_protocol_ladder_test.log" \
  "$SGCHEM_BIN" run-benchmark \
    --benchmark "$RELEASE" \
    --split test \
    --baselines baselines/paper_v2_protocol_ladder_baselines.yaml \
    --out "$RESULTS/raw_runs/protocol_ladder_test" \
    --seed "$SEED" \
    --n-bootstrap "$N_BOOTSTRAP_SWEEP"

if [[ -f "$RELEASE/audits/task_inventory_summary.md" ]]; then
  log_cmd "collect $RELEASE/audits/task_inventory_summary.md > $RESULTS/validation/task_inventory_audit.log"
  cp "$RELEASE/audits/task_inventory_summary.md" "$RESULTS/validation/task_inventory_audit.log"
else
  run_logged_capture_allow_failure "$RESULTS/validation/task_inventory_audit.log" \
    "$PYTHON_BIN" scripts/audit_task_inventory.py --release "$RELEASE"
fi

if [[ -f "$RELEASE/audits/oracle_scrambling_report.md" ]]; then
  log_cmd "collect $RELEASE/audits/oracle_scrambling_report.md > $RESULTS/validation/oracle_scrambling_audit.log"
  cp "$RELEASE/audits/oracle_scrambling_report.md" "$RESULTS/validation/oracle_scrambling_audit.log"
else
  run_logged_capture_allow_failure "$RESULTS/validation/oracle_scrambling_audit.log" \
    "$PYTHON_BIN" scripts/audit_oracle_scrambling.py --release "$RELEASE" --baselines baselines/paper_v2_full_offline_baselines.yaml
fi

if [[ -f "$RELEASE/audits/model_prompt_leakage_report.md" ]]; then
  log_cmd "collect $RELEASE/audits/model_prompt_leakage_report.md > $RESULTS/validation/model_prompt_leakage_audit.log"
  cp "$RELEASE/audits/model_prompt_leakage_report.md" "$RESULTS/validation/model_prompt_leakage_audit.log"
else
  run_logged_capture_allow_failure "$RESULTS/validation/model_prompt_leakage_audit.log" \
    "$PYTHON_BIN" scripts/audit_model_prompt_leakage.py --release "$RELEASE"
fi

if [[ -f "$RELEASE/audits/neurips_ed_preflight_report.md" ]]; then
  log_cmd "collect $RELEASE/audits/neurips_ed_preflight_report.md > $RESULTS/validation/neurips_ed_preflight.log"
  cp "$RELEASE/audits/neurips_ed_preflight_report.md" "$RESULTS/validation/neurips_ed_preflight.log"
else
  run_logged_capture_allow_failure "$RESULTS/validation/neurips_ed_preflight.log" \
    "$PYTHON_BIN" scripts/preflight_neurips_ed_artifact.py --release "$RELEASE"
fi

run_logged_capture "$RESULTS/validation/make_paper_v2_tables_and_figures.log" \
  "$PYTHON_BIN" scripts/make_paper_v2_tables_and_figures.py \
    --release "$RELEASE" \
    --results "$RESULTS" \
    --n-bootstrap "$N_BOOTSTRAP_TABLES" \
    --splits "$(echo "$SPLITS" | tr ' ' ',')"

run_logged_capture_allow_failure "$RESULTS/validation/paper_v2_consistency_check.log" \
  "$PYTHON_BIN" scripts/check_paper_consistency.py \
    --release "$RELEASE" \
    --results "$RESULTS"
