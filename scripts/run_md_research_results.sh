#!/usr/bin/env bash
set -euo pipefail

RELEASE="${RELEASE:-benchmarks/releases/sgchem_v1.0}"
RESULTS="${RESULTS:-md_research/results}"
SEED="${SEED:-7}"
N_BOOTSTRAP="${N_BOOTSTRAP:-400}"
MAX_PER_FAMILY="${MAX_PER_FAMILY:-8}"
SUBSET="${SUBSET:-md_research/config/external_subset.json}"
MODEL_CONFIG="${MODEL_CONFIG:-md_research/config/external_models.yaml}"
DETERMINISTIC_BASELINES="${DETERMINISTIC_BASELINES:-md_research/config/deterministic_subset_baselines.yaml}"
EXTERNAL_PROVIDERS="${EXTERNAL_PROVIDERS:-}"

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
UV_PROVIDER=(uv run --extra providers)

mkdir -p "$RESULTS"/{validation,raw_runs,cache,tables,figures,notes}
export MPLCONFIGDIR="$RESULTS/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
: > "$RESULTS/commands.log"

log_cmd() {
  printf '%s\n' "$*" >> "$RESULTS/commands.log"
}

run_logged_capture() {
  local output="$1"
  shift
  log_cmd "$* > $output 2>&1"
  "$@" > "$output" 2>&1
}

{
  echo "commit: $(git rev-parse HEAD)"
  echo "date_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "python: $("$PYTHON_BIN" --version 2>&1)"
} > "$RESULTS/environment.txt"
log_cmd "record environment > $RESULTS/environment.txt"

run_logged_capture "$RESULTS/validation/validate_dataset_strict.log" \
  "$SGCHEM_BIN" validate-dataset --strict "$RELEASE"

run_logged_capture "$RESULTS/validation/make_subset.log" \
  "$PYTHON_BIN" scripts/make_md_research_outputs.py \
    --release "$RELEASE" \
    --results "$RESULTS" \
    --subset-manifest "$SUBSET" \
    --generate-subset \
    --subset-only \
    --seed "$SEED" \
    --max-per-family "$MAX_PER_FAMILY"

run_logged_capture "$RESULTS/validation/run_benchmark_subset_offline.log" \
  "$SGCHEM_BIN" run-benchmark \
    --benchmark "$RELEASE" \
    --split test \
    --subset-manifest "$SUBSET" \
    --baselines "$DETERMINISTIC_BASELINES" \
    --out "$RESULTS/raw_runs/subset_offline" \
    --seed "$SEED" \
    --n-bootstrap "$N_BOOTSTRAP"

run_logged_capture "$RESULTS/validation/build_external_baselines.log" \
  "$PYTHON_BIN" scripts/build_md_external_baselines.py \
    --config "$MODEL_CONFIG" \
    --out "$RESULTS/external_baselines.generated.yaml" \
    --providers "$EXTERNAL_PROVIDERS"

run_logged_capture "$RESULTS/validation/preflight_external_models.log" \
  "${UV_PROVIDER[@]}" python scripts/preflight_external_models.py \
    --baselines "$RESULTS/external_baselines.generated.yaml" \
    --out "$RESULTS/validation/external_model_preflight.json"

run_logged_capture "$RESULTS/validation/run_benchmark_external_live.log" \
  "${UV_PROVIDER[@]}" specguard-chem run-benchmark \
    --benchmark "$RELEASE" \
    --split test \
    --subset-manifest "$SUBSET" \
    --baselines "$RESULTS/external_baselines.generated.yaml" \
    --out "$RESULTS/raw_runs/external_live" \
    --seed "$SEED" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --allow-external \
    --cache-dir "$RESULTS/cache/external_live"

run_logged_capture "$RESULTS/validation/run_benchmark_external_replay.log" \
  "${UV_PROVIDER[@]}" specguard-chem run-benchmark \
    --benchmark "$RELEASE" \
    --split test \
    --subset-manifest "$SUBSET" \
    --baselines "$RESULTS/external_baselines.generated.yaml" \
    --out "$RESULTS/raw_runs/external_replay" \
    --seed "$SEED" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --replay-cache "$RESULTS/cache/external_live"

run_logged_capture "$RESULTS/validation/make_md_research_outputs.log" \
  "$PYTHON_BIN" scripts/make_md_research_outputs.py \
    --release "$RELEASE" \
    --results "$RESULTS" \
    --subset-manifest "$SUBSET" \
    --offline-aggregate "$RESULTS/raw_runs/subset_offline/aggregate.json" \
    --external-aggregate "$RESULTS/raw_runs/external_replay/aggregate.json" \
    --live-aggregate "$RESULTS/raw_runs/external_live/aggregate.json" \
    --external-cache-root "$RESULTS/cache/external_live"
