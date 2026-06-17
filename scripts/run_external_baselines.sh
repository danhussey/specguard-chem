#!/usr/bin/env bash
set -euo pipefail

SCOPE="${SCOPE:-diagnostic}"
RELEASE="${RELEASE:-benchmarks/releases/sgchem_v1.0}"
SEED="${SEED:-7}"
MODEL_CONFIG="${MODEL_CONFIG:-external_baselines/config/external_models.yaml}"
DETERMINISTIC_BASELINES="${DETERMINISTIC_BASELINES:-external_baselines/config/deterministic_subset_baselines.yaml}"
EXTERNAL_PROVIDERS="${EXTERNAL_PROVIDERS:-}"
INTERFACE_TIERS="${INTERFACE_TIERS:-strict-tool-call}"
SKIP_NAMES="${SKIP_NAMES:-}"
ONLY_AVAILABLE_ENV="${ONLY_AVAILABLE_ENV:-1}"

if [[ "$SCOPE" == "full" ]]; then
  RESULTS="${RESULTS:-external_baselines/results_full_$(date -u +%Y_%m_%d_%H%M%S)}"
  N_BOOTSTRAP="${N_BOOTSTRAP:-400}"
  SUBSET=""
  MAX_PER_FAMILY="${MAX_PER_FAMILY:-}"
else
  RESULTS="${RESULTS:-external_baselines/results_diagnostic_$(date -u +%Y_%m_%d_%H%M%S)}"
  N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
  SUBSET="${SUBSET:-external_baselines/config/diagnostic_subset.json}"
  MAX_PER_FAMILY="${MAX_PER_FAMILY:-1}"
fi

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
SUBSET_ARGS=()

mkdir -p "$RESULTS"/{validation,raw_runs,cache,tables,notes}
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
  echo "scope: $SCOPE"
  echo "commit: $(git rev-parse HEAD)"
  echo "date_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "python: $("$PYTHON_BIN" --version 2>&1)"
  echo "providers: ${EXTERNAL_PROVIDERS:-available-env}"
  echo "interface_tiers: $INTERFACE_TIERS"
} > "$RESULTS/environment.txt"
log_cmd "record environment > $RESULTS/environment.txt"

run_logged_capture "$RESULTS/validation/validate_dataset_strict.log" \
  "$SGCHEM_BIN" validate-dataset --strict "$RELEASE"

if [[ "$SCOPE" != "full" ]]; then
  run_logged_capture "$RESULTS/validation/make_diagnostic_subset.log" \
    "$PYTHON_BIN" scripts/make_external_baseline_outputs.py \
      --release "$RELEASE" \
      --results "$RESULTS" \
      --subset-manifest "$SUBSET" \
      --generate-subset \
      --subset-only \
      --seed "$SEED" \
      --max-per-family "$MAX_PER_FAMILY"
  SUBSET_ARGS=(--subset-manifest "$SUBSET")

  run_logged_capture "$RESULTS/validation/run_benchmark_subset_offline.log" \
    "$SGCHEM_BIN" run-benchmark \
      --benchmark "$RELEASE" \
      --split test \
      "${SUBSET_ARGS[@]}" \
      --baselines "$DETERMINISTIC_BASELINES" \
      --out "$RESULTS/raw_runs/subset_offline" \
      --seed "$SEED" \
      --n-bootstrap "$N_BOOTSTRAP"
fi

BUILD_ARGS=(
  "$PYTHON_BIN" scripts/build_external_baselines.py
  --config "$MODEL_CONFIG"
  --out "$RESULTS/external_baselines.generated.yaml"
  --providers "$EXTERNAL_PROVIDERS"
  --interface-tiers "$INTERFACE_TIERS"
  --skip-names "$SKIP_NAMES"
)
if [[ "$ONLY_AVAILABLE_ENV" == "1" ]]; then
  BUILD_ARGS+=(--only-available-env)
fi
run_logged_capture "$RESULTS/validation/build_external_baselines.log" "${BUILD_ARGS[@]}"

run_logged_capture "$RESULTS/validation/preflight_external_models.log" \
  "${UV_PROVIDER[@]}" python scripts/preflight_external_models.py \
    --baselines "$RESULTS/external_baselines.generated.yaml" \
    --out "$RESULTS/validation/external_model_preflight.json"

LIVE_CMD=(
  "${UV_PROVIDER[@]}" specguard-chem run-benchmark
  --benchmark "$RELEASE"
  --split test
)
if (( ${#SUBSET_ARGS[@]} )); then
  LIVE_CMD+=("${SUBSET_ARGS[@]}")
fi
LIVE_CMD+=(
  --baselines "$RESULTS/external_baselines.generated.yaml"
  --out "$RESULTS/raw_runs/external_live"
  --seed "$SEED"
  --n-bootstrap "$N_BOOTSTRAP"
  --allow-external
  --cache-dir "$RESULTS/cache/external_live"
)
run_logged_capture "$RESULTS/validation/run_benchmark_external_live.log" "${LIVE_CMD[@]}"

REPLAY_CMD=(
  "${UV_PROVIDER[@]}" specguard-chem run-benchmark
  --benchmark "$RELEASE"
  --split test
)
if (( ${#SUBSET_ARGS[@]} )); then
  REPLAY_CMD+=("${SUBSET_ARGS[@]}")
fi
REPLAY_CMD+=(
  --baselines "$RESULTS/external_baselines.generated.yaml"
  --out "$RESULTS/raw_runs/external_replay"
  --seed "$SEED"
  --n-bootstrap "$N_BOOTSTRAP"
  --replay-cache "$RESULTS/cache/external_live"
)
run_logged_capture "$RESULTS/validation/run_benchmark_external_replay.log" "${REPLAY_CMD[@]}"

run_logged_capture "$RESULTS/validation/summarize_external_live.log" \
  "$PYTHON_BIN" scripts/summarize_external_baseline_run.py \
    --aggregate "$RESULTS/raw_runs/external_live/aggregate.json" \
    --cache-root "$RESULTS/cache/external_live" \
    --out "$RESULTS/tables/live"

run_logged_capture "$RESULTS/validation/summarize_external_replay.log" \
  "$PYTHON_BIN" scripts/summarize_external_baseline_run.py \
    --aggregate "$RESULTS/raw_runs/external_replay/aggregate.json" \
    --cache-root "$RESULTS/cache/external_live" \
    --out "$RESULTS/tables/replay"

if [[ "$SCOPE" != "full" ]]; then
  run_logged_capture "$RESULTS/validation/make_external_baseline_outputs.log" \
    "$PYTHON_BIN" scripts/make_external_baseline_outputs.py \
      --release "$RELEASE" \
      --results "$RESULTS" \
      --subset-manifest "$SUBSET" \
      --offline-aggregate "$RESULTS/raw_runs/subset_offline/aggregate.json" \
      --external-aggregate "$RESULTS/raw_runs/external_replay/aggregate.json" \
      --live-aggregate "$RESULTS/raw_runs/external_live/aggregate.json" \
      --external-cache-root "$RESULTS/cache/external_live"
fi

echo "$RESULTS"
