# External Baseline Runbook

This runbook makes the next online run an interface-tiered result set rather
than a loose provider leaderboard.

## Current Contract

- Main external tier: `strict-tool-call`.
- Output envelope: forced `specguard_action` function/tool call.
- Cached metadata records: provider, model id, interface tier, provider
  feature, prompt hash, schema hash, sampling config, timeout, and git commit.
- Interface failures are preserved as invalid/schema outputs instead of being
  silently converted to ordinary abstentions.

## Diagnostic Gate

Run a 10-task stratified diagnostic first:

```bash
SCOPE=diagnostic \
RESULTS=external_baselines/results_diagnostic_2026_05_20_strict_v3 \
N_BOOTSTRAP=20 \
ONLY_AVAILABLE_ENV=1 \
INTERFACE_TIERS=strict-tool-call \
scripts/run_external_baselines.sh
```

Pass criteria:

- provider preflight succeeds for every included baseline row;
- live run writes a cache for every baseline;
- replay from cache matches the live aggregate;
- `interface_error_step_rate == 0` for rows intended to support the paper;
- no unexplained empty raw outputs or missing forced tool calls.

## Full Online Gate

Only after the diagnostic gate passes:

```bash
SCOPE=full \
RESULTS=external_baselines/results_full_2026_05_20_strict_v3 \
N_BOOTSTRAP=400 \
ONLY_AVAILABLE_ENV=1 \
INTERFACE_TIERS=strict-tool-call \
scripts/run_external_baselines.sh
```

The full run evaluates all 266 held-out test tasks against the same generated
external baseline matrix, then immediately replays from cache. Treat the replay
aggregate, cache directory, generated baseline YAML, and summary tables as the
frozen online result set.

## Fallback Rule

If a provider rejects strict tool calling or repeatedly returns interface
failures, do not merge that row into the main online result table. Add a
separate `json-mode` or `prompt-json` diagnostic row and label it as an
interface failure/control, not a model-capability result.
