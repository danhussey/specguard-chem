# External Baseline Interface Plan

This memo reframes the online work as an external-baseline interface study rather
than a model leaderboard. A model row is only interpretable if the output
contract is clearly specified and the adapter records interface failures
separately from task-action failures.

## Why The Old Snapshot Is Diagnostic

Known issues from the previous cached external snapshot:

- Anthropic Haiku often returned markdown-fenced JSON. The older parser treated
  that as invalid JSON and normalized it to abstention.
- OpenAI fast returned many empty cached responses. That should be counted as an
  interface/API-output failure mode, not as evidence of chemistry reasoning.
- A corrected Anthropic rerun was started but not completed because the account
  ran out of credit.
- The current revived branch initially lacked the external adapter source and
  preflight scripts from the MD result tag, so those must be ported before a new
  online run.

## Interface Tiers

Report every external row with an explicit interface tier:

| tier | mechanism | use |
| --- | --- | --- |
| prompt-json | natural-language instruction to emit JSON | weak baseline only |
| json-mode | provider enforces syntactic JSON | syntax-constrained baseline |
| strict-schema | provider enforces JSON Schema for final actions | preferred final-action baseline |
| strict-tool-call | provider enforces function/tool argument schema | preferred verifier/tool baseline |

Rows from different tiers should not be merged into one leaderboard.

## Provider Targets

- OpenAI: use strict JSON Schema structured outputs for final action responses;
  use function calling for verifier interactions.
- Anthropic: use strict tool use with input schemas. Avoid free-text JSON for
  the main external result.
- DeepSeek: use strict function calling when supported by the selected model. If
  only JSON Output is available, label the row as `json-mode` and keep it
  separate from strict-schema rows.

## Required Metadata

Each external run should record:

- provider, model id, model access date, and API base URL
- interface tier and provider feature used
- prompt hash and schema hash
- temperature, top_p, max output tokens, timeout
- live cache root, replay aggregate path, and cache/replay equality status
- raw output, parsed response, and parse/interface failure reason
- explicit counts for empty response, malformed JSON, schema violation, invalid
  action, unavailable tool call, and timeout

## Run Ladder

1. Local adapter unit tests with mocked provider responses.
2. One-request provider preflight per adapter/tier.
3. Two-task live smoke subset per provider.
4. Ten-task live diagnostic subset with replay equality check.
5. Eighty-task external subset only after the smoke gates are clean.
6. Full 266-task external test only if the 80-task run has no unexplained
   interface failures and budget is approved.

## Current Execution Update

The revived branch now has the external adapter code and run orchestration
needed for the strict-interface rerun:

- OpenAI-compatible adapters support `prompt-json`, `json-mode`,
  `strict-schema`, and `strict-tool-call`.
- Anthropic and DeepSeek adapters are registered and configured for
  `strict-tool-call`.
- The diagnostic subset is a 10-task stratified sample, one task per task
  family.
- The external matrix is generated under
  `external_baselines/results_diagnostic_2026_05_20_strict/external_baselines.generated.yaml`.
- The deterministic offline reference for that subset has been run under
  `external_baselines/results_diagnostic_2026_05_20_strict/raw_runs/subset_offline`.

Live provider execution is blocked pending explicit approval because it sends
benchmark/task contents to third-party APIs. After approval, the diagnostic run
should be executed before the full 266-task online run.

## Paper Rule

External rows can support the paper only if their interface tier is at least
`strict-schema` for final actions or `strict-tool-call` for tool interactions.
Lower-tier rows remain diagnostic evidence about interface brittleness.
