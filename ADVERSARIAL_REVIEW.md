# Adversarial Review — SpecGuard-Chem

This review intentionally applies a hostile quality bar: assume the code will be used incorrectly, attacked through edge cases, and judged by production reliability standards rather than demo tolerance.

## Executive verdict

The project reads like a promising prototype but behaves like an under-defended research script. The architecture advertises protocol rigor, yet key execution paths quietly discard signal, accept malformed behavior, and produce misleading metrics. In its current state, this is not a trustworthy evaluation harness.

## Critical findings

### 1) Protocol logic drops evaluation state after `tool_call`

In `TaskRunner._run_task`, the `tool_call` branch evaluates SMILES and records a failure vector, but never updates `last_evaluation`, `final_smiles`, or `canonical_smiles`. It then `continue`s. If the final allowed round is a tool call, the run is marked as failed because `hard_pass` is computed from `last_evaluation`, which remains `None`.

**Why this is bad:** your protocol can observe a passing tool verification and still report rejection. This is not just noisy; it is logically wrong and poisons benchmark outputs.

## 2) Tool interface is effectively unenforced

The runner advertises tools only for L3 via `_tool_spec`, but when an adapter returns `action == "tool_call"`, the code does not verify that:
- a tool exists for the protocol,
- the tool name matches an allowed tool,
- required arguments are present and valid.

It accepts any `name` and any `args` payload and proceeds to evaluation.

**Why this is bad:** this undermines the entire concept of tool-gated behavior and allows adapters to bypass intended protocol boundaries.

### 3) `ProcessAdapter` command parsing is brittle and unsafe-by-construction

When the command is sourced from `SPEC_GUARD_PROCESS_ADAPTER_CMD`, it is parsed with plain `.split()`. That destroys quoted arguments and breaks legitimate paths containing spaces.

**Why this is bad:** adapter invocation becomes environment-fragile and fails in predictable real-world setups. This is avoidable with proper shell-like parsing (for example, `shlex.split`).

### 4) Metric function quietly dilutes violations

`hard_violation_rate` skips incrementing violations when `hard_pass` is missing in record-style inputs, but still divides by total record count.

**Why this is bad:** malformed/incomplete records reduce the measured violation rate instead of being rejected or excluded, producing optimistic artifacts.

### 5) `abstention_utility` silently truncates mismatched inputs

`abstention_utility` iterates with `zip(truths, decisions)`, which truncates to the shorter sequence.

**Why this is bad:** callers can accidentally drop evaluation rows without any warning; utility looks valid while reflecting incomplete data.

## Risk profile

- **Integrity risk:** benchmark metrics can be materially wrong due to state-loss and denominator issues.
- **Protocol risk:** L3 tool semantics are cosmetic unless enforcement is added.
- **Operational risk:** process adapter behavior depends on brittle env-var formatting.
- **Governance risk:** silent truncation/leniency normalizes corrupted inputs.

## Required remediation (minimum)

1. Fix runner state propagation in the `tool_call` branch and add tests for terminal-round tool calls.
2. Enforce tool policy strictly: validate allowed tools, arguments, and protocol compatibility.
3. Replace env command splitting with robust parsing and document escaping rules.
4. Make scoring functions fail fast on malformed/misaligned inputs.
5. Add negative tests that prove metrics reject partial data instead of smoothing it away.

## Final assessment

This codebase is close to useful, but currently too permissive and internally inconsistent for any claim of rigorous evaluation. Until protocol enforcement and metric integrity are hardened, reported performance should be treated as suspect.
