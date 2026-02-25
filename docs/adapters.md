# Adapter Integration Guide

This guide explains how to connect an external agent to SpecGuard-Chem.

## 1) Interface
Adapters subclass `specguard_chem.models.BaseAdapter` and implement:

- `step(req: AgentRequest) -> AgentResponse`

`AgentRequest` fields:
- `task`: full task payload
- `spec`: full resolved spec object (always present)
- `round`: 1-based round index
- `tools`: available tools for this protocol (`verify` only in L3)
- `failure_vector`: `None`, coarse feedback, or full feedback depending on protocol state
- `interrupt`: optional interrupt payload (includes `resume_token` for resume tasks)

`AgentResponse.action` must be one of:
- `propose`
- `tool_call`
- `abstain`

## 2) Feedback Semantics by Protocol
- `L1`: no feedback loop.
- `L2`: coarse feedback only.
- `L3`: coarse feedback on proposal rounds; full vector only after explicit `verify(smiles)` tool call.

## 3) Python Adapter Registration
Create `specguard_chem/models/your_adapter.py`, subclass `BaseAdapter`, then register via `register_adapter`.

```python
from specguard_chem.models import register_adapter
from my_agent import MyAdapter

register_adapter(MyAdapter)
```

After registration, use the adapter name with CLI `--model`.

## 4) External Process Adapter
Use built-in `ProcessAdapter` when your agent runs as a separate process/language.

```bash
export SPEC_GUARD_PROCESS_ADAPTER_CMD="python path/to/agent_bridge.py"
specguard-chem run basic_plain --model process
```

Bridge behavior:
- read one JSON request from `stdin`
- write one JSON response to `stdout`

Example:

```python
import json, sys

req = json.load(sys.stdin)

if req["tools"]:
    # L3 example: request a verifier call first
    json.dump(
        {"action": "tool_call", "name": "verify", "args": {"smiles": "CC"}},
        sys.stdout,
    )
else:
    json.dump(
        {"action": "propose", "smiles": "CC(=O)NC1=CC=CC=C1O", "p_hard_pass": 0.7},
        sys.stdout,
    )
```

## 5) OpenAI Adapter
```bash
pip install specguard-chem[providers]
export OPENAI_API_KEY=sk-...
specguard-chem run basic_plain --model openai_chat --limit 3
```

## 6) Interrupt and Resume Requirements
When `req["interrupt"]` is present:
- acknowledge interrupt
- restate goal if required
- report state if required
- if `resume_token` is present, echo it in `interrupt_ack.resume_token`
- choose an action allowed by `interrupt.expected_behavior.allowed_actions`

Example response fragment:

```json
{
  "action": "propose",
  "smiles": "CCO",
  "interrupt_ack": {
    "acknowledged": true,
    "restate_goal": true,
    "report_state": true,
    "resume_token": "<token-from-request>"
  }
}
```

## 7) Robustness Expectations
- Always emit valid JSON with a supported action.
- Provide `p_hard_pass` (`0..1`) for calibration/curve metrics.
- Handle schema normalization behavior: malformed outputs are converted to abstain and counted as schema/invalid-action/tool-call errors.

See `tests/test_process_adapter.py` for an end-to-end process-adapter example.
