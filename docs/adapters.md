# Adapter Integration Guide

This guide explains how to plug an external agent into SpecGuard-Chem by implementing the adapter interface.

## Adapter interface recap

Adapters subclass `specguard_chem.models.BaseAdapter` and implement `step(req: AgentRequest) -> AgentResponse`.
- `AgentRequest` contains the task metadata, round index, optional failure vector, available tools, and interrupt signals.
- An `AgentResponse` must supply an `action` (`propose`, `tool_call`, or `abstain`). Optional fields include `smiles`, `confidence`, `reason`, etc.

## Options

### 1. Python subclasses

Create a subclass in `specguard_chem/models/your_adapter.py`, register it via `register_adapter`, and return deterministic responses. Copy the existing `heuristic` or `abstention_guard` implementations as references.

### 2. External process adapter

For agents implemented in another language or running behind a CLI, use the built-in [`ProcessAdapter`](../src/specguard_chem/models/process_adapter.py):

```bash
export SPEC_GUARD_PROCESS_ADAPTER_CMD="python path/to/agent_bridge.py"
specguard-chem run --suite basic --model process
```

The bridge script receives the request JSON on STDIN and must emit a single JSON object to STDOUT. Example handler:

```python
import json, sys

req = json.load(sys.stdin)
if req["round"] == 1:
    json.dump({"action": "tool_call", "name": "verify", "args": {"smiles": "CC"}}, sys.stdout)
else:
    json.dump({"action": "propose", "smiles": "CC(=O)NC1=CC=CC=C1O", "confidence": 0.7}, sys.stdout)
```

### 3. Dynamic registration

If you cannot modify the registry, call `register_adapter` during your harness bootstrap:

```python
from specguard_chem.models import register_adapter
from my_agent import MyAdapter

register_adapter(MyAdapter)
```

After registration you can refer to the adapter via its `name` attribute in CLI runs.

## Tips
- Always fill `confidence` to enable calibration metrics.
- Use failure vectors to adjust proposals in L2/L3 protocols.
- Respect interrupts by pausing/acknowledging when `req["interrupt"]` is present.

Refer to `tests/test_process_adapter.py` for a fully working example that exercises the `ProcessAdapter` end-to-end.
