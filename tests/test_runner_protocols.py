from __future__ import annotations

from typing import List, cast

import pytest

import specguard_chem.models as adapters_module
from specguard_chem.models import BaseAdapter, register_adapter
from specguard_chem.runner.adapter_api import AgentRequest, AgentResponse
from specguard_chem.runner.runner import TaskRunner


class RecordingL2Adapter(BaseAdapter):
    name = "test_l2_recorder"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.requests: List[AgentRequest] = []

    def step(self, req: AgentRequest) -> AgentResponse:
        self.requests.append(req)
        if req["round"] == 1:
            # echo the starting SMILES, which is known to fail the spec
            input_smiles = (req["task"].get("input") or {}).get("smiles")
            return cast(
                AgentResponse,
                {"action": "propose", "smiles": input_smiles},
            )
        # second round should receive a failure vector and interrupt signal
        assert req.get("failure_vector") is not None
        assert req.get("interrupt") is not None
        return cast(
            AgentResponse,
            {
                "action": "propose",
                "smiles": "CC(=O)NC1=CC=CC=C1O",
                "confidence": 0.9,
            },
        )


class ToolCallingAdapter(BaseAdapter):
    name = "test_tool_adapter"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.requests: List[AgentRequest] = []
        self.tool_invoked = False

    def step(self, req: AgentRequest) -> AgentResponse:
        self.requests.append(req)
        if not self.tool_invoked:
            self.tool_invoked = True
            return cast(
                AgentResponse,
                {
                    "action": "tool_call",
                    "name": "verify",
                    "args": {"smiles": "CC"},
                },
            )
        assert req.get("failure_vector") is not None
        return cast(
            AgentResponse,
            {
                "action": "propose",
                "smiles": "CC(=O)NC1=CC=CC=C1O",
                "confidence": 0.75,
            },
        )


@pytest.fixture(autouse=True)
def restore_adapters():
    original = adapters_module.available_adapters()
    try:
        yield
    finally:
        adapters_module._ADAPTERS.clear()  # type: ignore[attr-defined]
        adapters_module._ADAPTERS.update(original)  # type: ignore[attr-defined]


def test_l2_runner_provides_failure_vector_and_interrupt():
    register_adapter(RecordingL2Adapter)
    runner = TaskRunner(RecordingL2Adapter.name, seed=11)
    record = runner.run_suite("basic", protocol="L2", limit=1)[0]

    assert len(runner.adapter.requests) == 2
    second_request = runner.adapter.requests[1]
    assert second_request["failure_vector"] is not None
    assert second_request["interrupt"]["policy"] == "confirm_then_continue"

    assert len(record.rounds) == 2
    assert record.hard_pass is True
    assert record.interrupt_handled is True


def test_l3_runner_handles_tool_calls():
    register_adapter(ToolCallingAdapter)
    runner = TaskRunner(ToolCallingAdapter.name, seed=5)
    record = runner.run_suite("basic", protocol="L3", limit=1)[0]

    assert runner.adapter.tool_invoked is True
    assert record.rounds[0].action == "tool_call"
    assert record.rounds[0].tool_name == "verify"
    assert record.rounds[0].failure_vector is not None
    assert record.rounds[1].action == "propose"
    assert record.hard_pass is True
