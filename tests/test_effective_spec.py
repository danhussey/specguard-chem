from __future__ import annotations

from typing import List, cast

import pytest

import specguard_chem.models as adapters_module
from specguard_chem.config import TaskModel
from specguard_chem.models import BaseAdapter, register_adapter
from specguard_chem.runner.adapter_api import AgentRequest, AgentResponse
from specguard_chem.runner.runner import TaskRunner


class CaptureEffectiveSpecAdapter(BaseAdapter):
    name = "capture_effective_spec"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.seen_specs: List[dict] = []

    def step(self, req: AgentRequest) -> AgentResponse:
        self.seen_specs.append(cast(dict, req["spec"]))
        return cast(AgentResponse, {"action": "abstain", "reason": "stop"})


@pytest.fixture(autouse=True)
def restore_adapters():
    original = adapters_module.available_adapters()
    try:
        yield
    finally:
        adapters_module._ADAPTERS.clear()  # type: ignore[attr-defined]
        adapters_module._ADAPTERS.update(original)  # type: ignore[attr-defined]


def test_runner_applies_task_constraints_into_effective_spec():
    register_adapter(CaptureEffectiveSpecAdapter)
    task = TaskModel.model_validate(
        {
            "task_id": "effective_spec_task",
            "suite": "unit",
            "protocol": "L1",
            "prompt": "test effective spec",
            "input": {"smiles": "CCO"},
            "spec_id": "spec_v1_basic",
            "scoring": {"primary": "spec_compliance"},
            "expected": "ABSTAIN",
            "task_constraints": {
                "additions": [
                    {
                        "id": "task_sa_cap",
                        "type": "hard",
                        "check": "sa_proxy_max",
                        "params": {"max": 8.0},
                    }
                ]
            },
        }
    )

    runner = TaskRunner(CaptureEffectiveSpecAdapter.name, seed=7)
    record = runner.run_tasks([task], suite="unit", protocol="L1")[0]

    assert len(runner.adapter.seen_specs) == 1
    spec_payload = runner.adapter.seen_specs[0]
    ids = {constraint["id"] for constraint in spec_payload["constraints"]}
    assert "task_sa_cap" in ids
    assert len(record.effective_spec_sha256) == 64
    assert len(record.spec_sha256) == 64
    assert record.spec_sha256 != record.effective_spec_sha256
