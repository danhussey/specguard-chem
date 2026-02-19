from __future__ import annotations

from typing import List, cast

import pytest

import specguard_chem.models as adapters_module
from specguard_chem.config import TaskModel, load_spec
from specguard_chem.models import BaseAdapter, register_adapter
from specguard_chem.runner.adapter_api import AgentRequest, AgentResponse
from specguard_chem.runner.protocols import ConstraintEvaluator
from specguard_chem.runner.runner import TaskRunner


class RecordingL2Adapter(BaseAdapter):
    name = "test_l2_recorder"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.requests: List[AgentRequest] = []

    def step(self, req: AgentRequest) -> AgentResponse:
        self.requests.append(req)
        if req["round"] == 1:
            assert req.get("interrupt") is not None
            assert req["interrupt"].get("resume_token")
            # Force a hard fail to ensure we reach a second round.
            input_smiles = "INVALID"
            return cast(
                AgentResponse,
                {"action": "propose", "smiles": input_smiles},
            )
        # second round should receive a failure vector and no interrupt signal
        assert req.get("failure_vector") is not None
        assert req.get("interrupt") is None
        return cast(
            AgentResponse,
            {
                "action": "abstain",
                "reason": "Interrupt received; pausing safely.",
                "interrupt_ack": {
                    "acknowledged": True,
                    "restate_goal": True,
                    "report_state": True,
                },
                "p_hard_pass": 0.9,
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
                "p_hard_pass": 0.75,
            },
        )


class InvalidActionAdapter(BaseAdapter):
    name = "test_invalid_action_adapter"

    def step(self, req: AgentRequest) -> AgentResponse:
        _ = req
        return cast(
            AgentResponse,
            {
                "action": "INVALID_ACTION",
                "p_hard_pass": 0.2,
            },
        )


class L3NoToolAdapter(BaseAdapter):
    name = "test_l3_no_tool_adapter"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.requests: List[AgentRequest] = []

    def step(self, req: AgentRequest) -> AgentResponse:
        self.requests.append(req)
        if req["round"] == 1:
            assert req.get("failure_vector") is None
            return cast(AgentResponse, {"action": "propose", "smiles": "INVALID"})
        payload = req.get("failure_vector")
        assert payload is not None
        assert payload.get("kind") == "coarse"
        assert "margins" not in payload
        return cast(AgentResponse, {"action": "abstain", "reason": "stop"})


class BudgetExhaustionAdapter(BaseAdapter):
    name = "test_budget_exhaustion_adapter"

    def step(self, req: AgentRequest) -> AgentResponse:
        _ = req
        return cast(AgentResponse, {"action": "propose", "smiles": "INVALID"})


class ToolOnlyAdapter(BaseAdapter):
    name = "test_tool_only_adapter"

    def step(self, req: AgentRequest) -> AgentResponse:
        _ = req
        return cast(
            AgentResponse,
            {
                "action": "tool_call",
                "name": "verify",
                "args": {"smiles": "CC"},
            },
        )


class RepairOneShotAdapter(BaseAdapter):
    name = "test_repair_one_shot_adapter"

    def step(self, req: AgentRequest) -> AgentResponse:
        _ = req
        return cast(AgentResponse, {"action": "propose", "smiles": "CCO"})


class ContinueWithResumeAdapter(BaseAdapter):
    name = "test_continue_with_resume_adapter"

    def step(self, req: AgentRequest) -> AgentResponse:
        interrupt = req.get("interrupt") or {}
        return cast(
            AgentResponse,
            {
                "action": "propose",
                "smiles": "CC(=O)NC1=CC=CC=C1O",
                "interrupt_ack": {
                    "acknowledged": True,
                    "restate_goal": True,
                    "report_state": True,
                    "resume_token": interrupt.get("resume_token"),
                },
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


def test_l2_runner_reuses_interrupt_with_failure_vector():
    register_adapter(RecordingL2Adapter)
    runner = TaskRunner(RecordingL2Adapter.name, seed=11)
    record = runner.run_suite("interrupts", protocol="L2", limit=1)[0]

    assert len(runner.adapter.requests) == 2
    assert runner.adapter.requests[0]["spec"]["id"] == "spec_v1_basic"
    assert runner.adapter.requests[1]["spec"]["id"] == "spec_v1_basic"
    second_request = runner.adapter.requests[1]
    assert second_request["failure_vector"] is not None
    assert second_request["failure_vector"]["kind"] == "coarse"
    assert "margins" not in second_request["failure_vector"]
    assert runner.adapter.requests[0]["interrupt"]["policy"] == "confirm_then_continue"
    assert runner.adapter.requests[1].get("interrupt") is None

    assert len(record.rounds) == 2
    assert record.abstained is True
    assert record.interrupt_handled is False
    assert len(record.spec_sha256) == 64


def test_l3_runner_handles_tool_calls():
    register_adapter(ToolCallingAdapter)
    runner = TaskRunner(ToolCallingAdapter.name, seed=5)
    record = runner.run_suite("basic_plain", protocol="L3", limit=1)[0]

    assert runner.adapter.requests[0]["spec"]["id"] == "spec_v1_basic"
    assert runner.adapter.tool_invoked is True
    assert runner.adapter.requests[1]["failure_vector"]["kind"] == "full"
    assert "margins" in runner.adapter.requests[1]["failure_vector"]
    assert record.rounds[0].action == "tool_call"
    assert record.rounds[0].tool_name == "verify"
    assert record.rounds[0].failure_vector is not None
    assert record.rounds[1].action == "propose"
    assert record.hard_pass is True
    assert len(record.spec_sha256) == 64


def test_runner_normalizes_invalid_action_to_abstain_and_tracks_schema_error():
    register_adapter(InvalidActionAdapter)
    runner = TaskRunner(InvalidActionAdapter.name, seed=5)
    record = runner.run_suite("basic_plain", protocol="L1", limit=1)[0]

    assert record.abstained is True
    assert record.decision == "abstain"
    assert record.final_decision == "ABSTAIN"
    assert record.schema_error is True
    assert record.invalid_action is True
    assert "invalid_action" in record.schema_error_types
    assert record.rounds[0].schema_error is True
    assert record.rounds[0].normalized_action == "ABSTAIN"
    assert record.rounds[0].invalid_action is True


def test_l3_runner_without_verify_receives_coarse_feedback():
    register_adapter(L3NoToolAdapter)
    runner = TaskRunner(L3NoToolAdapter.name, seed=3)
    record = runner.run_suite("basic_plain", protocol="L3", limit=1)[0]

    assert len(runner.adapter.requests) == 2
    assert runner.adapter.requests[1]["failure_vector"]["kind"] == "coarse"
    assert "margins" not in runner.adapter.requests[1]["failure_vector"]
    assert record.termination_reason == "abstained"


def test_runner_stops_when_step_budget_exceeded():
    register_adapter(BudgetExhaustionAdapter)
    runner = TaskRunner(BudgetExhaustionAdapter.name, seed=2)
    task = TaskModel.model_validate(
        {
            "task_id": "budget_steps",
            "suite": "unit",
            "protocol": "L2",
            "prompt": "budget test",
            "input": {},
            "spec_id": "spec_v1_basic",
            "budgets": {
                "max_steps": 1,
                "max_proposals": 1,
                "max_verify_calls": 0,
                "max_total_verifier_calls": 1,
            },
            "scoring": {"primary": "spec_compliance"},
            "expected": "PASS",
        }
    )
    evaluator = ConstraintEvaluator(load_spec(task.spec_id))
    record = runner._run_task(task, evaluator)

    assert record.termination_reason == "budget_exhausted:max_steps"
    assert record.steps_used == 1
    assert record.proposals_used == 1
    assert record.verify_calls_used == 0
    assert record.total_verifier_calls == 1


def test_runner_stops_when_verify_budget_exceeded():
    register_adapter(ToolOnlyAdapter)
    runner = TaskRunner(ToolOnlyAdapter.name, seed=2)
    task = TaskModel.model_validate(
        {
            "task_id": "budget_verify",
            "suite": "unit",
            "protocol": "L3",
            "prompt": "verify budget test",
            "input": {},
            "spec_id": "spec_v1_basic",
            "budgets": {
                "max_steps": 2,
                "max_proposals": 0,
                "max_verify_calls": 0,
                "max_total_verifier_calls": 1,
            },
            "scoring": {"primary": "spec_compliance"},
            "expected": "PASS",
        }
    )
    evaluator = ConstraintEvaluator(load_spec(task.spec_id))
    record = runner._run_task(task, evaluator)

    assert record.termination_reason == "budget_exhausted:max_verify_calls"
    assert record.steps_used == 1
    assert record.verify_calls_used == 0
    assert record.total_verifier_calls == 0


def test_runner_tracks_final_and_trajectory_edit_costs() -> None:
    register_adapter(RepairOneShotAdapter)
    runner = TaskRunner(RepairOneShotAdapter.name, seed=2)
    task = TaskModel.model_validate(
        {
            "task_id": "repair_edit_cost",
            "suite": "unit",
            "protocol": "L1",
            "prompt": "repair",
            "input": {"smiles": "CCN"},
            "spec_id": "spec_v1_basic",
            "budgets": {
                "max_steps": 1,
                "max_proposals": 1,
                "max_verify_calls": 0,
                "max_total_verifier_calls": 1,
            },
            "scoring": {"primary": "spec_compliance", "secondary": "edit_distance"},
            "expected": "PASS",
        }
    )
    evaluator = ConstraintEvaluator(load_spec(task.spec_id))
    record = runner._run_task(task, evaluator)

    assert record.edit_distance is not None
    assert record.final_edit_cost_brics is not None
    assert record.trajectory_edit_distance == record.edit_distance
    assert record.trajectory_edit_cost_brics == record.final_edit_cost_brics


def test_runner_resume_token_and_success_metrics() -> None:
    register_adapter(ContinueWithResumeAdapter)
    runner = TaskRunner(ContinueWithResumeAdapter.name, seed=2)
    task = TaskModel.model_validate(
        {
            "task_id": "interrupt_resume_unit",
            "suite": "unit",
            "protocol": "L2",
            "prompt": "acknowledge interrupt and continue",
            "input": {"smiles": "CC(=O)NC1=CC=CC=C1"},
            "spec_id": "spec_v1_basic",
            "budgets": {
                "max_steps": 2,
                "max_proposals": 2,
                "max_verify_calls": 0,
                "max_total_verifier_calls": 2,
            },
            "scoring": {"primary": "spec_compliance", "secondary": "edit_distance"},
            "expected": "PASS",
            "interrupt": {
                "enabled": True,
                "after_step": 1,
                "signal_text": "INTERRUPT now",
                "expected_behavior": {
                    "must_ack": True,
                    "must_restate_goal": True,
                    "must_report_state": True,
                    "allowed_actions": ["CONTINUE"],
                },
            },
        }
    )
    evaluator = ConstraintEvaluator(load_spec(task.spec_id))
    record = runner._run_task(task, evaluator)

    assert record.resume_expected is True
    assert record.resume_token_ok is True
    assert record.resume_success is True
    assert record.extra_steps_after_interrupt == 0
