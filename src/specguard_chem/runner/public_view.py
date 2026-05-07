from __future__ import annotations

"""Public adapter-facing task views.

The runner keeps raw ``TaskModel`` objects for scoring and audit records, but
model adapters receive only this sanitized view. Hidden oracle/certificate
fields never cross this boundary.
"""

import json
from copy import deepcopy
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from ..config import SpecModel, TaskModel
from .adapter_api import AgentRequest, ToolSpec


VISIBLE_TASK_NAMES: dict[str, str] = {
    "construct_feasible": "construct",
    "repair_near_miss": "repair",
    "repair_multi_violation": "repair",
    "audit_accept": "candidate_audit",
    "audit_reject": "candidate_audit",
    "abstain_contradiction": "feasibility_check",
    "boundary_precision": "boundary_audit",
    "smiles_invariance": "representation_invariance",
    "interrupt_resume": "repair",
    "tool_forced_l3": "repair",
}

HIDDEN_FIELD_NAMES: set[str] = {
    "expected",
    "expected_action",
    "oracle_type",
    "evidence",
    "witness",
    "feasible_witness_smiles",
    "feasible_witness_canonical_smiles",
    "proof",
    "unsat_certificate",
    "violation_certificate",
    "boundary_certificate",
    "equivalence_certificate",
    "split",
    "task_id",
    "bundle_id",
    "task_type",
    "task_family",
    "curation_status",
}


class PublicTaskView(BaseModel):
    """Sanitized payload consumed by non-oracle adapters and baselines."""

    model_config = ConfigDict(extra="forbid")

    rendered_agent_input: str
    visible_task_name: str
    input: dict[str, Any]
    spec: dict[str, Any]
    protocol: str
    budgets: dict[str, Any]
    allowed_actions: list[str]
    allowed_tools: list[str]
    round_index: int
    failure_feedback: dict[str, Any] | None = None
    interrupt: dict[str, Any] | None = None


def visible_task_name(task_type: str | None) -> str:
    return VISIBLE_TASK_NAMES.get(str(task_type or ""), "evaluation_task")


def public_spec_payload(spec: SpecModel | Mapping[str, Any]) -> dict[str, Any]:
    payload = (
        spec.model_dump(mode="json")
        if isinstance(spec, SpecModel)
        else dict(deepcopy(spec))
    )
    payload.pop("spec_split", None)
    return payload


def sanitize_failure_feedback(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    allowed = {
        "kind",
        "round",
        "hard_fail_ids",
        "soft_miss_ids",
        "parse_error_type",
        "hard_fails",
        "soft_misses",
        "margins",
        "constraint_results",
    }
    sanitized = {key: deepcopy(value) for key, value in payload.items() if key in allowed}
    if not sanitized:
        return None
    return sanitized


def sanitize_interrupt_payload(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    allowed = {"policy", "round", "signal_text", "resume_token"}
    sanitized = {key: deepcopy(value) for key, value in payload.items() if key in allowed}
    return sanitized or None


def _public_allowed_actions(task: TaskModel) -> list[str]:
    visible = task.agent_visible_payload if isinstance(task.agent_visible_payload, dict) else {}
    allowed = visible.get("allowed_actions")
    if isinstance(allowed, list) and all(isinstance(item, str) for item in allowed):
        return list(allowed)
    if task.task_type in {"audit_accept", "audit_reject", "boundary_precision", "smiles_invariance"}:
        return [
            "ACCEPT if the provided molecule satisfies every hard constraint.",
            "REJECT if the provided molecule violates any hard constraint.",
        ]
    if task.task_type == "abstain_contradiction":
        return [
            "ACCEPT only if a molecule can satisfy every hard constraint.",
            "ABSTAIN if the hard constraints are internally contradictory.",
        ]
    return [
        "ACCEPT by returning a molecule that satisfies every hard constraint.",
        "ABSTAIN if the task cannot be completed under the stated rules.",
    ]


def build_public_task_view(
    *,
    task: TaskModel,
    spec: SpecModel | Mapping[str, Any],
    round_index: int,
    tools: list[ToolSpec],
    failure_feedback: Mapping[str, Any] | None,
    interrupt: Mapping[str, Any] | None,
) -> PublicTaskView:
    input_payload = task.input.model_dump(mode="json", exclude_none=True)
    budgets = task.budgets.model_dump(mode="json") if task.budgets is not None else {}
    tool_names = [str(tool["name"]) for tool in tools if tool.get("name")]
    return PublicTaskView(
        rendered_agent_input=task.rendered_agent_input or task.prompt,
        visible_task_name=visible_task_name(task.task_type or task.task_family),
        input=input_payload,
        spec=public_spec_payload(spec),
        protocol=task.protocol,
        budgets=budgets,
        allowed_actions=_public_allowed_actions(task),
        allowed_tools=tool_names,
        round_index=round_index,
        failure_feedback=sanitize_failure_feedback(failure_feedback),
        interrupt=sanitize_interrupt_payload(interrupt),
    )


def build_public_adapter_request(
    *,
    task: TaskModel,
    spec: SpecModel | Mapping[str, Any],
    round_index: int,
    tools: list[ToolSpec],
    failure_feedback: Mapping[str, Any] | None,
    interrupt: Mapping[str, Any] | None,
) -> AgentRequest:
    view = build_public_task_view(
        task=task,
        spec=spec,
        round_index=round_index,
        tools=tools,
        failure_feedback=failure_feedback,
        interrupt=interrupt,
    )
    request: AgentRequest = {
        "task": view.model_dump(mode="json", exclude_none=True),
        "spec": view.spec,
        "round": round_index,
        "tools": tools,
        "failure_vector": view.failure_feedback,
    }
    if view.interrupt:
        request["interrupt"] = view.interrupt
    return request


def render_adapter_request_prompt(request: AgentRequest) -> str:
    """Return the canonical prompt-like payload visible to an adapter."""

    return json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def assert_no_public_view_hidden_keys(payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    for key in HIDDEN_FIELD_NAMES:
        if f'"{key}"' in rendered:
            raise AssertionError(f"public adapter request contains hidden key {key}")
