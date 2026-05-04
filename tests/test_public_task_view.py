from __future__ import annotations

import json
from pathlib import Path

from specguard_chem.benchmark.effective_spec import build_effective_spec
from specguard_chem.config import SpecModel, TaskModel
from specguard_chem.runner.public_view import (
    build_public_adapter_request,
    render_adapter_request_prompt,
)
from specguard_chem.runner.runner import TaskRunner
from specguard_chem.utils import jsonio


def _spec_payload(release: Path, task: TaskModel) -> dict:
    spec = SpecModel.model_validate(jsonio.read_json(release / "specs" / f"{task.spec_id}.json"))
    return build_effective_spec(spec, task.task_constraints).model_dump(mode="json")


def test_public_task_view_excludes_oracle_and_identity_fields(v1_release: Path, v1_tasks: list[dict]) -> None:
    task = TaskModel.model_validate(next(row for row in v1_tasks if row["task_type"] == "audit_reject"))
    request = build_public_adapter_request(
        task=task,
        spec=_spec_payload(v1_release, task),
        round_index=1,
        tools=TaskRunner._tool_spec(task.protocol),
        failure_feedback=None,
        interrupt=None,
    )
    rendered = render_adapter_request_prompt(request)
    forbidden = [
        "expected_action",
        '"expected"',
        "oracle_type",
        "evidence",
        "witness",
        "proof",
        "certificate",
        "task_id",
        "bundle_id",
        "audit_accept",
        "audit_reject",
        "spec_split",
    ]
    for term in forbidden:
        assert term not in rendered
    assert request["task"]["visible_task_name"] == "candidate_audit"
    assert request["task"]["rendered_agent_input"]
    assert request["task"]["input"]


def test_public_task_view_is_stable_when_hidden_fields_change(v1_release: Path, v1_tasks: list[dict]) -> None:
    raw = dict(v1_tasks[0])
    task = TaskModel.model_validate(raw)
    changed = dict(raw)
    changed["expected_action"] = "REJECT"
    changed["expected"] = "FAIL"
    changed["evidence"] = {"feasible_witness_smiles": "C", "proof": "hidden"}
    changed["oracle_type"] = "feasible_witness"
    changed["task_id"] = "changed"
    changed["bundle_id"] = "changed_bundle"
    changed_task = TaskModel.model_validate(changed)
    spec_payload = _spec_payload(v1_release, task)

    def public_json(model_task: TaskModel) -> str:
        request = build_public_adapter_request(
            task=model_task,
            spec=spec_payload,
            round_index=1,
            tools=TaskRunner._tool_spec(model_task.protocol),
            failure_feedback=None,
            interrupt=None,
        )
        return json.dumps(request, sort_keys=True)

    assert public_json(task) == public_json(changed_task)
