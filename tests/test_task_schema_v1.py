from __future__ import annotations

import json
from pathlib import Path

from specguard_chem.config import TaskModel
from specguard_chem.dataset.oracles import oracle_is_compatible


def test_v1_tasks_satisfy_required_schema_fields(v1_tasks: list[dict]) -> None:
    schema = json.loads(Path("tasks/schema.json").read_text(encoding="utf-8"))
    required = set(schema["required"])
    allowed_actions = {"ACCEPT", "REJECT", "ABSTAIN"}
    allowed_types = set(schema["properties"]["task_type"]["enum"])
    for task in v1_tasks:
        TaskModel.model_validate(task)
        assert required.issubset(task)
        assert task["expected_action"] in allowed_actions
        assert task["task_type"] in allowed_types
        assert oracle_is_compatible(task["task_type"], task["oracle_type"])
        assert task.get("rendered_agent_input")
        assert task.get("prompt_template")
