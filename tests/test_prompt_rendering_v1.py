from __future__ import annotations

import re


def _allowed_actions(rendered: str) -> set[str]:
    actions: set[str] = set()
    in_section = False
    for line in rendered.splitlines():
        stripped = line.strip()
        if stripped == "Allowed actions:":
            in_section = True
            continue
        if in_section and not stripped:
            break
        if in_section and stripped.startswith("-"):
            label = stripped.lstrip("-").strip().split(maxsplit=1)[0].upper()
            if label in {"ACCEPT", "REJECT", "ABSTAIN"}:
                actions.add(label)
    return actions


def _schema_actions(rendered: str) -> set[str]:
    match = re.search(r'"action"\s*:\s*"([^"]+)"', rendered)
    assert match is not None
    return {part.strip() for part in match.group(1).split("|")}


def test_rendered_prompt_contains_required_visible_sections(v1_tasks: list[dict]) -> None:
    for task in v1_tasks:
        rendered = task["rendered_agent_input"]
        assert "Allowed actions:" in rendered
        assert "Hard constraints:" in rendered
        assert "Protocol:" in rendered
        assert "Budget:" in rendered
        assert "expected_action" not in rendered
        assert "witness" not in rendered.lower()
        assert "certificate" not in rendered.lower()
        assert "proof" not in rendered.lower()
        assert _schema_actions(rendered) == _allowed_actions(rendered)
        assert "instance_soft_window" not in rendered


def test_task_type_specific_prompt_visibility(v1_tasks: list[dict]) -> None:
    by_type = {task["task_type"]: task for task in v1_tasks}
    assert "Candidate SMILES:" in by_type["audit_accept"]["rendered_agent_input"]
    assert "Candidate SMILES:" in by_type["audit_reject"]["rendered_agent_input"]
    assert "Input molecule SMILES:" in by_type["repair_near_miss"]["rendered_agent_input"]
    assert "Candidate SMILES:" not in by_type["construct_feasible"]["rendered_agent_input"]
    abstain = by_type["abstain_contradiction"]["rendered_agent_input"]
    assert "ABSTAIN" in abstain
    assert "proof" not in abstain.lower()
    assert _schema_actions(by_type["audit_accept"]["rendered_agent_input"]) == {"ACCEPT", "REJECT"}
    assert _schema_actions(by_type["audit_reject"]["rendered_agent_input"]) == {"ACCEPT", "REJECT"}
    assert _schema_actions(by_type["construct_feasible"]["rendered_agent_input"]) == {"ACCEPT", "ABSTAIN"}
    assert _schema_actions(by_type["repair_near_miss"]["rendered_agent_input"]) == {"ACCEPT", "ABSTAIN"}
