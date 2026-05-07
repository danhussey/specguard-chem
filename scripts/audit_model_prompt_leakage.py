from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from specguard_chem.benchmark.effective_spec import build_effective_spec
from specguard_chem.config import SpecModel, TaskModel
from specguard_chem.dataset.validate_v1 import load_release_tasks_by_split
from specguard_chem.models.openai_adapter import OpenAIChatAdapter
from specguard_chem.runner.public_view import (
    build_public_adapter_request,
    render_adapter_request_prompt,
)
from specguard_chem.runner.runner import TaskRunner
from specguard_chem.utils import jsonio

_OPENAI_AUDIT_ADAPTER = OpenAIChatAdapter(client=object())

FORBIDDEN_PROMPT_TERMS: tuple[str, ...] = (
    "expected_action",
    '"expected"',
    "oracle_type",
    "evidence",
    "witness",
    "feasible_witness",
    "proof",
    "certificate",
    "unsat_certificate",
    "violation_certificate",
    "boundary_certificate",
    "equivalence_certificate",
    "hard_pass",
    "failing_constraints",
    "curation_status",
    "split",
    "task_id",
    "bundle_id",
    "audit_accept",
    "audit_reject",
)


def _spec_loader(release: Path):
    cache: dict[str, Any] = {}

    def load(spec_id: str):
        if spec_id not in cache:
            cache[spec_id] = jsonio.read_json(release / "specs" / f"{spec_id}.json")
        return cache[spec_id]

    return load


def _effective_spec_payload(task: TaskModel, release: Path) -> dict[str, Any]:
    raw_spec = _spec_loader(release)(task.spec_id)
    base_spec = SpecModel.model_validate(raw_spec)
    constraints = task.task_constraints
    return build_effective_spec(base_spec, constraints).model_dump(mode="json")


def _first_round_prompt(task_payload: Mapping[str, Any], release: Path) -> str:
    task = TaskModel.model_validate(task_payload)
    spec_payload = _effective_spec_payload(task, release)
    effective_spec = SpecModel.model_validate(spec_payload)
    tools = TaskRunner._tool_spec(task.protocol)
    interrupt = TaskRunner._interrupt_payload(task, effective_spec, 1)
    if interrupt:
        interrupt["resume_token"] = TaskRunner._resume_token(
            task=task,
            spec=effective_spec,
            round_index=1,
            steps_used=0,
            proposals_used=0,
            verify_calls_used=0,
            last_evaluation=None,
        )
    request = build_public_adapter_request(
        task=task,
        spec=spec_payload,
        round_index=1,
        tools=tools,
        failure_feedback=None,
        interrupt=interrupt,
    )
    openai_messages = _OPENAI_AUDIT_ADAPTER._build_prompt(request)
    return render_adapter_request_prompt(request) + "\n" + json.dumps(
        openai_messages, sort_keys=True, ensure_ascii=True
    )


def _hidden_literals(task: Mapping[str, Any]) -> list[str]:
    evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
    input_block = task.get("input") if isinstance(task.get("input"), dict) else {}
    visible_smiles = {
        value
        for value in (
            input_block.get("smiles"),
            input_block.get("candidate_smiles"),
        )
        if isinstance(value, str) and value
    }
    literals: list[str] = []
    for key in (
        "feasible_witness_smiles",
        "feasible_witness_canonical_smiles",
        "pass_smiles",
        "fail_smiles",
    ):
        value = evidence.get(key)
        if isinstance(value, str) and value and value not in visible_smiles:
            literals.append(value)
    for key in ("unsat_certificate", "contradiction_proof"):
        value = evidence.get(key)
        if isinstance(value, dict):
            reason = value.get("reason") or value.get("details")
            if isinstance(reason, str) and reason:
                literals.append(reason)
            constraints = value.get("constraints") or value.get("constraint_ids")
            if isinstance(constraints, list) and constraints:
                literals.append(json.dumps(constraints, sort_keys=True))
    failing = evidence.get("failing_constraints")
    if isinstance(failing, list) and failing:
        literals.append(json.dumps(failing, sort_keys=True))
    return literals


def audit_release_model_prompts(release: Path) -> dict[str, Any]:
    tasks_by_split = load_release_tasks_by_split(release)
    tasks = [task for rows in tasks_by_split.values() for task in rows]
    leaks: list[dict[str, Any]] = []
    literal_leaks: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        prompt = _first_round_prompt(task, release)
        for term in FORBIDDEN_PROMPT_TERMS:
            if term in prompt:
                leaks.append({"task_id": task_id, "term": term})
        for literal in _hidden_literals(task):
            if literal and literal in prompt:
                literal_leaks.append({"task_id": task_id, "literal": literal[:120]})
    label_leaks = [
        leak
        for leak in leaks
        if leak["term"] in {"task_id", "bundle_id", "split", "audit_accept", "audit_reject"}
    ]
    audit_accept_reject = [
        leak for leak in leaks if leak["term"] in {"audit_accept", "audit_reject"}
    ]
    return {
        "valid": not leaks and not literal_leaks,
        "actual_model_prompts_checked": len(tasks),
        "oracle_field_leaks": len(leaks),
        "literal_witness_leaks": len(literal_leaks),
        "label_leaks": len(label_leaks),
        "audit_accept_reject_name_leaks": len(audit_accept_reject),
        "leaks": leaks[:100],
        "literal_leaks": literal_leaks[:100],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Model Prompt Leakage Report",
        "",
        f"actual_model_prompts_checked: {summary.get('actual_model_prompts_checked', 0)}",
        f"oracle_field_leaks: {summary.get('oracle_field_leaks', 0)}",
        f"literal_witness_leaks: {summary.get('literal_witness_leaks', 0)}",
        f"label_leaks: {summary.get('label_leaks', 0)}",
        f"audit_accept_reject_name_leaks: {summary.get('audit_accept_reject_name_leaks', 0)}",
        f"valid: {str(bool(summary.get('valid'))).lower()}",
    ]
    leaks = summary.get("leaks") if isinstance(summary.get("leaks"), list) else []
    literal_leaks = summary.get("literal_leaks") if isinstance(summary.get("literal_leaks"), list) else []
    if leaks or literal_leaks:
        lines.extend(["", "Findings:"])
        for leak in leaks[:50]:
            lines.append(f"- {leak.get('task_id')}: forbidden term {leak.get('term')}")
        for leak in literal_leaks[:50]:
            lines.append(f"- {leak.get('task_id')}: hidden literal {leak.get('literal')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_release_model_prompts(args.release)
    out = args.release / "audits" / "model_prompt_leakage_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
