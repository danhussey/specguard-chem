from __future__ import annotations

"""Deterministic task-level effective-spec construction."""

from typing import Any, Dict

from ..config import SpecModel, TaskConstraintsModel


def build_effective_spec(
    base_spec: SpecModel,
    task_constraints: TaskConstraintsModel | None,
) -> SpecModel:
    """Merge optional task constraints into a base spec deterministically."""

    if task_constraints is None:
        return base_spec

    payload = base_spec.model_dump(mode="json")
    base_constraints = payload.get("constraints", [])
    if not isinstance(base_constraints, list):
        raise ValueError("Spec constraints payload is not a list")

    merged_constraints: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for constraint_payload in base_constraints:
        if not isinstance(constraint_payload, dict):
            raise ValueError("Constraint payload must be an object")
        merged = dict(constraint_payload)
        constraint_id = str(merged.get("id", "")).strip()
        if not constraint_id:
            raise ValueError("Constraint missing id in base spec")
        override = task_constraints.overrides.get(constraint_id)
        if override is not None:
            if override.check is not None:
                merged["check"] = override.check
            if override.type is not None:
                merged["type"] = override.type
            if override.severity is not None:
                merged["severity"] = override.severity
            if override.weight is not None:
                merged["weight"] = override.weight
            if override.params:
                params = dict(merged.get("params") or {})
                params.update(override.params)
                merged["params"] = params
        merged_constraints.append(merged)
        seen_ids.add(constraint_id)

    for addition in task_constraints.additions:
        addition_payload = addition.model_dump(mode="json")
        addition_id = str(addition_payload.get("id", "")).strip()
        if not addition_id:
            raise ValueError("Task constraint addition missing id")
        if addition_id in seen_ids:
            raise ValueError(
                f"Task constraint addition id '{addition_id}' already exists in base spec"
            )
        merged_constraints.append(addition_payload)
        seen_ids.add(addition_id)

    payload["constraints"] = merged_constraints
    return SpecModel.model_validate(payload)

