from __future__ import annotations

"""Oracle/certificate helpers for sgchem_v1 benchmark compilation."""

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping

from ..config import OracleTypeV1, SpecModel, TaskConstraintsModel
from ..runner.protocols import EvaluationResult

ORACLE_TYPES: tuple[str, ...] = (
    "feasible_witness",
    "repair_witness",
    "violation_certificate",
    "unsat_certificate",
    "equivalence_certificate",
    "boundary_certificate",
    "interrupt_certificate",
)

TASK_ORACLE_COMPATIBILITY: Mapping[str, set[str]] = {
    "construct_feasible": {"feasible_witness"},
    "repair_near_miss": {"repair_witness"},
    "repair_multi_violation": {"repair_witness"},
    "audit_accept": {"feasible_witness"},
    "audit_reject": {"violation_certificate"},
    "abstain_contradiction": {"unsat_certificate"},
    "boundary_precision": {"boundary_certificate"},
    "smiles_invariance": {"equivalence_certificate"},
    "interrupt_resume": {"interrupt_certificate"},
    "tool_forced_l3": {"repair_witness", "interrupt_certificate"},
}


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def spec_instance_hash(
    spec: SpecModel, task_constraints: TaskConstraintsModel | Mapping[str, Any] | None = None
) -> str:
    constraints_payload: Any
    if isinstance(task_constraints, TaskConstraintsModel):
        constraints_payload = task_constraints.model_dump(mode="json")
    else:
        constraints_payload = task_constraints
    return stable_json_hash(
        {
            "spec": spec.model_dump(mode="json"),
            "task_constraints": constraints_payload or {},
        }
    )


def failing_hard_constraints(result: EvaluationResult) -> list[str]:
    failures: list[str] = []
    for outcome in result.hard_outcomes:
        if outcome.passed:
            continue
        failures.append(outcome.constraint.id)
    return failures


def hard_violation_units(result: EvaluationResult) -> int:
    total = 0
    for outcome in result.hard_outcomes:
        if outcome.passed:
            continue
        property_details = outcome.info.get("property_details") or []
        if property_details:
            violated = sum(
                1
                for item in property_details
                if float(item.get("signed_margin", 0.0)) < 0.0
            )
            total += max(violated, 1)
        else:
            total += 1
    return total


def verifier_result_payload(result: EvaluationResult) -> Dict[str, Any]:
    return {
        "valid": bool(result.valid),
        "hard_pass": bool(result.hard_pass),
        "failing_constraints": failing_hard_constraints(result),
        "hard_violation_units": hard_violation_units(result),
        "canonical_smiles": result.canonical_smiles,
        "properties": dict(result.properties),
    }


def oracle_type_for_task(task_type: str) -> OracleTypeV1:
    if task_type in {"construct_feasible", "audit_accept"}:
        return "feasible_witness"
    if task_type in {"repair_near_miss", "repair_multi_violation", "tool_forced_l3"}:
        return "repair_witness"
    if task_type == "audit_reject":
        return "violation_certificate"
    if task_type == "abstain_contradiction":
        return "unsat_certificate"
    if task_type == "boundary_precision":
        return "boundary_certificate"
    if task_type == "smiles_invariance":
        return "equivalence_certificate"
    if task_type == "interrupt_resume":
        return "interrupt_certificate"
    raise ValueError(f"Unsupported task_type for oracle: {task_type}")


def oracle_is_compatible(task_type: str, oracle_type: str) -> bool:
    return oracle_type in TASK_ORACLE_COMPATIBILITY.get(task_type, set())


def first_property_bounds_constraint(spec: SpecModel) -> tuple[str, float, float, str] | None:
    for constraint in spec.constraints:
        if constraint.type != "hard" or constraint.check != "property_bounds":
            continue
        bounds = constraint.params.get("bounds")
        if not isinstance(bounds, dict):
            continue
        for prop, payload in sorted(bounds.items()):
            if not isinstance(payload, dict):
                continue
            if "min" not in payload or "max" not in payload:
                continue
            return str(prop), float(payload["min"]), float(payload["max"]), constraint.id
    return None


def all_group_ids(task: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
    for key in ("invariance_group_id", "boundary_group_id", "interrupt_group_id"):
        value = evidence.get(key)
        if isinstance(value, str) and value:
            yield key, value
