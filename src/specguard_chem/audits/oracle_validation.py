from __future__ import annotations

"""Oracle and certificate validation for sgchem_v1 tasks."""

from collections import defaultdict
from typing import Any, Callable, Mapping

from ..benchmark.effective_spec import build_effective_spec
from ..config import SpecModel, TaskConstraintsModel
from ..dataset.oracles import hard_violation_units
from ..runner.protocols import ConstraintEvaluator
from ..verifiers import canonicalize_smiles


def _counter() -> dict[str, int]:
    return {"checked": 0, "failed": 0}


def _effective_spec(task: Mapping[str, Any], spec_loader: Callable[[str], SpecModel]) -> SpecModel:
    spec = spec_loader(str(task.get("spec_id", "")))
    raw_constraints = task.get("task_constraints")
    task_constraints = (
        TaskConstraintsModel.model_validate(raw_constraints)
        if isinstance(raw_constraints, dict)
        else None
    )
    return build_effective_spec(spec, task_constraints)


def validate_oracles(
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    *,
    spec_loader: Callable[[str], SpecModel],
) -> dict[str, Any]:
    errors: list[str] = []
    counts = {
        "construct_witnesses": _counter(),
        "audit_accept": _counter(),
        "audit_reject": _counter(),
        "repair_near_miss": _counter(),
        "repair_multi_violation": _counter(),
        "abstain_certificates": _counter(),
        "boundary_groups": _counter(),
        "invariance_groups": _counter(),
    }
    boundary_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    invariance_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for _split, tasks in tasks_by_split.items():
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            task_type = str(task.get("task_type") or task.get("task_family") or "")
            evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
            input_block = task.get("input") if isinstance(task.get("input"), dict) else {}
            input_smiles = input_block.get("smiles") if isinstance(input_block.get("smiles"), str) else None
            spec = _effective_spec(task, spec_loader)
            evaluator = ConstraintEvaluator(spec, input_smiles=input_smiles)

            if task_type == "construct_feasible":
                counts["construct_witnesses"]["checked"] += 1
                witness = evidence.get("feasible_witness_smiles")
                if not isinstance(witness, str) or not evaluator.evaluate(witness).hard_pass:
                    counts["construct_witnesses"]["failed"] += 1
                    errors.append(f"{task_id}: construct witness does not hard-pass")
            elif task_type == "audit_accept":
                counts["audit_accept"]["checked"] += 1
                if not input_smiles or not evaluator.evaluate(input_smiles).hard_pass:
                    counts["audit_accept"]["failed"] += 1
                    errors.append(f"{task_id}: audit_accept candidate does not hard-pass")
            elif task_type == "audit_reject":
                counts["audit_reject"]["checked"] += 1
                if not input_smiles:
                    counts["audit_reject"]["failed"] += 1
                    errors.append(f"{task_id}: audit_reject missing candidate")
                else:
                    result = evaluator.evaluate(input_smiles)
                    if result.hard_pass or hard_violation_units(result) < 1:
                        counts["audit_reject"]["failed"] += 1
                        errors.append(f"{task_id}: audit_reject candidate does not hard-fail")
            elif task_type in {"repair_near_miss", "repair_multi_violation", "tool_forced_l3", "interrupt_resume"}:
                key = "repair_near_miss" if task_type == "repair_near_miss" else "repair_multi_violation"
                if task_type in {"tool_forced_l3", "interrupt_resume"}:
                    key = "repair_multi_violation"
                counts[key]["checked"] += 1
                witness = evidence.get("feasible_witness_smiles")
                if not input_smiles or not isinstance(witness, str):
                    counts[key]["failed"] += 1
                    errors.append(f"{task_id}: repair-style task missing input or witness")
                else:
                    input_result = evaluator.evaluate(input_smiles)
                    witness_result = evaluator.evaluate(witness)
                    units = hard_violation_units(input_result)
                    needs_exact_one = task_type == "repair_near_miss"
                    distinct_failing_constraints = set(
                        failure
                        for failure in (
                            outcome.constraint.id
                            for outcome in input_result.hard_outcomes
                            if not outcome.passed
                        )
                    )
                    if not input_result.valid:
                        counts[key]["failed"] += 1
                        errors.append(f"{task_id}: repair input is not valid SMILES")
                    elif input_result.hard_pass or not witness_result.hard_pass:
                        counts[key]["failed"] += 1
                        errors.append(f"{task_id}: repair input/witness oracle failed")
                    elif needs_exact_one and not (units == 1 or len(distinct_failing_constraints) == 1):
                        counts[key]["failed"] += 1
                        errors.append(f"{task_id}: near-miss input fails {units} hard units across {len(distinct_failing_constraints)} constraints")
                    elif task_type == "repair_multi_violation" and len(distinct_failing_constraints) < 2:
                        counts[key]["failed"] += 1
                        errors.append(f"{task_id}: multi-violation input fails {len(distinct_failing_constraints)} distinct hard constraints")
            elif task_type == "abstain_contradiction":
                counts["abstain_certificates"]["checked"] += 1
                certificate = evidence.get("unsat_certificate")
                if not isinstance(certificate, dict):
                    counts["abstain_certificates"]["failed"] += 1
                    errors.append(f"{task_id}: missing unsat_certificate")
                elif certificate.get("kind") != "explicit_constraint_contradiction":
                    counts["abstain_certificates"]["failed"] += 1
                    errors.append(f"{task_id}: unsat certificate is not explicit")
                elif not certificate.get("constraints") or not certificate.get("reason"):
                    counts["abstain_certificates"]["failed"] += 1
                    errors.append(f"{task_id}: unsat certificate lacks concrete constraints")
                else:
                    referenced = certificate.get("constraints")
                    visible_ids = {constraint.id for constraint in spec.constraints}
                    if not isinstance(referenced, list) or not set(map(str, referenced)).issubset(visible_ids):
                        counts["abstain_certificates"]["failed"] += 1
                        errors.append(
                            f"{task_id}: unsat certificate references constraints not visible in task/spec"
                        )

            boundary_group_id = evidence.get("boundary_group_id")
            if isinstance(boundary_group_id, str) and boundary_group_id:
                boundary_groups[boundary_group_id].append(task)
            invariance_group_id = evidence.get("invariance_group_id")
            if isinstance(invariance_group_id, str) and invariance_group_id:
                invariance_groups[invariance_group_id].append(task)

    for group_id, tasks in boundary_groups.items():
        counts["boundary_groups"]["checked"] += 1
        actions = {str(task.get("expected_action")) for task in tasks}
        if not {"ACCEPT", "REJECT"}.issubset(actions):
            counts["boundary_groups"]["failed"] += 1
            errors.append(f"boundary group {group_id}: missing pass/fail contrast")
            continue
        sample = tasks[0]
        evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
        pass_smiles = evidence.get("pass_smiles")
        fail_smiles = evidence.get("fail_smiles")
        spec = _effective_spec(sample, spec_loader)
        evaluator = ConstraintEvaluator(spec)
        if not isinstance(pass_smiles, str) or not isinstance(fail_smiles, str):
            counts["boundary_groups"]["failed"] += 1
            errors.append(f"boundary group {group_id}: missing pass/fail smiles")
            continue
        if not evaluator.evaluate(pass_smiles).hard_pass or evaluator.evaluate(fail_smiles).hard_pass:
            counts["boundary_groups"]["failed"] += 1
            errors.append(f"boundary group {group_id}: pass/fail verifier contrast failed")

    for group_id, tasks in invariance_groups.items():
        counts["invariance_groups"]["checked"] += 1
        sample = tasks[0]
        evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
        variants = evidence.get("variant_smiles")
        canonical = evidence.get("canonical_smiles") or evidence.get("invariance_canonical_smiles")
        if not isinstance(variants, list) or len(variants) < 2 or not isinstance(canonical, str):
            counts["invariance_groups"]["failed"] += 1
            errors.append(f"invariance group {group_id}: missing variants/canonical")
            continue
        variant_canonicals = [canonicalize_smiles(str(value)) for value in variants]
        if any(value is None for value in variant_canonicals) or len(set(variant_canonicals)) != 1:
            counts["invariance_groups"]["failed"] += 1
            errors.append(f"invariance group {group_id}: variants do not canonicalize together")
            continue
        spec = _effective_spec(sample, spec_loader)
        decisions = [ConstraintEvaluator(spec).evaluate(str(value)).hard_pass for value in variants]
        if len(set(decisions)) != 1:
            counts["invariance_groups"]["failed"] += 1
            errors.append(f"invariance group {group_id}: verifier decisions differ")

    return {
        "valid": len(errors) == 0,
        "num_errors": len(errors),
        "errors": errors,
        "counts": counts,
    }


def render_oracle_validation_report(summary: Mapping[str, Any]) -> str:
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    labels = [
        ("construct_witnesses", "construct witnesses"),
        ("audit_accept", "audit_accept"),
        ("audit_reject", "audit_reject"),
        ("repair_near_miss", "repair_near_miss"),
        ("repair_multi_violation", "repair_multi_violation"),
        ("abstain_certificates", "abstain certificates"),
        ("boundary_groups", "boundary groups"),
        ("invariance_groups", "invariance groups"),
    ]
    lines = ["# Oracle Validation Report", ""]
    for key, label in labels:
        row = counts.get(key, {}) if isinstance(counts.get(key), dict) else {}
        lines.append(f"{label} checked / failed: {row.get('checked', 0)} / {row.get('failed', 0)}")
    lines.append("")
    lines.append(f"total oracle validation errors: {summary.get('num_errors', 0)}")
    errors = summary.get("errors")
    if isinstance(errors, list) and errors:
        lines.extend(["", "Errors:"])
        for error in errors[:50]:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"
