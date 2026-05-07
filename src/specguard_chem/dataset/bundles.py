from __future__ import annotations

"""Bundle-first oracle compiler for sgchem_v1 benchmark tasks."""

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..benchmark.effective_spec import build_effective_spec
from ..config import SpecModel, TaskBudgetsModel, TaskConstraintsModel, default_task_budgets
from ..runner.protocols import ConstraintEvaluator
from ..verifiers import canonicalize_smiles, compute_properties, parse_smiles
from .oracles import (
    failing_hard_constraints,
    first_property_bounds_constraint,
    hard_violation_units,
    oracle_type_for_task,
    spec_instance_hash,
    stable_json_hash,
    verifier_result_payload,
)
from .splits import assign_bundle_splits
from .tasks import _equivalent_smiles_forms, _pick_input_relative_witness, _spec_candidates


TASK_TYPES_V1: tuple[str, ...] = (
    "construct_feasible",
    "repair_near_miss",
    "repair_multi_violation",
    "audit_accept",
    "audit_reject",
    "abstain_contradiction",
    "boundary_precision",
    "smiles_invariance",
    "interrupt_resume",
    "tool_forced_l3",
)

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


class BundleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    suite: str
    split: Literal["train", "dev", "test"]
    seed: int
    source_molecule_id: str | None
    source_smiles: str | None
    source_canonical_smiles: str | None
    spec_id: str
    spec_instance_hash: str
    scaffold_hash: str | None
    oracle_summary: dict[str, Any]
    task_ids: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BundleCompilationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundles: list[BundleModel]
    tasks: list[dict[str, Any]]
    requested_bundles: int
    generated_bundles: int
    requested_tasks: int
    generated_tasks: int
    generation_shortfall: int
    generation_shortfall_reason: str | None
    split_policy: dict[str, Any]
    test_task_type_minimums: dict[str, int] = Field(default_factory=dict)
    test_task_type_counts: dict[str, int] = Field(default_factory=dict)
    test_task_type_minimums_met: bool = True
    diagnostic_only_test_task_types: list[str] = Field(default_factory=list)


def _short_hash(value: str) -> str:
    return stable_json_hash({"value": value}).split(":", 1)[1][:16]


def _constraint_line(constraint: Any) -> str:
    prefix = f"{constraint.id}: "
    if constraint.check == "property_bounds":
        bounds = constraint.params.get("bounds", {})
        parts = []
        for prop, payload in sorted(bounds.items()):
            parts.append(f"{prop} between {float(payload['min']):.3f} and {float(payload['max']):.3f}")
        return prefix + "; ".join(parts)
    if constraint.check == "alert_set_absent":
        return prefix + f"alert set absent: {constraint.params.get('alert_set')}"
    if constraint.check == "alert_set_present":
        return prefix + f"alert set present: {constraint.params.get('alert_set')}"
    if constraint.check == "substructure_present":
        return prefix + f"substructure present: {constraint.params.get('smarts_id')}"
    if constraint.check == "substructure_absent":
        return prefix + f"substructure absent: {constraint.params.get('smarts_id')}"
    if constraint.check == "sa_proxy_max":
        return prefix + f"SA proxy maximum {float(constraint.params.get('max', 0.0)):.3f}"
    if constraint.check == "similarity_min_to_input":
        return prefix + f"similarity to input at least {float(constraint.params.get('min', 0.0)):.3f}"
    if constraint.check == "equivalent_to_input":
        return prefix + f"equivalent to input using {constraint.params.get('policy')}"
    return prefix + str(constraint.check)


def _constraint_sections(spec: SpecModel) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    for constraint in spec.constraints:
        line = _constraint_line(constraint)
        if constraint.type == "hard":
            hard.append(line)
        else:
            soft.append(line + f" (weight {float(constraint.weight):.3f})")
    return hard, soft


def _allowed_actions(task_type: str) -> list[str]:
    if task_type in {"audit_accept", "audit_reject", "boundary_precision", "smiles_invariance"}:
        return [
            "ACCEPT if the provided molecule satisfies every hard constraint.",
            "REJECT if the provided molecule violates any hard constraint.",
        ]
    if task_type == "abstain_contradiction":
        return [
            "ACCEPT only if a molecule can satisfy every hard constraint.",
            "ABSTAIN if the hard constraints are internally contradictory.",
        ]
    return [
        "ACCEPT by returning a molecule that satisfies every hard constraint.",
        "ABSTAIN if the task cannot be completed under the stated rules.",
    ]


def _allowed_action_set(task_type: str) -> list[str]:
    actions: list[str] = []
    for action in _allowed_actions(task_type):
        label = action.split(maxsplit=1)[0].strip().upper()
        if label and label not in actions:
            actions.append(label)
    return actions


def _render_task(
    *,
    task_type: str,
    protocol: str,
    budgets: dict[str, Any],
    input_smiles: str | None,
    spec: SpecModel,
) -> tuple[str, str, dict[str, Any]]:
    hard, soft = _constraint_sections(spec)
    title_by_type = {
        "construct_feasible": "Task: Medicinal-chemistry molecule construction.",
        "repair_near_miss": "Task: Medicinal-chemistry near-miss repair.",
        "repair_multi_violation": "Task: Medicinal-chemistry multi-violation repair.",
        "audit_accept": "Task: Medicinal-chemistry specification audit.",
        "audit_reject": "Task: Medicinal-chemistry specification audit.",
        "abstain_contradiction": "Task: Medicinal-chemistry contradiction handling.",
        "boundary_precision": "Task: Medicinal-chemistry boundary precision audit.",
        "smiles_invariance": "Task: Medicinal-chemistry SMILES invariance audit.",
        "interrupt_resume": "Task: Medicinal-chemistry interrupted repair.",
        "tool_forced_l3": "Task: Medicinal-chemistry tool-assisted repair.",
    }
    template = f"{task_type}_v1"
    lines: list[str] = [
        title_by_type[task_type],
        "",
        "You are given a machine-checkable medicinal-chemistry specification.",
    ]
    if task_type.startswith("repair") or task_type in {"interrupt_resume", "tool_forced_l3"}:
        lines.append("Repair the input molecule so the final molecule satisfies all hard medicinal-chemistry constraints.")
    elif task_type == "construct_feasible":
        lines.append("Propose a valid molecule satisfying the stated medicinal-chemistry specification.")
    elif task_type == "abstain_contradiction":
        lines.append("Decide whether the stated hard constraints can all be satisfied together.")
    else:
        lines.append("Decide whether the provided molecule satisfies all hard medicinal-chemistry constraints.")
    lines.append("")
    lines.append("Allowed actions:")
    for action in _allowed_actions(task_type):
        lines.append(f"- {action}")
    if input_smiles:
        label = "Input molecule SMILES" if task_type.startswith("repair") or task_type in {"interrupt_resume", "tool_forced_l3"} else "Candidate SMILES"
        lines.extend(["", f"{label}:", input_smiles])
    lines.extend(["", "Hard constraints:"])
    for index, line in enumerate(hard, start=1):
        lines.append(f"{index}. {line}")
    lines.extend(["", "Soft preferences:"])
    if soft:
        for index, line in enumerate(soft, start=1):
            lines.append(f"{index}. {line}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            f"Protocol: {protocol}.",
            "Budget:",
            f"- max_steps: {budgets.get('max_steps')}",
            f"- max_proposals: {budgets.get('max_proposals')}",
            f"- max_verify_calls: {budgets.get('max_verify_calls')}",
            "",
            "Output schema:",
            '{"action": "' + "|".join(_allowed_action_set(task_type)) + '", "rationale": "...", "smiles": "..."}',
            "Include a SMILES value only when returning or accepting a molecule.",
        ]
    )
    if "ABSTAIN" in _allowed_action_set(task_type):
        lines.append("Abstention rule: abstain only for an internal contradiction or an explicit inability to comply with the visible hard constraints.")
    else:
        lines.append("Decision rule: choose ACCEPT or REJECT; abstention is not available for this task.")
    if protocol == "L3":
        lines.extend(["Verifier-tool availability: verify(smiles) may be used within the verify-call budget."])
    payload = {
        "visible_task_name": VISIBLE_TASK_NAMES.get(task_type, "evaluation_task"),
        "rendered_agent_input": "\n".join(lines),
        "input": {"smiles": input_smiles} if input_smiles else {},
        "spec_id": spec.id,
        "hard_constraints": hard,
        "soft_preferences": soft,
        "protocol": protocol,
        "budgets": budgets,
        "allowed_actions": _allowed_actions(task_type),
        "allowed_tools": ["verify"] if protocol == "L3" else [],
    }
    return template, payload["rendered_agent_input"], payload


def _task_budgets(protocol: str, *, interrupt: bool = False) -> dict[str, Any]:
    if interrupt:
        return {
            "max_steps": 3,
            "max_proposals": 2,
            "max_verify_calls": 1,
            "max_total_verifier_calls": 3,
        }
    return default_task_budgets(protocol).model_dump(mode="json")


def _difficulty_tags(
    *,
    task_type: str,
    protocol: str,
    spec: SpecModel,
    evidence: dict[str, Any],
) -> list[str]:
    tags: set[str] = set()
    if task_type == "boundary_precision":
        tags.add("tight_property_boundary")
    if task_type == "repair_multi_violation":
        tags.add("multi_constraint_violation")
    if task_type == "repair_near_miss":
        tags.add("minimal_edit_required")
    if task_type == "audit_reject":
        tags.add("reject_near_miss")
    if task_type == "abstain_contradiction":
        tags.add("abstain_explicit_contradiction")
    if task_type == "smiles_invariance":
        tags.add("invariance_equivalent_representation")
    if protocol == "L3" or task_type == "tool_forced_l3":
        tags.add("tool_required_by_protocol")
    if task_type == "interrupt_resume" or evidence.get("interrupt_group_id"):
        tags.add("interrupt_state_required")
    if any(constraint.check == "similarity_min_to_input" for constraint in spec.constraints):
        tags.add("high_similarity_guard")
    if any(constraint.type == "hard" for constraint in spec.constraints) and any(
        constraint.type == "soft" for constraint in spec.constraints
    ):
        tags.add("mixed_hard_soft_tradeoff")
    return sorted(tags)


def _make_task(
    *,
    benchmark_id: str,
    bundle_id: str,
    task_type: str,
    ordinal: int,
    spec: SpecModel,
    seed: int,
    source_record: dict[str, Any],
    protocol: str,
    expected_action: str,
    input_smiles: str | None,
    evidence: dict[str, Any],
    task_constraints: dict[str, Any] | None = None,
    budgets: dict[str, Any] | None = None,
    intentional_pair: bool = False,
    interrupt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_constraints = _with_contextual_soft_preference(task_constraints, source_record)
    task_constraints_model = (
        TaskConstraintsModel.model_validate(task_constraints)
        if task_constraints is not None
        else None
    )
    effective_spec = build_effective_spec(spec, task_constraints_model)
    budget_payload = budgets or _task_budgets(protocol)
    prompt_template, rendered, visible_payload = _render_task(
        task_type=task_type,
        protocol=protocol,
        budgets=budget_payload,
        input_smiles=input_smiles,
        spec=effective_spec,
    )
    oracle_type = evidence.get("oracle_type") or oracle_type_for_task(task_type)
    evidence = dict(evidence)
    evidence["oracle_type"] = oracle_type
    difficulty_tags = _difficulty_tags(
        task_type=task_type,
        protocol=protocol,
        spec=effective_spec,
        evidence=evidence,
    )
    generation = {
        "seed": seed,
        "stage": "bundle_compiler_v1",
        "source_molecule_id": str(source_record.get("molecule_id") or _short_hash(str(source_record.get("canonical_smiles", "")))),
        "source_canonical_smiles": str(source_record.get("canonical_smiles", "")),
        "scaffold_hash": str(source_record.get("scaffold_hash", "")) or None,
        "curation_status": "generated",
        "curation_reason": None,
        "difficulty_tags": difficulty_tags,
    }
    task_id = f"{bundle_id}__{task_type}__{ordinal:02d}"
    expected = "ABSTAIN" if expected_action == "ABSTAIN" else ("FAIL" if expected_action == "REJECT" else "PASS")
    task: dict[str, Any] = {
        "task_id": task_id,
        "suite": benchmark_id,
        "bundle_id": bundle_id,
        "task_type": task_type,
        "task_family": task_type,
        "protocol": protocol,
        "prompt": rendered,
        "prompt_template": prompt_template,
        "rendered_agent_input": rendered,
        "agent_visible_payload": visible_payload,
        "agent_visible_hash": stable_json_hash(visible_payload),
        "input": {"smiles": input_smiles} if input_smiles else {},
        "spec_id": spec.id,
        "spec_instance_hash": spec_instance_hash(spec, task_constraints_model),
        "scoring": {
            "primary": "spec_compliance",
            **({"secondary": "edit_distance"} if input_smiles and task_type not in {"audit_accept", "audit_reject", "boundary_precision", "smiles_invariance"} else {}),
        },
        "expected": expected,
        "expected_action": expected_action,
        "oracle_type": oracle_type,
        "evidence": evidence,
        "budgets": budget_payload,
        "generation": generation,
        "source_molecule_id": generation["source_molecule_id"],
        "source_smiles": str(source_record.get("canonical_smiles", "")),
        "source_canonical_smiles": str(source_record.get("canonical_smiles", "")),
        "generation_seed": seed,
        "generation_stage": "bundle_compiler_v1",
        "intentional_pair": intentional_pair,
        "difficulty_tags": difficulty_tags,
        "challenge_slice": any(tag != "mixed_hard_soft_tradeoff" for tag in difficulty_tags),
    }
    if task_constraints is not None:
        task["task_constraints"] = task_constraints
    if interrupt is not None:
        task["interrupt"] = interrupt
        task["evidence"]["interrupt"] = {
            "interrupt_after_step": interrupt.get("after_step", 1),
            "expected_resume_policy": "confirm_then_continue",
            "expected_state_fields": ["state", "report_state", "continue"],
        }
    return task


def _find_failing_smiles(
    spec: SpecModel, candidates: dict[str, list[dict[str, Any]]], *, multi: bool
) -> str | None:
    evaluator = ConstraintEvaluator(spec)
    pools = [candidates["multi_violation"], candidates["near_miss"]] if multi else [candidates["near_miss"], candidates["multi_violation"]]
    for pool in pools:
        for item in pool:
            smiles = str(item.get("smiles", ""))
            if not smiles:
                continue
            result = evaluator.evaluate(smiles)
            units = hard_violation_units(result)
            if multi and units >= 2:
                return smiles
            if not multi and units == 1:
                return smiles
    for fallback in ("C", "CCCCCCCCCCCCCCCC", "N"):
        result = evaluator.evaluate(fallback)
        units = hard_violation_units(result)
        if multi and units >= 2:
            return fallback
        if not multi and units == 1:
            return fallback
    return None


def _candidate_smiles_for_repair(candidates: dict[str, list[dict[str, Any]]], *, multi: bool) -> list[str]:
    pools = [candidates["multi_violation"], candidates["near_miss"]] if multi else [candidates["near_miss"], candidates["multi_violation"]]
    values: list[str] = []
    seen: set[str] = set()
    for pool in pools:
        for item in pool:
            smiles = str(item.get("smiles", ""))
            if smiles and smiles not in seen:
                values.append(smiles)
                seen.add(smiles)
    for fallback in ("C", "CCCCCCCCCCCCCCCC", "N"):
        if fallback not in seen:
            values.append(fallback)
            seen.add(fallback)
    return values


def _property_values(smiles: str) -> dict[str, float] | None:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return {key: float(value) for key, value in compute_properties(mol).items()}


def _contextual_soft_preference(source_record: dict[str, Any]) -> dict[str, Any] | None:
    props = _property_values(str(source_record.get("canonical_smiles", "")))
    if props is None:
        return None
    bounds: dict[str, dict[str, float]] = {}
    if "MW" in props:
        value = props["MW"]
        bounds["MW"] = {"min": max(0.0, value - 45.0), "max": value + 45.0}
    if "TPSA" in props:
        value = props["TPSA"]
        bounds["TPSA"] = {"min": max(0.0, value - 35.0), "max": value + 35.0}
    if not bounds:
        return None
    return {
        "id": "contextual_property_preference",
        "type": "soft",
        "check": "property_bounds",
        "params": {
            "mode": "all",
            "bounds": bounds,
        },
        "weight": 0.05,
    }


def _with_contextual_soft_preference(
    task_constraints: dict[str, Any] | None,
    source_record: dict[str, Any],
) -> dict[str, Any] | None:
    addition = _contextual_soft_preference(source_record)
    if addition is None:
        return task_constraints
    merged: dict[str, Any] = deepcopy(task_constraints) if task_constraints is not None else {}
    additions = merged.setdefault("additions", [])
    if not isinstance(additions, list):
        merged["additions"] = [addition]
        return merged
    if not any(isinstance(item, dict) and item.get("id") == addition["id"] for item in additions):
        additions.append(addition)
    return merged


def _repair_distinct_hard_constraint(input_smiles: str, witness_smiles: str) -> dict[str, Any] | None:
    input_props = _property_values(input_smiles)
    witness_props = _property_values(witness_smiles)
    if input_props is None or witness_props is None:
        return None
    widths = {
        "MW": 20.0,
        "TPSA": 15.0,
        "logP": 1.0,
        "HBA": 0.25,
        "HBD": 0.25,
        "ROTB": 0.25,
    }
    for prop in ("MW", "TPSA", "logP", "HBA", "HBD", "ROTB"):
        if prop not in input_props or prop not in witness_props:
            continue
        witness_value = float(witness_props[prop])
        input_value = float(input_props[prop])
        width = widths[prop]
        lower = witness_value - width
        upper = witness_value + width
        if lower <= input_value <= upper:
            continue
        return {
            "id": "repair_distinct_property_guard",
            "type": "hard",
            "check": "property_bounds",
            "params": {
                "mode": "all",
                "bounds": {prop: {"min": lower, "max": upper}},
            },
        }
    return None


def _repair_task(
    *,
    benchmark_id: str,
    bundle_id: str,
    ordinal: int,
    task_type: str,
    spec: SpecModel,
    seed: int,
    source_record: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    protocol: str,
) -> dict[str, Any] | None:
    witness = str(source_record["canonical_smiles"])
    input_smiles: str | None = None
    input_result = None
    witness_result = None
    task_constraints: dict[str, Any] | None = None
    if task_type == "repair_multi_violation":
        for candidate in _candidate_smiles_for_repair(candidates, multi=True):
            addition = _repair_distinct_hard_constraint(candidate, witness)
            if addition is None:
                continue
            candidate_constraints = {"additions": [addition]}
            effective_spec = build_effective_spec(spec, TaskConstraintsModel.model_validate(candidate_constraints))
            evaluator = ConstraintEvaluator(effective_spec)
            candidate_result = evaluator.evaluate(candidate)
            candidate_witness_result = evaluator.evaluate(witness)
            distinct_failures = set(failing_hard_constraints(candidate_result))
            if candidate_result.valid and not candidate_result.hard_pass and candidate_witness_result.hard_pass and len(distinct_failures) >= 2:
                input_smiles = candidate
                input_result = candidate_result
                witness_result = candidate_witness_result
                task_constraints = candidate_constraints
                break
        if input_smiles is None or input_result is None or witness_result is None:
            return None
    else:
        input_smiles = _find_failing_smiles(spec, candidates, multi=False)
        if not input_smiles:
            return None
        evaluator = ConstraintEvaluator(spec)
        input_result = evaluator.evaluate(input_smiles)
        witness_result = evaluator.evaluate(witness)
        distinct_failures = set(failing_hard_constraints(input_result))
        units = hard_violation_units(input_result)
        if not (units == 1 or len(distinct_failures) == 1):
            return None
        if not witness_result.hard_pass:
            return None
    distinct_failure_count = len(set(failing_hard_constraints(input_result)))
    evidence = {
        "oracle_type": "repair_witness",
        "input_verifier_result": verifier_result_payload(input_result),
        "feasible_witness_smiles": witness,
        "feasible_witness_canonical_smiles": witness_result.canonical_smiles,
        "witness_verifier_result": verifier_result_payload(witness_result),
        "expected_num_failing_constraints": 1 if task_type == "repair_near_miss" else distinct_failure_count,
    }
    return _make_task(
        benchmark_id=benchmark_id,
        bundle_id=bundle_id,
        task_type=task_type,
        ordinal=ordinal,
        spec=spec,
        seed=seed,
        source_record=source_record,
        protocol=protocol,
        expected_action="ACCEPT",
        input_smiles=input_smiles,
        evidence=evidence,
        task_constraints=task_constraints,
    )


def _contradiction_task_constraints(spec: SpecModel) -> tuple[dict[str, Any], dict[str, Any]] | None:
    source = first_property_bounds_constraint(spec)
    if source is None:
        return None
    prop, _lower, upper, constraint_id = source
    required_min = upper + 1.0
    addition_id = f"contradict_{prop.lower()}_minimum"
    task_constraints = {
        "additions": [
            {
                "id": addition_id,
                "type": "hard",
                "check": "property_bounds",
                "params": {
                    "mode": "all",
                    "bounds": {prop: {"min": required_min, "max": required_min + 1.0}},
                },
            }
        ]
    }
    certificate = {
        "kind": "explicit_constraint_contradiction",
        "constraints": [constraint_id, addition_id],
        "reason": f"requires {prop} <= {upper:.3f} and {prop} >= {required_min:.3f}",
        "property": prop,
        "spec_upper": upper,
        "required_min": required_min,
    }
    return task_constraints, certificate


def _boundary_tasks(
    *,
    benchmark_id: str,
    bundle_id: str,
    ordinal: int,
    spec: SpecModel,
    seed: int,
    source_record: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    pass_smiles = str(source_record["canonical_smiles"])
    fail_smiles = _find_failing_smiles(spec, candidates, multi=False) or _find_failing_smiles(spec, candidates, multi=True)
    if not fail_smiles:
        return []
    evaluator = ConstraintEvaluator(spec)
    pass_result = evaluator.evaluate(pass_smiles)
    fail_result = evaluator.evaluate(fail_smiles)
    if not pass_result.hard_pass or fail_result.hard_pass:
        return []
    property_name = next(iter(pass_result.properties.keys()), "MW")
    threshold = float(pass_result.properties.get(property_name, 0.0))
    group_id = f"{bundle_id}__boundary"
    evidence = {
        "oracle_type": "boundary_certificate",
        "boundary_group_id": group_id,
        "property": property_name,
        "boundary_property": property_name,
        "threshold": threshold,
        "pass_smiles": pass_smiles,
        "fail_smiles": fail_smiles,
        "pass_margin": 0.0,
        "fail_margin": -1.0,
        "pass_verifier_result": verifier_result_payload(pass_result),
        "fail_verifier_result": verifier_result_payload(fail_result),
    }
    return [
        _make_task(
            benchmark_id=benchmark_id,
            bundle_id=bundle_id,
            task_type="boundary_precision",
            ordinal=ordinal,
            spec=spec,
            seed=seed,
            source_record=source_record,
            protocol="L2",
            expected_action="ACCEPT",
            input_smiles=pass_smiles,
            evidence={**evidence, "boundary_role": "pass"},
            intentional_pair=True,
        ),
        _make_task(
            benchmark_id=benchmark_id,
            bundle_id=bundle_id,
            task_type="boundary_precision",
            ordinal=ordinal + 1,
            spec=spec,
            seed=seed,
            source_record=source_record,
            protocol="L2",
            expected_action="REJECT",
            input_smiles=fail_smiles,
            evidence={**evidence, "boundary_role": "fail"},
            intentional_pair=True,
        ),
    ]


def _invariance_tasks(
    *,
    benchmark_id: str,
    bundle_id: str,
    ordinal: int,
    spec: SpecModel,
    seed: int,
    source_record: dict[str, Any],
) -> list[dict[str, Any]]:
    canonical = str(source_record["canonical_smiles"])
    variants = _equivalent_smiles_forms(canonical)
    variants = [value for value in variants if canonicalize_smiles(value) == canonical]
    if len(variants) < 2:
        return []
    group_id = f"{bundle_id}__invariance"
    evaluator = ConstraintEvaluator(spec)
    decisions = [evaluator.evaluate(value).hard_pass for value in variants[:2]]
    if len(set(decisions)) != 1:
        return []
    evidence = {
        "oracle_type": "equivalence_certificate",
        "invariance_group_id": group_id,
        "canonical_smiles": canonical,
        "invariance_canonical_smiles": canonical,
        "variant_smiles": variants[:2],
        "expected_same_decision": True,
    }
    tasks: list[dict[str, Any]] = []
    for offset, variant in enumerate(variants[:2]):
        tasks.append(
            _make_task(
                benchmark_id=benchmark_id,
                bundle_id=bundle_id,
                task_type="smiles_invariance",
                ordinal=ordinal + offset,
                spec=spec,
                seed=seed,
                source_record=source_record,
                protocol="L1",
                expected_action="ACCEPT" if decisions[offset] else "REJECT",
                input_smiles=variant,
                evidence=evidence,
                intentional_pair=True,
            )
        )
    return tasks


def _interrupt_task(
    *,
    benchmark_id: str,
    bundle_id: str,
    ordinal: int,
    spec: SpecModel,
    seed: int,
    source_record: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    task_type: str,
) -> dict[str, Any] | None:
    input_smiles = _find_failing_smiles(spec, candidates, multi=False) or _find_failing_smiles(spec, candidates, multi=True)
    witness = str(source_record["canonical_smiles"])
    if not input_smiles:
        return None
    evaluator = ConstraintEvaluator(spec)
    input_result = evaluator.evaluate(input_smiles)
    witness_result = evaluator.evaluate(witness)
    if input_result.hard_pass or not witness_result.hard_pass:
        return None
    task_constraints: dict[str, Any] | None = None
    picked = _pick_input_relative_witness(
        passing_records=candidates["passing"],
        input_smiles=input_smiles,
        target_min_similarity=0.70,
    )
    if picked is not None:
        witness = picked[0]
        witness_result = evaluator.evaluate(witness)
    evidence = {
        "oracle_type": "interrupt_certificate" if task_type == "interrupt_resume" else "repair_witness",
        "input_verifier_result": verifier_result_payload(input_result),
        "feasible_witness_smiles": witness,
        "feasible_witness_canonical_smiles": witness_result.canonical_smiles,
        "witness_verifier_result": verifier_result_payload(witness_result),
        "expected_num_failing_constraints": max(1, hard_violation_units(input_result)),
    }
    interrupt_payload = None
    budgets = {
        "max_steps": 3,
        "max_proposals": 2,
        "max_verify_calls": 1,
        "max_total_verifier_calls": 3,
    }
    if task_type == "interrupt_resume":
        evidence["interrupt_group_id"] = f"{bundle_id}__interrupt"
        interrupt_payload = {
            "enabled": True,
            "after_step": 1,
            "signal_text": "Pause and confirm before continuing.",
            "expected_behavior": {
                "must_ack": True,
                "must_restate_goal": True,
                "must_report_state": True,
                "allowed_actions": ["CONTINUE"],
            },
        }
    if task_type == "tool_forced_l3":
        task_constraints = None
    return _make_task(
        benchmark_id=benchmark_id,
        bundle_id=bundle_id,
        task_type=task_type,
        ordinal=ordinal,
        spec=spec,
        seed=seed,
        source_record=source_record,
        protocol="L3",
        expected_action="ACCEPT",
        input_smiles=input_smiles,
        evidence=evidence,
        task_constraints=task_constraints,
        budgets=budgets,
        interrupt=interrupt_payload,
    )


def compile_bundles_from_corpus(
    *,
    corpus_records: Sequence[dict[str, Any]],
    specs: Sequence[SpecModel],
    benchmark_id: str,
    seed: int,
    target_bundles: int,
    min_tasks: int | None = None,
    max_tasks: int | None = None,
    min_test_task_type_counts: dict[str, int] | None = None,
) -> BundleCompilationResult:
    if target_bundles <= 0:
        raise ValueError("target_bundles must be positive")
    specs_sorted = sorted(specs, key=lambda spec: spec.id)
    if not specs_sorted:
        raise ValueError("No specs provided for bundle compilation")

    candidates_by_spec = {
        spec.id: _spec_candidates(
            spec=spec,
            corpus_records=corpus_records,
            near_miss_margin_band=5.0,
            boundary_margin_band=1.0,
        )
        for spec in specs_sorted
    }

    raw_bundles: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    per_spec_source_index: defaultdict[str, int] = defaultdict(int)
    slot_cycle = (
        "repair_near_miss",
        "repair_multi_violation",
        "boundary_precision",
        "smiles_invariance",
        "interrupt_resume",
        "tool_forced_l3",
    )

    for bundle_index in range(target_bundles):
        spec = specs_sorted[(bundle_index + seed) % len(specs_sorted)]
        candidates = candidates_by_spec[spec.id]
        passing = candidates["passing"]
        if not passing:
            skipped["no_passing_witness"] += 1
            continue
        source_index = per_spec_source_index[spec.id] % len(passing)
        per_spec_source_index[spec.id] += 1
        source = dict(passing[source_index])
        source.setdefault("molecule_id", _short_hash(str(source.get("canonical_smiles", ""))))
        bundle_id = f"{benchmark_id}__bundle__{bundle_index + 1:05d}"
        evaluator = ConstraintEvaluator(spec)
        witness = str(source["canonical_smiles"])
        witness_result = evaluator.evaluate(witness)
        if not witness_result.hard_pass:
            skipped["source_witness_failed"] += 1
            continue
        tasks: list[dict[str, Any]] = []

        construct_evidence = {
            "oracle_type": "feasible_witness",
            "feasible_witness_smiles": witness,
            "feasible_witness_canonical_smiles": witness_result.canonical_smiles,
            "witness_verifier_result": verifier_result_payload(witness_result),
        }
        tasks.append(
            _make_task(
                benchmark_id=benchmark_id,
                bundle_id=bundle_id,
                task_type="construct_feasible",
                ordinal=len(tasks) + 1,
                spec=spec,
                seed=seed,
                source_record=source,
                protocol=("L1", "L2", "L3")[bundle_index % 3],
                expected_action="ACCEPT",
                input_smiles=None,
                evidence=construct_evidence,
            )
        )
        tasks.append(
            _make_task(
                benchmark_id=benchmark_id,
                bundle_id=bundle_id,
                task_type="audit_accept",
                ordinal=len(tasks) + 1,
                spec=spec,
                seed=seed,
                source_record=source,
                protocol=("L1", "L2", "L3")[(bundle_index + 1) % 3],
                expected_action="ACCEPT",
                input_smiles=witness,
                evidence={
                    "oracle_type": "feasible_witness",
                    "feasible_witness_smiles": witness,
                    "feasible_witness_canonical_smiles": witness_result.canonical_smiles,
                    "candidate_verifier_result": verifier_result_payload(witness_result),
                    "witness_verifier_result": verifier_result_payload(witness_result),
                },
            )
        )

        failing_smiles = _find_failing_smiles(spec, candidates, multi=False) or _find_failing_smiles(spec, candidates, multi=True)
        if failing_smiles:
            failing_result = evaluator.evaluate(failing_smiles)
            tasks.append(
                _make_task(
                    benchmark_id=benchmark_id,
                    bundle_id=bundle_id,
                    task_type="audit_reject",
                    ordinal=len(tasks) + 1,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                    protocol=("L1", "L2", "L3")[(bundle_index + 2) % 3],
                    expected_action="REJECT",
                    input_smiles=failing_smiles,
                    evidence={
                        "oracle_type": "violation_certificate",
                        "candidate_verifier_result": verifier_result_payload(failing_result),
                        "failing_constraints": failing_hard_constraints(failing_result),
                    },
                )
            )
        else:
            skipped["audit_reject_no_failing_candidate"] += 1

        contradiction = _contradiction_task_constraints(spec)
        if contradiction is not None:
            task_constraints, certificate = contradiction
            tasks.append(
                _make_task(
                    benchmark_id=benchmark_id,
                    bundle_id=bundle_id,
                    task_type="abstain_contradiction",
                    ordinal=len(tasks) + 1,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                    protocol="L1",
                    expected_action="ABSTAIN",
                    input_smiles=None,
                    evidence={
                        "oracle_type": "unsat_certificate",
                        "unsat_certificate": certificate,
                        "contradiction_proof": {
                            "type": "bounds_contradiction",
                            "property": certificate["property"],
                            "spec_upper": certificate["spec_upper"],
                            "required_min": certificate["required_min"],
                            "constraint_ids": certificate["constraints"],
                            "details": certificate["reason"],
                        },
                    },
                    task_constraints=task_constraints,
                )
            )
        else:
            skipped["abstain_no_property_bounds"] += 1

        slot = slot_cycle[bundle_index % len(slot_cycle)]
        added_slot: list[dict[str, Any]] = []
        if slot in {"repair_near_miss", "repair_multi_violation"}:
            repair = _repair_task(
                benchmark_id=benchmark_id,
                bundle_id=bundle_id,
                ordinal=len(tasks) + 1,
                task_type=slot,
                spec=spec,
                seed=seed,
                source_record=source,
                candidates=candidates,
                protocol="L2" if slot == "repair_near_miss" else "L3",
            )
            if repair is not None:
                added_slot.append(repair)
        elif slot == "boundary_precision":
            added_slot.extend(
                _boundary_tasks(
                    benchmark_id=benchmark_id,
                    bundle_id=bundle_id,
                    ordinal=len(tasks) + 1,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                    candidates=candidates,
                )
            )
        elif slot == "smiles_invariance":
            added_slot.extend(
                _invariance_tasks(
                    benchmark_id=benchmark_id,
                    bundle_id=bundle_id,
                    ordinal=len(tasks) + 1,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                )
            )
        elif slot in {"interrupt_resume", "tool_forced_l3"}:
            maybe_task = _interrupt_task(
                benchmark_id=benchmark_id,
                bundle_id=bundle_id,
                ordinal=len(tasks) + 1,
                spec=spec,
                seed=seed,
                source_record=source,
                candidates=candidates,
                task_type=slot,
            )
            if maybe_task is not None:
                added_slot.append(maybe_task)
        if not added_slot:
            added_slot.extend(
                _boundary_tasks(
                    benchmark_id=benchmark_id,
                    bundle_id=bundle_id,
                    ordinal=len(tasks) + 1,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                    candidates=candidates,
                )
            )
        if not added_slot:
            repair = _repair_task(
                benchmark_id=benchmark_id,
                bundle_id=bundle_id,
                ordinal=len(tasks) + 1,
                task_type="repair_near_miss",
                spec=spec,
                seed=seed,
                source_record=source,
                candidates=candidates,
                protocol="L2",
            )
            if repair is not None:
                added_slot.append(repair)
        tasks.extend(added_slot)

        has_feasible = any(task["task_type"] in {"construct_feasible", "audit_accept"} and task["expected_action"] == "ACCEPT" for task in tasks)
        has_audit = any(task["task_type"] in {"audit_accept", "audit_reject"} for task in tasks)
        has_repair_or_boundary_or_invariance = any(task["task_type"] in {"repair_near_miss", "repair_multi_violation", "boundary_precision", "smiles_invariance", "interrupt_resume", "tool_forced_l3"} for task in tasks)
        if len(tasks) < 3 or not (has_feasible and has_audit and has_repair_or_boundary_or_invariance):
            skipped["bundle_minimum_not_met"] += 1
            continue

        bundle = {
            "bundle_id": bundle_id,
            "suite": benchmark_id,
            "split": "train",
            "seed": seed,
            "source_molecule_id": str(source.get("molecule_id")),
            "source_smiles": witness,
            "source_canonical_smiles": witness,
            "spec_id": spec.id,
            "spec_instance_hash": spec_instance_hash(spec),
            "scaffold_hash": str(source.get("scaffold_hash", "")) or None,
            "oracle_summary": {
                "task_types": dict(Counter(task["task_type"] for task in tasks)),
                "oracle_types": dict(Counter(task["oracle_type"] for task in tasks)),
                "expected_actions": dict(Counter(task["expected_action"] for task in tasks)),
            },
            "task_ids": [task["task_id"] for task in tasks],
            "metadata": {"source_index": source_index},
        }
        raw_bundles.append(bundle)
        all_tasks.extend(tasks)

    spec_by_id = {spec.id: spec for spec in specs_sorted}

    def _refresh_bundle_summary(bundle: dict[str, Any]) -> None:
        bundle_tasks = [task for task in all_tasks if task["bundle_id"] == bundle["bundle_id"]]
        bundle["task_ids"] = [task["task_id"] for task in sorted(bundle_tasks, key=lambda item: item["task_id"])]
        bundle["oracle_summary"] = {
            "task_types": dict(Counter(task["task_type"] for task in bundle_tasks)),
            "oracle_types": dict(Counter(task["oracle_type"] for task in bundle_tasks)),
            "expected_actions": dict(Counter(task["expected_action"] for task in bundle_tasks)),
        }

    def _bundle_still_valid(bundle: dict[str, Any]) -> bool:
        bundle_tasks = [task for task in all_tasks if task["bundle_id"] == bundle["bundle_id"]]
        has_feasible = any(
            task["task_type"] in {"construct_feasible", "audit_accept"} and task["expected_action"] == "ACCEPT"
            for task in bundle_tasks
        )
        has_audit = any(task["task_type"] in {"audit_accept", "audit_reject"} for task in bundle_tasks)
        has_repair_or_boundary_or_invariance = any(
            task["task_type"]
            in {"repair_near_miss", "repair_multi_violation", "boundary_precision", "smiles_invariance", "interrupt_resume", "tool_forced_l3"}
            for task in bundle_tasks
        )
        return len(bundle_tasks) >= 3 and has_feasible and has_audit and has_repair_or_boundary_or_invariance

    def _prune_duplicate_agent_visible_tasks() -> int:
        nonlocal all_tasks
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for task in all_tasks:
            groups[str(task.get("agent_visible_hash", ""))].append(task)
        split_priority = {"test": 0, "dev": 1, "train": 2}
        kept_ids: set[str] = set()
        pruned = 0
        for task_hash, rows in groups.items():
            if not task_hash or len(rows) == 1:
                kept_ids.update(str(row["task_id"]) for row in rows)
                continue
            keep = sorted(
                rows,
                key=lambda row: (
                    split_priority.get(str(row.get("split", "")), 3),
                    str(row.get("task_id", "")),
                ),
            )[0]
            kept_ids.add(str(keep["task_id"]))
            pruned += len(rows) - 1
        if pruned:
            all_tasks = [task for task in all_tasks if str(task["task_id"]) in kept_ids]
        return pruned

    def _drop_invalid_bundles_after_prune() -> int:
        nonlocal all_tasks, raw_bundles
        valid_bundle_ids = {
            str(bundle["bundle_id"])
            for bundle in raw_bundles
            if _bundle_still_valid(bundle)
        }
        dropped = len(raw_bundles) - len(valid_bundle_ids)
        if dropped:
            raw_bundles = [bundle for bundle in raw_bundles if str(bundle["bundle_id"]) in valid_bundle_ids]
            all_tasks = [task for task in all_tasks if str(task["bundle_id"]) in valid_bundle_ids]
        for bundle in raw_bundles:
            _refresh_bundle_summary(bundle)
        return dropped

    def _coalesce_leakage_linked_bundles() -> int:
        parent: dict[str, str] = {str(bundle["bundle_id"]): str(bundle["bundle_id"]) for bundle in raw_bundles}

        def _find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def _union(left: str, right: str) -> None:
            root_left = _find(left)
            root_right = _find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        owners_by_key: dict[str, list[str]] = defaultdict(list)
        for task in all_tasks:
            bundle_id = str(task.get("bundle_id", ""))
            if bundle_id not in parent:
                continue
            task_hash = task.get("agent_visible_hash")
            if isinstance(task_hash, str) and task_hash:
                owners_by_key[f"agent::{task_hash}"].append(bundle_id)
            input_block = task.get("input") if isinstance(task.get("input"), dict) else {}
            input_smiles = input_block.get("smiles") or input_block.get("candidate_smiles")
            spec_key = task.get("spec_instance_hash") or task.get("spec_id")
            if isinstance(input_smiles, str) and input_smiles:
                owners_by_key[f"input::{spec_key}::{input_smiles}"].append(bundle_id)
        for owners in owners_by_key.values():
            if len(owners) < 2:
                continue
            first = owners[0]
            for owner in owners[1:]:
                _union(first, owner)

        split_priority = {"test": 0, "dev": 1, "train": 2}
        component_splits: dict[str, str] = {}
        for bundle in raw_bundles:
            bundle_id = str(bundle["bundle_id"])
            root = _find(bundle_id)
            split = str(bundle.get("split", "test"))
            current = component_splits.get(root)
            if current is None or split_priority.get(split, 3) < split_priority.get(current, 3):
                component_splits[root] = split

        changed = 0
        for bundle in raw_bundles:
            bundle_id = str(bundle["bundle_id"])
            split = component_splits[_find(bundle_id)]
            if bundle.get("split") != split:
                changed += 1
                bundle["split"] = split
        split_by_id = {str(bundle["bundle_id"]): str(bundle["split"]) for bundle in raw_bundles}
        for task in all_tasks:
            task["split"] = split_by_id.get(str(task.get("bundle_id")), str(task.get("split", "test")))
        return changed

    def _force_missing_task_type(task_type: str) -> bool:
        for bundle in raw_bundles:
            spec = spec_by_id[str(bundle["spec_id"])]
            candidates = candidates_by_spec[spec.id]
            source = {
                "canonical_smiles": bundle["source_canonical_smiles"],
                "scaffold_hash": bundle.get("scaffold_hash"),
                "molecule_id": bundle.get("source_molecule_id"),
            }
            ordinal = len([task for task in all_tasks if task["bundle_id"] == bundle["bundle_id"]]) + 1
            new_tasks: list[dict[str, Any]] = []
            if task_type in {"repair_near_miss", "repair_multi_violation"}:
                task = _repair_task(
                    benchmark_id=benchmark_id,
                    bundle_id=str(bundle["bundle_id"]),
                    ordinal=ordinal,
                    task_type=task_type,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                    candidates=candidates,
                    protocol="L2" if task_type == "repair_near_miss" else "L3",
                )
                if task is not None:
                    new_tasks.append(task)
            elif task_type == "boundary_precision":
                new_tasks.extend(
                    _boundary_tasks(
                        benchmark_id=benchmark_id,
                        bundle_id=str(bundle["bundle_id"]),
                        ordinal=ordinal,
                        spec=spec,
                        seed=seed,
                        source_record=source,
                        candidates=candidates,
                    )
                )
            elif task_type == "smiles_invariance":
                new_tasks.extend(
                    _invariance_tasks(
                        benchmark_id=benchmark_id,
                        bundle_id=str(bundle["bundle_id"]),
                        ordinal=ordinal,
                        spec=spec,
                        seed=seed,
                        source_record=source,
                    )
                )
            elif task_type in {"interrupt_resume", "tool_forced_l3"}:
                task = _interrupt_task(
                    benchmark_id=benchmark_id,
                    bundle_id=str(bundle["bundle_id"]),
                    ordinal=ordinal,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                    candidates=candidates,
                    task_type=task_type,
                )
                if task is not None:
                    new_tasks.append(task)
            if new_tasks:
                all_tasks.extend(new_tasks)
                _refresh_bundle_summary(bundle)
                return True
        return False

    if raw_bundles:
        present_types = {str(task["task_type"]) for task in all_tasks}
        for required_type in TASK_TYPES_V1:
            if required_type in present_types:
                continue
            if _force_missing_task_type(required_type):
                present_types = {str(task["task_type"]) for task in all_tasks}
            else:
                skipped[f"missing_task_type_{required_type}"] += 1

    if max_tasks is not None and max_tasks > 0:
        task_counts_by_bundle = Counter(str(task["bundle_id"]) for task in all_tasks)
        while sum(task_counts_by_bundle.values()) > max_tasks and raw_bundles:
            removed = raw_bundles.pop()
            removed_id = str(removed["bundle_id"])
            task_counts_by_bundle.pop(removed_id, None)
            all_tasks = [task for task in all_tasks if task["bundle_id"] != removed_id]

    split_by_bundle = assign_bundle_splits(raw_bundles, seed=seed)
    split_policy = {
        "name": "bundle_hash_seeded_50_20_30_with_duplicate_public_view_pruning",
        "seed": seed,
        "proportions": {"train": 0.50, "dev": 0.20, "test": 0.30},
        "unit": "bundle",
        "duplicate_public_view_policy": "drop later exact agent-visible duplicates before release writing; prefer test, then dev, then train when retaining one copy",
    }
    for bundle in raw_bundles:
        bundle["split"] = split_by_bundle.get(str(bundle["bundle_id"]), "test")
    for task in all_tasks:
        task["split"] = split_by_bundle.get(str(task["bundle_id"]), "test")

    def _task_type_count_for_split(split: str) -> Counter[str]:
        return Counter(
            str(task["task_type"])
            for task in all_tasks
            if str(task.get("split")) == split
        )

    def _add_task_type_to_bundle(bundle: dict[str, Any], task_type: str) -> list[dict[str, Any]]:
        spec = spec_by_id[str(bundle["spec_id"])]
        candidates = candidates_by_spec[spec.id]
        source = {
            "canonical_smiles": bundle["source_canonical_smiles"],
            "scaffold_hash": bundle.get("scaffold_hash"),
            "molecule_id": bundle.get("source_molecule_id"),
        }
        ordinal = len([task for task in all_tasks if task["bundle_id"] == bundle["bundle_id"]]) + 1
        new_tasks: list[dict[str, Any]] = []
        if task_type == "construct_feasible":
            evaluator = ConstraintEvaluator(spec)
            witness = str(source["canonical_smiles"])
            witness_result = evaluator.evaluate(witness)
            if witness_result.hard_pass:
                new_tasks.append(
                    _make_task(
                        benchmark_id=benchmark_id,
                        bundle_id=str(bundle["bundle_id"]),
                        task_type=task_type,
                        ordinal=ordinal,
                        spec=spec,
                        seed=seed,
                        source_record=source,
                        protocol=("L1", "L2", "L3")[ordinal % 3],
                        expected_action="ACCEPT",
                        input_smiles=None,
                        evidence={
                            "oracle_type": "feasible_witness",
                            "feasible_witness_smiles": witness,
                            "feasible_witness_canonical_smiles": witness_result.canonical_smiles,
                            "witness_verifier_result": verifier_result_payload(witness_result),
                        },
                    )
                )
        elif task_type in {"audit_accept", "audit_reject", "abstain_contradiction"}:
            skipped[f"quota_duplicate_unsafe_{task_type}"] += 1
        elif task_type in {"repair_near_miss", "repair_multi_violation"}:
            task = _repair_task(
                benchmark_id=benchmark_id,
                bundle_id=str(bundle["bundle_id"]),
                ordinal=ordinal,
                task_type=task_type,
                spec=spec,
                seed=seed,
                source_record=source,
                candidates=candidates,
                protocol="L2" if task_type == "repair_near_miss" else "L3",
            )
            if task is not None:
                new_tasks.append(task)
        elif task_type == "boundary_precision":
            new_tasks.extend(
                _boundary_tasks(
                    benchmark_id=benchmark_id,
                    bundle_id=str(bundle["bundle_id"]),
                    ordinal=ordinal,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                    candidates=candidates,
                )
            )
        elif task_type == "smiles_invariance":
            new_tasks.extend(
                _invariance_tasks(
                    benchmark_id=benchmark_id,
                    bundle_id=str(bundle["bundle_id"]),
                    ordinal=ordinal,
                    spec=spec,
                    seed=seed,
                    source_record=source,
                )
            )
        elif task_type in {"interrupt_resume", "tool_forced_l3"}:
            task = _interrupt_task(
                benchmark_id=benchmark_id,
                bundle_id=str(bundle["bundle_id"]),
                ordinal=ordinal,
                spec=spec,
                seed=seed,
                source_record=source,
                candidates=candidates,
                task_type=task_type,
            )
            if task is not None:
                new_tasks.append(task)
        for task in new_tasks:
            task["split"] = str(bundle.get("split", "test"))
        return new_tasks

    requested_test_minimums = {
        key: int(value)
        for key, value in (min_test_task_type_counts or {}).items()
        if key in TASK_TYPES_V1 and int(value) > 0
    }
    if requested_test_minimums:
        test_bundles = [bundle for bundle in raw_bundles if str(bundle.get("split")) == "test"]
        for task_type, minimum in sorted(requested_test_minimums.items()):
            attempts = 0
            while _task_type_count_for_split("test").get(task_type, 0) < minimum and test_bundles:
                made_progress = False
                for bundle in test_bundles:
                    current_bundle_tasks = [
                        task for task in all_tasks if task["bundle_id"] == bundle["bundle_id"]
                    ]
                    if any(task["task_type"] == task_type for task in current_bundle_tasks):
                        continue
                    additions = _add_task_type_to_bundle(bundle, task_type)
                    if not additions:
                        continue
                    all_tasks.extend(additions)
                    _refresh_bundle_summary(bundle)
                    made_progress = True
                    if _task_type_count_for_split("test").get(task_type, 0) >= minimum:
                        break
                attempts += 1
                if not made_progress or attempts > 3:
                    break
    coalesced = _coalesce_leakage_linked_bundles()
    if coalesced:
        skipped["bundles_reassigned_to_keep_public_input_groups_in_split"] += coalesced
    duplicate_pruned = _prune_duplicate_agent_visible_tasks()
    if duplicate_pruned:
        skipped["duplicate_agent_visible_tasks_pruned"] += duplicate_pruned
    invalid_bundles_dropped = _drop_invalid_bundles_after_prune()
    if invalid_bundles_dropped:
        skipped["bundles_dropped_after_duplicate_pruning"] += invalid_bundles_dropped
    test_type_counts = dict(sorted(_task_type_count_for_split("test").items()))
    diagnostic_only = [
        task_type
        for task_type, minimum in sorted(requested_test_minimums.items())
        if test_type_counts.get(task_type, 0) < minimum
    ]

    bundles = [BundleModel.model_validate(bundle) for bundle in sorted(raw_bundles, key=lambda item: str(item["bundle_id"]))]
    all_tasks.sort(key=lambda item: str(item["task_id"]))
    generated_tasks = len(all_tasks)
    requested_tasks = int(min_tasks or 0)
    shortfall = max(0, requested_tasks - generated_tasks)
    reason = None
    if shortfall:
        reason = "insufficient valid oracle-backed tasks after validation"
    elif skipped:
        reason = "; ".join(f"{key}={value}" for key, value in sorted(skipped.items()))

    return BundleCompilationResult(
        bundles=bundles,
        tasks=all_tasks,
        requested_bundles=target_bundles,
        generated_bundles=len(bundles),
        requested_tasks=requested_tasks,
        generated_tasks=generated_tasks,
        generation_shortfall=shortfall,
        generation_shortfall_reason=reason,
        split_policy=split_policy,
        test_task_type_minimums=requested_test_minimums,
        test_task_type_counts=test_type_counts,
        test_task_type_minimums_met=not diagnostic_only,
        diagnostic_only_test_task_types=diagnostic_only,
    )
