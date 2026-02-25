from __future__ import annotations

"""Dataset invariant validation for generated task suites."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from ..benchmark.effective_spec import build_effective_spec
from ..config import PATHS, ProjectPaths, TaskModel, load_spec
from ..runner.protocols import ConstraintEvaluator
from ..utils import jsonio
from ..verifiers import canonicalize_smiles


def _hard_violation_units(result: Any) -> int:
    total = 0
    for outcome in result.hard_outcomes:
        if outcome.passed:
            continue
        property_details = outcome.info.get("property_details") or []
        if property_details:
            violated = sum(
                1 for item in property_details if float(item.get("signed_margin", 0.0)) < 0
            )
            total += max(violated, 1)
        else:
            total += 1
    return total


def _nearest_hard_boundary(result: Any) -> tuple[str, str, float] | None:
    nearest: tuple[str, str, float] | None = None
    for outcome in result.hard_outcomes:
        if outcome.constraint.check != "property_bounds":
            continue
        property_details = outcome.info.get("property_details") or []
        for item in property_details:
            bounds = item.get("bounds") or {}
            if "min" not in bounds or "max" not in bounds:
                continue
            prop = str(item.get("property"))
            value = float(item.get("value"))
            lower = float(bounds["min"])
            upper = float(bounds["max"])
            lower_distance = abs(value - lower)
            upper_distance = abs(upper - value)
            if lower_distance <= upper_distance:
                side = "lower"
                distance = lower_distance
            else:
                side = "upper"
                distance = upper_distance
            candidate = (prop, side, distance)
            if nearest is None or distance < nearest[2]:
                nearest = candidate
    return nearest


def _is_bounds_contradiction(proof: Mapping[str, Any]) -> bool:
    if proof.get("type") != "bounds_contradiction":
        return False
    required_min = proof.get("required_min")
    spec_upper = proof.get("spec_upper")
    required_max = proof.get("required_max")
    spec_lower = proof.get("spec_lower")
    if required_min is not None and spec_upper is not None:
        return float(required_min) > float(spec_upper)
    if required_max is not None and spec_lower is not None:
        return float(required_max) < float(spec_lower)
    return False


def validate_dataset_records(
    records: Iterable[Dict[str, Any]],
    *,
    paths: ProjectPaths = PATHS,
    near_miss_margin_band: float = 5.0,
    boundary_margin_band: float = 1.0,
    repair_start_hard_fail_threshold: float = 0.70,
    min_counts: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    """Validate generated task invariants and evidence blocks."""

    if min_counts is None:
        min_counts = {
            "feasible_propose": 1,
            "repair_near_miss": 1,
            "repair_multi_violation": 1,
            "contradiction_abstain": 1,
            "smiles_invariance": 1,
            "boundary_precision": 1,
            "interrupt_resume": 1,
        }

    parsed_tasks: List[TaskModel] = []
    errors: List[str] = []
    for index, record in enumerate(records, start=1):
        try:
            parsed_tasks.append(TaskModel.model_validate(record))
        except Exception as exc:
            errors.append(f"task#{index}: schema validation failed: {exc}")

    task_ids = [task.task_id for task in parsed_tasks]
    id_counts = Counter(task_ids)
    duplicate_ids = sorted(task_id for task_id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        errors.append(
            "duplicate task_id values: " + ", ".join(duplicate_ids[:10])
            + (" ..." if len(duplicate_ids) > 10 else "")
        )

    family_counts: Dict[str, int] = defaultdict(int)
    spec_cache: Dict[tuple[str, str], ConstraintEvaluator] = {}
    invariance_groups: Dict[tuple[str, str, str], List[TaskModel]] = defaultdict(list)
    repair_total = 0
    repair_hard_fail_starts = 0

    for task in parsed_tasks:
        family = task.task_family or "unspecified"
        family_counts[family] += 1
        task_constraints_payload = (
            task.task_constraints.model_dump(mode="json")
            if task.task_constraints is not None
            else None
        )
        task_constraints_key = (
            json.dumps(
                task_constraints_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            if task_constraints_payload is not None
            else ""
        )
        cache_key = (task.spec_id, task_constraints_key)
        if cache_key not in spec_cache:
            base_spec = load_spec(task.spec_id, paths=paths)
            effective_spec = build_effective_spec(base_spec, task.task_constraints)
            spec_cache[cache_key] = ConstraintEvaluator(effective_spec)
        evaluator = spec_cache[cache_key]

        if task.expected_action == "ACCEPT":
            evidence = task.evidence
            witness = evidence.feasible_witness_smiles if evidence else None
            if not witness:
                errors.append(f"{task.task_id}: missing feasible_witness_smiles evidence")
            else:
                witness_result = evaluator.evaluate(witness)
                if not witness_result.hard_pass:
                    errors.append(f"{task.task_id}: witness does not hard-pass spec")

            if family in {"repair_near_miss", "repair_multi_violation"}:
                repair_total += 1
                if not task.input.smiles:
                    errors.append(f"{task.task_id}: repair task missing input.smiles")
                else:
                    start_result = evaluator.evaluate(task.input.smiles)
                    hard_fail_count = sum(
                        1 for outcome in start_result.hard_outcomes if not outcome.passed
                    )
                    hard_units = _hard_violation_units(start_result)
                    if hard_fail_count > 0:
                        repair_hard_fail_starts += 1
                    if family == "repair_near_miss":
                        if hard_units != 1:
                            errors.append(
                                f"{task.task_id}: near-miss task must violate exactly one hard unit"
                            )
                        margins = [
                            abs(value)
                            for value in start_result.property_margins.values()
                            if value < 0
                        ]
                        if margins and min(margins) > near_miss_margin_band:
                            errors.append(
                                f"{task.task_id}: near-miss margin {min(margins):.3f} exceeds band"
                            )
                    if family == "repair_multi_violation" and hard_units < 2:
                        errors.append(
                            f"{task.task_id}: multi-violation task needs >=2 hard violation units"
                        )
            elif family == "smiles_invariance":
                if not evidence:
                    errors.append(f"{task.task_id}: smiles_invariance task missing evidence")
                else:
                    group_id = evidence.invariance_group_id
                    canonical = evidence.invariance_canonical_smiles
                    if not group_id:
                        errors.append(
                            f"{task.task_id}: smiles_invariance task missing invariance_group_id"
                        )
                    if not canonical:
                        errors.append(
                            f"{task.task_id}: smiles_invariance task missing invariance_canonical_smiles"
                        )
                    if not task.input.smiles:
                        errors.append(
                            f"{task.task_id}: smiles_invariance task missing input.smiles"
                        )
                    else:
                        input_canonical = canonicalize_smiles(task.input.smiles)
                        if canonical and input_canonical != canonical:
                            errors.append(
                                f"{task.task_id}: input SMILES is not equivalent to invariance canonical"
                            )
                    if witness and canonicalize_smiles(witness) != canonical:
                        errors.append(
                            f"{task.task_id}: witness does not match invariance canonical"
                        )
                    if group_id:
                        invariance_groups[
                            (task.spec_id, task_constraints_key, group_id)
                        ].append(task)
            elif family == "boundary_precision":
                if not evidence:
                    errors.append(f"{task.task_id}: boundary_precision task missing evidence")
                else:
                    boundary_property = evidence.boundary_property
                    boundary_side = evidence.boundary_side
                    boundary_distance = evidence.boundary_distance
                    if not boundary_property:
                        errors.append(
                            f"{task.task_id}: boundary_precision missing boundary_property"
                        )
                    if boundary_side not in {"lower", "upper"}:
                        errors.append(
                            f"{task.task_id}: boundary_precision missing valid boundary_side"
                        )
                    if boundary_distance is None:
                        errors.append(
                            f"{task.task_id}: boundary_precision missing boundary_distance"
                        )
                    if witness:
                        witness_result = evaluator.evaluate(witness)
                        nearest = _nearest_hard_boundary(witness_result)
                        if nearest is None:
                            errors.append(
                                f"{task.task_id}: witness has no hard property boundary context"
                            )
                        else:
                            if boundary_property and nearest[0] != boundary_property:
                                errors.append(
                                    f"{task.task_id}: boundary_property mismatch (evidence={boundary_property}, observed={nearest[0]})"
                                )
                            if boundary_side and nearest[1] != boundary_side:
                                errors.append(
                                    f"{task.task_id}: boundary_side mismatch (evidence={boundary_side}, observed={nearest[1]})"
                                )
                            if (
                                boundary_distance is not None
                                and abs(nearest[2] - boundary_distance) > 1e-6
                            ):
                                errors.append(
                                    f"{task.task_id}: boundary_distance mismatch (evidence={boundary_distance:.6f}, observed={nearest[2]:.6f})"
                                )

        if task.expected_action == "ABSTAIN":
            evidence = task.evidence
            contradiction = evidence.contradiction_proof if evidence else None
            if not contradiction:
                errors.append(f"{task.task_id}: ABSTAIN task needs contradiction_proof")
            if contradiction and not _is_bounds_contradiction(contradiction):
                errors.append(
                    f"{task.task_id}: contradiction_proof present but not contradictory"
                )

    for (spec_id, task_constraints_key, group_id), tasks_in_group in invariance_groups.items():
        if len(tasks_in_group) < 2:
            errors.append(
                f"invariance group {group_id} ({spec_id}) must contain at least two tasks"
            )
            continue
        evaluator = spec_cache.get((spec_id, task_constraints_key))
        if evaluator is None:
            evaluator = ConstraintEvaluator(load_spec(spec_id, paths=paths))
        base_task = tasks_in_group[0]
        if not base_task.input.smiles:
            errors.append(f"{base_task.task_id}: invariance base task missing input.smiles")
            continue
        base_result = evaluator.evaluate(base_task.input.smiles)
        for candidate_task in tasks_in_group[1:]:
            if not candidate_task.input.smiles:
                errors.append(
                    f"{candidate_task.task_id}: invariance task missing input.smiles"
                )
                continue
            candidate_result = evaluator.evaluate(candidate_task.input.smiles)
            if candidate_result.hard_pass != base_result.hard_pass:
                errors.append(
                    f"{candidate_task.task_id}: invariance hard_pass mismatch within group {group_id}"
                )
            if candidate_result.alerts != base_result.alerts:
                errors.append(
                    f"{candidate_task.task_id}: invariance alert mismatch within group {group_id}"
                )
            for prop_name, base_value in base_result.properties.items():
                candidate_value = candidate_result.properties.get(prop_name)
                if candidate_value is None or abs(candidate_value - base_value) > 1e-9:
                    errors.append(
                        f"{candidate_task.task_id}: invariance property mismatch for {prop_name}"
                    )

    for family, min_count in min_counts.items():
        if family_counts.get(family, 0) < min_count:
            errors.append(
                f"distribution sanity failed: {family} has {family_counts.get(family, 0)} < {min_count}"
            )

    repair_start_hard_fail_rate = (
        (repair_hard_fail_starts / repair_total) if repair_total else 0.0
    )
    if repair_total and repair_start_hard_fail_rate < repair_start_hard_fail_threshold:
        errors.append(
            "distribution sanity failed: repair start hard-fail rate "
            f"{repair_start_hard_fail_rate:.3f} < {repair_start_hard_fail_threshold:.3f}"
        )

    return {
        "valid": len(errors) == 0,
        "num_tasks": len(parsed_tasks),
        "num_errors": len(errors),
        "errors": errors,
        "family_counts": dict(sorted(family_counts.items())),
        "repair_start_hard_fail_rate": repair_start_hard_fail_rate,
        "repair_start_hard_fail_threshold": repair_start_hard_fail_threshold,
    }


def validate_dataset_file(
    path: Path,
    *,
    paths: ProjectPaths = PATHS,
    near_miss_margin_band: float = 5.0,
    boundary_margin_band: float = 1.0,
    repair_start_hard_fail_threshold: float = 0.70,
    min_counts: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    records = jsonio.read_jsonl(path)
    return validate_dataset_records(
        records,
        paths=paths,
        near_miss_margin_band=near_miss_margin_band,
        boundary_margin_band=boundary_margin_band,
        repair_start_hard_fail_threshold=repair_start_hard_fail_threshold,
        min_counts=min_counts,
    )
