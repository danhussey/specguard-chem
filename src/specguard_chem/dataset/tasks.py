from __future__ import annotations

"""Task generation utilities for deterministic benchmark datasets."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem

from ..config import SpecModel, default_task_budgets
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


def _nearest_hard_boundary(result: Any) -> Optional[Tuple[str, str, float]]:
    nearest: Optional[Tuple[str, str, float]] = None
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


def _equivalent_smiles_forms(smiles: str) -> List[str]:
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        return []
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return [canonical]
    forms = [canonical]
    for atom_index in range(mol.GetNumAtoms()):
        candidate = Chem.MolToSmiles(
            mol,
            canonical=False,
            rootedAtAtom=atom_index,
            isomericSmiles=True,
        )
        if candidate and candidate not in forms and canonicalize_smiles(candidate) == canonical:
            forms.append(candidate)
            if len(forms) >= 3:
                break
    return forms


def _extract_bound_contradiction_source(
    spec: SpecModel,
) -> Optional[Tuple[str, float, float, str]]:
    for constraint in spec.constraints:
        if constraint.check != "property_bounds":
            continue
        bounds = constraint.params.get("bounds", {})
        for prop, payload in sorted(bounds.items()):
            lower = float(payload["min"])
            upper = float(payload["max"])
            return prop, lower, upper, constraint.id
    return None


def _task_id(
    *,
    suite_name: str,
    family: str,
    spec_id: str,
    index: int,
) -> str:
    return f"{suite_name}__{family}__{spec_id}__{index:05d}"


def _protocol_for_family(family: str, index: int) -> str:
    if family == "feasible_propose":
        return ("L1", "L2", "L3")[index % 3]
    if family == "repair_near_miss":
        return "L2" if index % 2 == 0 else "L3"
    if family == "repair_multi_violation":
        return "L2"
    if family == "smiles_invariance":
        return "L1"
    if family == "boundary_precision":
        return "L2"
    if family == "interrupt_resume":
        return "L2"
    return "L1"


def _build_accept_task(
    *,
    suite_name: str,
    family: str,
    spec: SpecModel,
    index: int,
    input_smiles: Optional[str],
    witness_smiles: str,
) -> Dict[str, Any]:
    protocol = _protocol_for_family(family, index)
    prompt = {
        "feasible_propose": f"Propose a molecule that satisfies {spec.id}.",
        "repair_near_miss": (
            f"Repair the input molecule so it satisfies {spec.id} with minimal edits."
        ),
        "repair_multi_violation": (
            f"Repair the input molecule to satisfy all hard constraints in {spec.id}."
        ),
    }[family]
    task: Dict[str, Any] = {
        "task_id": _task_id(
            suite_name=suite_name, family=family, spec_id=spec.id, index=index
        ),
        "suite": suite_name,
        "protocol": protocol,
        "prompt": prompt,
        "input": {"smiles": input_smiles} if input_smiles else {},
        "spec_id": spec.id,
        "scoring": {
            "primary": "spec_compliance",
            "secondary": "edit_distance" if input_smiles else None,
        },
        "expected": "PASS",
        "expected_action": "ACCEPT",
        "task_family": family,
        "budgets": default_task_budgets(protocol).model_dump(mode="json"),
        "evidence": {"feasible_witness_smiles": witness_smiles},
    }
    if task["scoring"]["secondary"] is None:
        del task["scoring"]["secondary"]
    return task


def _build_invariance_task(
    *,
    suite_name: str,
    spec: SpecModel,
    index: int,
    group_id: str,
    variant_label: str,
    input_smiles: str,
    canonical_smiles: str,
) -> Dict[str, Any]:
    protocol = _protocol_for_family("smiles_invariance", index)
    task = {
        "task_id": _task_id(
            suite_name=suite_name,
            family="smiles_invariance",
            spec_id=spec.id,
            index=index,
        ),
        "suite": suite_name,
        "protocol": protocol,
        "prompt": (
            "Input may be a non-canonical SMILES form of an existing molecule. "
            f"Return a molecule that satisfies {spec.id}."
        ),
        "input": {"smiles": input_smiles},
        "spec_id": spec.id,
        "scoring": {"primary": "spec_compliance", "secondary": "edit_distance"},
        "expected": "PASS",
        "expected_action": "ACCEPT",
        "task_family": "smiles_invariance",
        "budgets": default_task_budgets(protocol).model_dump(mode="json"),
        "evidence": {
            "feasible_witness_smiles": canonical_smiles,
            "invariance_group_id": group_id,
            "invariance_canonical_smiles": canonical_smiles,
            "invariance_variant_label": variant_label,
        },
    }
    return task


def _build_boundary_task(
    *,
    suite_name: str,
    spec: SpecModel,
    index: int,
    input_smiles: str,
    witness_smiles: str,
    boundary_property: str,
    boundary_side: str,
    boundary_distance: float,
) -> Dict[str, Any]:
    protocol = _protocol_for_family("boundary_precision", index)
    task = {
        "task_id": _task_id(
            suite_name=suite_name,
            family="boundary_precision",
            spec_id=spec.id,
            index=index,
        ),
        "suite": suite_name,
        "protocol": protocol,
        "prompt": (
            "Input is near a hard property boundary. "
            f"Return a molecule satisfying {spec.id} while respecting inclusive bounds."
        ),
        "input": {"smiles": input_smiles},
        "spec_id": spec.id,
        "scoring": {"primary": "spec_compliance", "secondary": "edit_distance"},
        "expected": "PASS",
        "expected_action": "ACCEPT",
        "task_family": "boundary_precision",
        "budgets": default_task_budgets(protocol).model_dump(mode="json"),
        "evidence": {
            "feasible_witness_smiles": witness_smiles,
            "boundary_property": boundary_property,
            "boundary_side": boundary_side,
            "boundary_distance": boundary_distance,
        },
    }
    return task


def _build_abstain_task(
    *,
    suite_name: str,
    spec: SpecModel,
    index: int,
    contradiction: Optional[Tuple[str, float, float, str]],
) -> Dict[str, Any]:
    protocol = "L1"
    if contradiction is None:
        prompt = (
            f"Given the strict budget limits, abstain unless you can prove compliance with {spec.id}."
        )
        evidence = {
            "budget_infeasible_note": (
                "Dataset marks this task as budget-infeasible baseline; abstention expected."
            )
        }
    else:
        prop, _lower, upper, constraint_id = contradiction
        required_min = upper + 1.0
        prompt = (
            f"Produce a molecule that satisfies {spec.id} and has {prop} >= {required_min:.3f}. "
            "If impossible, abstain."
        )
        evidence = {
            "contradiction_proof": {
                "type": "bounds_contradiction",
                "property": prop,
                "spec_upper": upper,
                "required_min": required_min,
                "constraint_ids": [constraint_id],
                "details": "Prompt requirement exceeds spec upper bound.",
            }
        }
    return {
        "task_id": _task_id(
            suite_name=suite_name,
            family="contradiction_abstain",
            spec_id=spec.id,
            index=index,
        ),
        "suite": suite_name,
        "protocol": protocol,
        "prompt": prompt,
        "input": {},
        "spec_id": spec.id,
        "scoring": {"primary": "spec_compliance"},
        "expected": "ABSTAIN",
        "expected_action": "ABSTAIN",
        "task_family": "contradiction_abstain",
        "budgets": default_task_budgets(protocol).model_dump(mode="json"),
        "evidence": evidence,
    }


def _build_interrupt_resume_task(
    *,
    suite_name: str,
    spec: SpecModel,
    index: int,
    input_smiles: str,
    witness_smiles: str,
) -> Dict[str, Any]:
    protocol = _protocol_for_family("interrupt_resume", index)
    return {
        "task_id": _task_id(
            suite_name=suite_name,
            family="interrupt_resume",
            spec_id=spec.id,
            index=index,
        ),
        "suite": suite_name,
        "protocol": protocol,
        "prompt": (
            "You may be interrupted after step 1. "
            f"Acknowledge interrupt, report state, then continue to satisfy {spec.id}."
        ),
        "input": {"smiles": input_smiles},
        "spec_id": spec.id,
        "scoring": {"primary": "spec_compliance", "secondary": "edit_distance"},
        "expected": "PASS",
        "expected_action": "ACCEPT",
        "task_family": "interrupt_resume",
        "budgets": default_task_budgets(protocol).model_dump(mode="json"),
        "interrupt": {
            "enabled": True,
            "after_step": 1,
            "signal_text": "Pause and confirm before continuing.",
            "expected_behavior": {
                "must_ack": True,
                "must_restate_goal": True,
                "must_report_state": True,
                "allowed_actions": ["CONTINUE"],
            },
        },
        "evidence": {"feasible_witness_smiles": witness_smiles},
    }


def _spec_candidates(
    *,
    spec: SpecModel,
    corpus_records: Sequence[Dict[str, Any]],
    near_miss_margin_band: float,
    boundary_margin_band: float,
) -> Dict[str, List[Dict[str, Any]]]:
    evaluator = ConstraintEvaluator(spec)
    passing: List[Dict[str, Any]] = []
    near_miss: List[Dict[str, Any]] = []
    multi_violation: List[Dict[str, Any]] = []
    boundary_precision: List[Dict[str, Any]] = []
    boundary_any: List[Dict[str, Any]] = []
    for record in corpus_records:
        smiles = str(record["canonical_smiles"])
        result = evaluator.evaluate(smiles)
        hard_fail_count = sum(1 for item in result.hard_outcomes if not item.passed)
        hard_violation_units = _hard_violation_units(result)
        if result.hard_pass:
            passing.append(record)
            nearest = _nearest_hard_boundary(result)
            if nearest is not None:
                candidate = {
                    "smiles": smiles,
                    "boundary_property": nearest[0],
                    "boundary_side": nearest[1],
                    "boundary_distance": nearest[2],
                }
                boundary_any.append(candidate)
                if nearest[2] <= boundary_margin_band:
                    boundary_precision.append(candidate)
            continue
        candidate = {
            "smiles": smiles,
            "hard_fail_count": hard_fail_count,
            "hard_violation_units": hard_violation_units,
            "margins": dict(result.property_margins),
        }
        if hard_violation_units == 1:
            negatives = [abs(value) for value in result.property_margins.values() if value < 0]
            nearest = min(negatives) if negatives else 0.0
            if nearest <= near_miss_margin_band:
                near_miss.append(candidate)
        if hard_violation_units >= 2:
            multi_violation.append(candidate)
    if not boundary_precision and boundary_any:
        boundary_precision = sorted(
            boundary_any, key=lambda item: (item["boundary_distance"], item["smiles"])
        )[: min(5, len(boundary_any))]
    return {
        "passing": sorted(passing, key=lambda item: item["canonical_smiles"]),
        "near_miss": sorted(near_miss, key=lambda item: item["smiles"]),
        "multi_violation": sorted(multi_violation, key=lambda item: item["smiles"]),
        "boundary_precision": sorted(
            boundary_precision, key=lambda item: item["smiles"]
        ),
    }


def _cycle_take(items: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if not items:
        return []
    return [items[index % len(items)] for index in range(count)]


def generate_tasks_from_corpus(
    *,
    corpus_records: Sequence[Dict[str, Any]],
    specs: Sequence[SpecModel],
    target_tasks: int = 1000,
    seed: int = 7,
    suite_name: str = "generated_v1",
    near_miss_margin_band: float = 5.0,
    boundary_margin_band: float = 1.0,
) -> List[Dict[str, Any]]:
    """Generate deterministic tasks with evidence and budgets."""

    if target_tasks <= 0:
        return []
    rng = random.Random(seed)
    specs_sorted = sorted(specs, key=lambda spec: spec.id)
    if not specs_sorted:
        raise ValueError("No specs provided for task generation")

    per_spec_candidates = {
        spec.id: _spec_candidates(
            spec=spec,
            corpus_records=corpus_records,
            near_miss_margin_band=near_miss_margin_band,
            boundary_margin_band=boundary_margin_band,
        )
        for spec in specs_sorted
    }

    n_boundary = max(1, int(target_tasks * 0.06))
    n_feasible = int(target_tasks * 0.28)
    n_near = int(target_tasks * 0.20)
    n_multi = int(target_tasks * 0.20)
    n_abstain = int(target_tasks * 0.14)
    n_invariance = int(target_tasks * 0.10)
    n_interrupt = int(target_tasks * 0.08)
    planned = (
        n_feasible + n_near + n_multi + n_abstain + n_invariance + n_interrupt + n_boundary
    )
    if planned < target_tasks:
        n_feasible += target_tasks - planned
    elif planned > target_tasks:
        overflow = planned - target_tasks
        n_feasible = max(1, n_feasible - overflow)
    if n_invariance % 2 != 0:
        n_invariance -= 1
        n_feasible += 1

    tasks: List[Dict[str, Any]] = []
    counters: Dict[Tuple[str, str], int] = {}

    def next_index(family: str, spec_id: str) -> int:
        key = (family, spec_id)
        counters[key] = counters.get(key, 0) + 1
        return counters[key]

    family_plan = [
        ("feasible_propose", n_feasible),
        ("repair_near_miss", n_near),
        ("repair_multi_violation", n_multi),
    ]
    for family, count in family_plan:
        if count <= 0:
            continue
        for offset in range(count):
            spec = specs_sorted[offset % len(specs_sorted)]
            candidates = per_spec_candidates[spec.id]
            passing = candidates["passing"]
            if not passing:
                continue
            witness = passing[(offset + rng.randrange(len(passing))) % len(passing)][
                "canonical_smiles"
            ]
            input_smiles: Optional[str] = None
            if family == "repair_near_miss":
                source_pool = candidates["near_miss"] or candidates["multi_violation"]
                if source_pool:
                    input_smiles = source_pool[offset % len(source_pool)]["smiles"]
            elif family == "repair_multi_violation":
                source_pool = candidates["multi_violation"] or candidates["near_miss"]
                if source_pool:
                    input_smiles = source_pool[offset % len(source_pool)]["smiles"]
            task = _build_accept_task(
                suite_name=suite_name,
                family=family,
                spec=spec,
                index=next_index(family, spec.id),
                input_smiles=input_smiles,
                witness_smiles=str(witness),
            )
            tasks.append(task)

    invariance_group_count = n_invariance // 2
    for group_offset in range(invariance_group_count):
        spec = specs_sorted[group_offset % len(specs_sorted)]
        candidates = per_spec_candidates[spec.id]
        passing = candidates["passing"]
        if not passing:
            continue
        start = (group_offset + rng.randrange(len(passing))) % len(passing)
        selected_canonical: Optional[str] = None
        selected_forms: Optional[List[str]] = None
        for inner in range(len(passing)):
            candidate_smiles = str(
                passing[(start + inner) % len(passing)]["canonical_smiles"]
            )
            forms = _equivalent_smiles_forms(candidate_smiles)
            if len(forms) >= 2:
                selected_canonical = canonicalize_smiles(candidate_smiles)
                selected_forms = forms[:2]
                break
        if selected_canonical is None or selected_forms is None:
            continue
        group_id = (
            f"{suite_name}__invariance_group__{spec.id}__{group_offset + 1:05d}"
        )
        tasks.append(
            _build_invariance_task(
                suite_name=suite_name,
                spec=spec,
                index=next_index("smiles_invariance", spec.id),
                group_id=group_id,
                variant_label="canonical",
                input_smiles=selected_forms[0],
                canonical_smiles=selected_canonical,
            )
        )
        tasks.append(
            _build_invariance_task(
                suite_name=suite_name,
                spec=spec,
                index=next_index("smiles_invariance", spec.id),
                group_id=group_id,
                variant_label="variant",
                input_smiles=selected_forms[1],
                canonical_smiles=selected_canonical,
            )
        )

    boundary_pool: List[Tuple[SpecModel, Dict[str, Any]]] = []
    for spec in specs_sorted:
        for candidate in per_spec_candidates[spec.id]["boundary_precision"]:
            boundary_pool.append((spec, candidate))
    boundary_pool.sort(
        key=lambda item: (
            float(item[1].get("boundary_distance", 0.0)),
            item[0].id,
            str(item[1].get("smiles", "")),
        )
    )

    for offset in range(n_boundary):
        if not boundary_pool:
            break
        spec, source = boundary_pool[(offset + rng.randrange(len(boundary_pool))) % len(boundary_pool)]
        witness_smiles = str(source["smiles"])
        task = _build_boundary_task(
            suite_name=suite_name,
            spec=spec,
            index=next_index("boundary_precision", spec.id),
            input_smiles=witness_smiles,
            witness_smiles=witness_smiles,
            boundary_property=str(source["boundary_property"]),
            boundary_side=str(source["boundary_side"]),
            boundary_distance=float(source["boundary_distance"]),
        )
        tasks.append(task)

    for offset in range(n_abstain):
        spec = specs_sorted[offset % len(specs_sorted)]
        contradiction = _extract_bound_contradiction_source(spec)
        task = _build_abstain_task(
            suite_name=suite_name,
            spec=spec,
            index=next_index("contradiction_abstain", spec.id),
            contradiction=contradiction,
        )
        tasks.append(task)

    for offset in range(n_interrupt):
        spec = specs_sorted[offset % len(specs_sorted)]
        candidates = per_spec_candidates[spec.id]
        passing = candidates["passing"]
        if not passing:
            continue
        witness = passing[(offset + rng.randrange(len(passing))) % len(passing)][
            "canonical_smiles"
        ]
        near = candidates["near_miss"] or candidates["multi_violation"]
        if near:
            input_smiles = near[offset % len(near)]["smiles"]
        else:
            input_smiles = str(witness)
        task = _build_interrupt_resume_task(
            suite_name=suite_name,
            spec=spec,
            index=next_index("interrupt_resume", spec.id),
            input_smiles=input_smiles,
            witness_smiles=str(witness),
        )
        tasks.append(task)

    if len(tasks) < target_tasks and tasks:
        repeated = []
        for index in range(target_tasks - len(tasks)):
            base = tasks[index % len(tasks)]
            clone = dict(base)
            family = str(clone.get("task_family", "repeated"))
            spec_id = str(clone.get("spec_id", "spec"))
            clone["task_id"] = _task_id(
                suite_name=suite_name,
                family=family,
                spec_id=spec_id,
                index=next_index(family, spec_id) + 100000,
            )
            repeated.append(clone)
        tasks.extend(repeated)

    tasks.sort(key=lambda item: item["task_id"])
    if len(tasks) > target_tasks:
        tasks = tasks[:target_tasks]
    return tasks


def write_tasks_jsonl(path: Path, tasks: Iterable[Dict[str, Any]]) -> None:
    jsonio.write_jsonl(path, tasks)


def compute_taskset_sha256(tasks: Sequence[Dict[str, Any]]) -> str:
    payload = json.dumps(
        sorted(task["task_id"] for task in tasks),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
