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
from ..verifiers import canonicalize_smiles, morgan_tanimoto


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
    rooted_indices = sorted({0, max(mol.GetNumAtoms() // 2, 0), max(mol.GetNumAtoms() - 1, 0)})
    for atom_index in rooted_indices:
        candidate = Chem.MolToSmiles(
            mol,
            canonical=False,
            rootedAtAtom=int(atom_index),
            isomericSmiles=True,
            kekuleSmiles=False,
        )
        if candidate and candidate not in forms and canonicalize_smiles(candidate) == canonical:
            forms.append(candidate)
    aromatic_variant = Chem.MolToSmiles(
        mol,
        canonical=False,
        rootedAtAtom=0,
        isomericSmiles=True,
        kekuleSmiles=True,
    )
    if aromatic_variant and aromatic_variant not in forms and canonicalize_smiles(aromatic_variant) == canonical:
        forms.append(aromatic_variant)
    forms = forms[:4]
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


def _pick_input_relative_witness(
    *,
    passing_records: Sequence[Dict[str, Any]],
    input_smiles: str,
    target_min_similarity: float,
    margin: float = 0.02,
    floor: float = 0.60,
) -> Optional[Tuple[str, float, float]]:
    input_canonical = canonicalize_smiles(input_smiles)
    if not input_canonical:
        return None
    scored: List[Tuple[float, str]] = []
    for record in passing_records:
        candidate = str(record.get("canonical_smiles", ""))
        if not candidate:
            continue
        if candidate == input_canonical:
            continue
        similarity = morgan_tanimoto(input_canonical, candidate)
        if similarity is None:
            continue
        scored.append((float(similarity), candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_similarity, _best_smiles = scored[0]
    if best_similarity >= floor:
        effective_threshold = min(
            target_min_similarity,
            max(floor, best_similarity - margin),
        )
    else:
        effective_threshold = max(0.0, best_similarity - margin)
    effective_threshold = min(effective_threshold, best_similarity)
    for similarity, candidate in scored:
        if similarity >= effective_threshold:
            return candidate, round(effective_threshold, 3), similarity
    return None


def _protocol_for_family(family: str, index: int) -> str:
    if family == "feasible_propose":
        return ("L1", "L2", "L3")[index % 3]
    if family == "repair_near_miss":
        return "L2" if index % 3 == 0 else "L3"
    if family == "repair_multi_violation":
        return "L2" if index % 4 == 0 else "L3"
    if family == "smiles_invariance":
        return "L1"
    if family == "boundary_precision":
        return "L2" if index % 2 == 0 else "L3"
    if family == "interrupt_resume":
        return "L3"
    if family == "tool_forced_l3":
        return "L3"
    return "L1"


def _build_accept_task(
    *,
    suite_name: str,
    family: str,
    spec: SpecModel,
    index: int,
    input_smiles: Optional[str],
    witness_smiles: str,
    similarity_min_to_input: Optional[float] = None,
    similarity_radius: int = 2,
    similarity_n_bits: int = 2048,
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
    if (
        input_smiles
        and similarity_min_to_input is not None
        and family.startswith("repair")
    ):
        task["task_constraints"] = {
            "additions": [
                {
                    "id": "input_similarity_guard",
                    "type": "hard",
                    "check": "similarity_min_to_input",
                    "params": {
                        "min": float(similarity_min_to_input),
                        "fp": "morgan",
                        "radius": int(similarity_radius),
                        "nBits": int(similarity_n_bits),
                    },
                }
            ]
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
        "task_constraints": {
            "additions": [
                {
                    "id": "invariance_similarity_guard",
                    "type": "hard",
                    "check": "similarity_min_to_input",
                    "params": {
                        "min": 0.985,
                        "fp": "morgan",
                        "radius": 2,
                        "nBits": 2048,
                    },
                }
            ]
        },
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
    boundary_target_value: float,
    boundary_epsilon: float,
    similarity_min_to_input: float,
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
        "task_constraints": {
            "additions": [
                {
                    "id": "boundary_tight_window",
                    "type": "hard",
                    "check": "property_bounds",
                    "params": {
                        "mode": "all",
                        "bounds": {
                            boundary_property: {
                                "min": float(boundary_target_value - boundary_epsilon),
                                "max": float(boundary_target_value + boundary_epsilon),
                            }
                        },
                    },
                },
                {
                    "id": "boundary_similarity_guard",
                    "type": "hard",
                    "check": "similarity_min_to_input",
                    "params": {
                        "min": float(similarity_min_to_input),
                        "fp": "morgan",
                        "radius": 2,
                        "nBits": 2048,
                    },
                },
            ]
        },
        "evidence": {
            "feasible_witness_smiles": witness_smiles,
            "boundary_property": boundary_property,
            "boundary_side": "lower",
            "boundary_distance": float(boundary_epsilon),
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
    similarity_min_to_input: float,
) -> Dict[str, Any]:
    protocol = _protocol_for_family("interrupt_resume", index)
    task = {
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
            f"Acknowledge interrupt, report state, verify, then continue to satisfy {spec.id}."
        ),
        "input": {"smiles": input_smiles},
        "spec_id": spec.id,
        "scoring": {"primary": "spec_compliance", "secondary": "edit_distance"},
        "expected": "PASS",
        "expected_action": "ACCEPT",
        "task_family": "interrupt_resume",
        "budgets": {
            "max_steps": 3,
            "max_proposals": 2,
            "max_verify_calls": 1,
            "max_total_verifier_calls": 3,
        },
        "task_constraints": {
            "additions": [
                {
                    "id": "interrupt_similarity_guard",
                    "type": "hard",
                    "check": "similarity_min_to_input",
                    "params": {
                        "min": float(similarity_min_to_input),
                        "fp": "morgan",
                        "radius": 2,
                        "nBits": 2048,
                    },
                }
            ]
        },
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
    return task


def _build_tool_forced_l3_task(
    *,
    suite_name: str,
    spec: SpecModel,
    index: int,
    input_smiles: str,
    witness_smiles: str,
    similarity_min_to_input: float,
    steering_property: str,
    steering_target_value: float,
    steering_epsilon: float,
) -> Dict[str, Any]:
    protocol = _protocol_for_family("tool_forced_l3", index)
    return {
        "task_id": _task_id(
            suite_name=suite_name,
            family="tool_forced_l3",
            spec_id=spec.id,
            index=index,
        ),
        "suite": suite_name,
        "protocol": protocol,
        "prompt": (
            "This is an L3 tool-forced task. "
            "Use verify() to inspect margins, then repair under tight constraints."
        ),
        "input": {"smiles": input_smiles},
        "spec_id": spec.id,
        "scoring": {"primary": "spec_compliance", "secondary": "edit_distance"},
        "expected": "PASS",
        "expected_action": "ACCEPT",
        "task_family": "tool_forced_l3",
        "budgets": {
            "max_steps": 3,
            "max_proposals": 2,
            "max_verify_calls": 1,
            "max_total_verifier_calls": 3,
        },
        "task_constraints": {
            "additions": [
                {
                    "id": "tool_similarity_guard",
                    "type": "hard",
                    "check": "similarity_min_to_input",
                    "params": {
                        "min": float(similarity_min_to_input),
                        "fp": "morgan",
                        "radius": 2,
                        "nBits": 2048,
                    },
                },
                {
                    "id": "tool_boundary_window",
                    "type": "hard",
                    "check": "property_bounds",
                    "params": {
                        "mode": "all",
                        "bounds": {
                            steering_property: {
                                "min": float(steering_target_value - steering_epsilon),
                                "max": float(steering_target_value + steering_epsilon),
                            }
                        },
                    },
                },
            ]
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
                boundary_prop = nearest[0]
                candidate = {
                    "smiles": smiles,
                    "boundary_property": boundary_prop,
                    "boundary_side": nearest[1],
                    "boundary_distance": nearest[2],
                    "boundary_value": float(result.properties.get(boundary_prop, 0.0)),
                    "properties": dict(result.properties),
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

    n_boundary = max(1, int(target_tasks * 0.05))
    n_feasible = int(target_tasks * 0.08)
    n_near = int(target_tasks * 0.28)
    n_multi = int(target_tasks * 0.24)
    n_abstain = int(target_tasks * 0.10)
    n_invariance = int(target_tasks * 0.08)
    n_interrupt = int(target_tasks * 0.05)
    n_tool_forced = max(1, int(target_tasks * 0.12))
    planned = (
        n_feasible
        + n_near
        + n_multi
        + n_abstain
        + n_invariance
        + n_interrupt
        + n_boundary
        + n_tool_forced
    )
    if planned < target_tasks:
        n_near += target_tasks - planned
    elif planned > target_tasks:
        overflow = planned - target_tasks
        n_near = max(1, n_near - overflow)
    if n_invariance % 3 != 0:
        remainder = n_invariance % 3
        n_invariance -= remainder
        n_feasible += remainder
    if n_invariance <= 0:
        n_invariance = 3
        n_feasible = max(0, n_feasible - 3)

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
            input_smiles: Optional[str] = None
            witness_smiles: Optional[str] = None
            similarity_min: Optional[float] = None
            if family == "repair_near_miss":
                source_pool = candidates["near_miss"] or candidates["multi_violation"]
                if source_pool:
                    input_smiles = source_pool[offset % len(source_pool)]["smiles"]
                    picked = _pick_input_relative_witness(
                        passing_records=passing,
                        input_smiles=str(input_smiles),
                        target_min_similarity=0.78,
                    )
                    if picked is not None:
                        witness_smiles = picked[0]
                        similarity_min = picked[1]
            elif family == "repair_multi_violation":
                source_pool = candidates["multi_violation"] or candidates["near_miss"]
                if source_pool:
                    input_smiles = source_pool[offset % len(source_pool)]["smiles"]
                    picked = _pick_input_relative_witness(
                        passing_records=passing,
                        input_smiles=str(input_smiles),
                        target_min_similarity=0.70,
                    )
                    if picked is not None:
                        witness_smiles = picked[0]
                        similarity_min = picked[1]
            else:
                witness_smiles = str(
                    passing[(offset + rng.randrange(len(passing))) % len(passing)][
                        "canonical_smiles"
                    ]
                )
            if family.startswith("repair"):
                if not input_smiles or not witness_smiles or similarity_min is None:
                    continue
            task = _build_accept_task(
                suite_name=suite_name,
                family=family,
                spec=spec,
                index=next_index(family, spec.id),
                input_smiles=input_smiles,
                witness_smiles=str(witness_smiles),
                similarity_min_to_input=similarity_min,
            )
            tasks.append(task)

    invariance_group_count = n_invariance // 3
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
            if len(forms) >= 3:
                selected_canonical = canonicalize_smiles(candidate_smiles)
                selected_forms = forms[:3]
                break
        if selected_canonical is None or selected_forms is None:
            continue
        group_id = (
            f"{suite_name}__invariance_group__{spec.id}__{group_offset + 1:05d}"
        )
        labels = ("canonical", "branch_variant", "kekule_variant")
        for label, form in zip(labels, selected_forms):
            tasks.append(
                _build_invariance_task(
                    suite_name=suite_name,
                    spec=spec,
                    index=next_index("smiles_invariance", spec.id),
                    group_id=group_id,
                    variant_label=label,
                    input_smiles=form,
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
        input_smiles = witness_smiles
        near_candidates = (
            per_spec_candidates[spec.id]["near_miss"]
            or per_spec_candidates[spec.id]["multi_violation"]
        )
        if near_candidates:
            scored_inputs: List[Tuple[float, str]] = []
            for candidate in near_candidates:
                candidate_smiles = str(candidate["smiles"])
                similarity = morgan_tanimoto(candidate_smiles, witness_smiles)
                if similarity is None:
                    continue
                scored_inputs.append((float(similarity), candidate_smiles))
            if scored_inputs:
                scored_inputs.sort(key=lambda item: (-item[0], item[1]))
                input_smiles = scored_inputs[0][1]
        witness_similarity = morgan_tanimoto(input_smiles, witness_smiles)
        similarity_min = max(
            0.0,
            min(0.95, float(witness_similarity if witness_similarity is not None else 0.0) - 0.02),
        )
        boundary_epsilon = (0.0, 0.02, 0.05)[offset % 3]
        task = _build_boundary_task(
            suite_name=suite_name,
            spec=spec,
            index=next_index("boundary_precision", spec.id),
            input_smiles=input_smiles,
            witness_smiles=witness_smiles,
            boundary_property=str(source["boundary_property"]),
            boundary_target_value=float(source.get("boundary_value", 0.0)),
            boundary_epsilon=boundary_epsilon,
            similarity_min_to_input=similarity_min,
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
        near = candidates["near_miss"] or candidates["multi_violation"]
        if not near:
            continue
        input_smiles = str(near[offset % len(near)]["smiles"])
        picked = _pick_input_relative_witness(
            passing_records=passing,
            input_smiles=input_smiles,
            target_min_similarity=0.72,
        )
        if picked is None:
            continue
        witness, similarity_min, _similarity = picked
        task = _build_interrupt_resume_task(
            suite_name=suite_name,
            spec=spec,
            index=next_index("interrupt_resume", spec.id),
            input_smiles=input_smiles,
            witness_smiles=str(witness),
            similarity_min_to_input=similarity_min,
        )
        tasks.append(task)

    for offset in range(n_tool_forced):
        spec = specs_sorted[offset % len(specs_sorted)]
        candidates = per_spec_candidates[spec.id]
        passing = candidates["passing"]
        near = candidates["near_miss"] or candidates["multi_violation"]
        if not passing or not near:
            continue
        source = near[offset % len(near)]
        input_smiles = str(source["smiles"])
        picked = _pick_input_relative_witness(
            passing_records=passing,
            input_smiles=input_smiles,
            target_min_similarity=0.75,
        )
        if picked is None:
            continue
        witness_smiles, similarity_min, _similarity = picked
        witness_record = next(
            (
                record
                for record in passing
                if str(record.get("canonical_smiles", "")) == witness_smiles
            ),
            None,
        )
        if witness_record is None:
            continue
        properties = witness_record.get("properties") or {}
        if not isinstance(properties, dict) or not properties:
            continue
        steering_property = sorted(properties.keys())[offset % len(properties)]
        steering_value = float(properties[steering_property])
        task = _build_tool_forced_l3_task(
            suite_name=suite_name,
            spec=spec,
            index=next_index("tool_forced_l3", spec.id),
            input_smiles=input_smiles,
            witness_smiles=witness_smiles,
            similarity_min_to_input=similarity_min,
            steering_property=steering_property,
            steering_target_value=steering_value,
            steering_epsilon=0.08,
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
