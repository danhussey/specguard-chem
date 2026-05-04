from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import pytest

from specguard_chem.benchmark.effective_spec import build_effective_spec
from specguard_chem.config import SpecModel, TaskConstraintsModel
from specguard_chem.dataset.oracles import hard_violation_units
from specguard_chem.dataset.validate_v1 import RELEASE_SPLITS, validate_release_v1
from specguard_chem.runner.protocols import ConstraintEvaluator
from specguard_chem.utils import jsonio


def _copy_release(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _load_tasks(release: Path) -> dict[str, list[dict]]:
    return {
        split: jsonio.read_jsonl(release / "tasks" / f"{split}.jsonl")
        for split in RELEASE_SPLITS
    }


def _write_tasks(release: Path, tasks: dict[str, list[dict]]) -> None:
    for split, rows in tasks.items():
        jsonio.write_jsonl(release / "tasks" / f"{split}.jsonl", rows)


def _mutate_first(
    release: Path,
    predicate: Callable[[dict], bool],
    mutate: Callable[[dict], None],
) -> str:
    tasks = _load_tasks(release)
    for split, rows in tasks.items():
        for row in rows:
            if predicate(row):
                mutate(row)
                _write_tasks(release, tasks)
                return str(row["task_id"])
    raise AssertionError("No task matched mutation predicate")


def _assert_invalid(release: Path, contains: str) -> None:
    result = validate_release_v1(release, strict=True)
    assert result["valid"] is False
    rendered = "\n".join(result["errors"])
    assert contains in rendered


@pytest.mark.parametrize(
    ("name", "mutate", "contains"),
    [
        (
            "audit_accept_candidate_fails",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "audit_accept",
                lambda row: row["input"].update({"smiles": "not_a_smiles"}),
            ),
            "audit_accept candidate does not hard-pass",
        ),
        (
            "audit_reject_candidate_passes",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "audit_reject",
                lambda row: row["input"].update({"smiles": row["source_canonical_smiles"]}),
            ),
            "audit_reject candidate does not hard-fail",
        ),
        (
            "near_miss_fails_two",
            lambda release: _set_near_miss_to_multi_violation_input(release),
            "near-miss input fails",
        ),
        (
            "multi_violation_fails_one",
            lambda release: _reduce_multi_violation_to_one_distinct_failure(release),
            "multi-violation input fails",
        ),
        (
            "repair_witness_invalid",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") in {"repair_near_miss", "repair_multi_violation"},
                lambda row: row["evidence"].update({"feasible_witness_smiles": "not_a_smiles"}),
            ),
            "repair input/witness oracle failed",
        ),
        (
            "construct_witness_fails",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "construct_feasible",
                lambda row: row["evidence"].update({"feasible_witness_smiles": "not_a_smiles"}),
            ),
            "construct witness does not hard-pass",
        ),
        (
            "abstain_certificate_removed",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "abstain_contradiction",
                lambda row: row["evidence"].pop("unsat_certificate", None),
            ),
            "missing unsat_certificate",
        ),
        (
            "abstain_certificate_not_visible",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "abstain_contradiction",
                lambda row: row["evidence"]["unsat_certificate"].update({"constraints": ["not_visible"]}),
            ),
            "references constraints not visible",
        ),
        (
            "invariance_non_equivalent",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "smiles_invariance",
                lambda row: row["evidence"].update({"variant_smiles": [row["input"]["smiles"], "CC"]}),
            ),
            "variants do not canonicalize together",
        ),
        (
            "duplicate_hash_cross_split",
            lambda release: _duplicate_agent_hash_cross_split(release),
            "agent-visible duplicate crosses splits",
        ),
        (
            "rendered_exposes_witness",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "construct_feasible",
                lambda row: row.update(
                    {
                        "rendered_agent_input": row["rendered_agent_input"]
                        + "\n"
                        + row["evidence"]["feasible_witness_smiles"]
                    }
                ),
            ),
            "exposes hidden witness",
        ),
        (
            "l3_no_verify_calls",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("protocol") == "L3",
                lambda row: row["budgets"].update({"max_verify_calls": 0}),
            ),
            "L3 requires max_verify_calls > 0",
        ),
        (
            "l1_visible_verify_tool",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("protocol") == "L1",
                lambda row: row["agent_visible_payload"].update({"allowed_tools": ["verify"]}),
            ),
            "L1 visible tools must not include verify",
        ),
        (
            "prompt_internal_audit_label",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "audit_reject",
                lambda row: row.update({"rendered_agent_input": row["rendered_agent_input"] + "\naudit_reject"}),
            ),
            "visible prompt contains internal label audit_reject",
        ),
        (
            "prompt_output_schema_extra_action",
            lambda release: _mutate_first(
                release,
                lambda row: row.get("task_type") == "audit_reject",
                lambda row: row.update(
                    {
                        "rendered_agent_input": row["rendered_agent_input"].replace(
                            '"action": "ACCEPT|REJECT"',
                            '"action": "ACCEPT|REJECT|ABSTAIN"',
                        )
                    }
                ),
            ),
            "rendered output schema actions",
        ),
        (
            "prompt_instance_soft_window",
            lambda release: _mutate_first(
                release,
                lambda row: True,
                lambda row: row.update(
                    {
                        "rendered_agent_input": row["rendered_agent_input"].replace(
                            "Soft preferences:\n",
                            "Soft preferences:\n1. instance_soft_window_bad: MW between 130.000 and 130.001 (weight 0.010)\n",
                            1,
                        )
                    }
                ),
            ),
            "rendered_agent_input exposes instance_soft_window",
        ),
        (
            "prompt_forbidden_claim",
            lambda release: _mutate_first(
                release,
                lambda row: True,
                lambda row: row.update({"rendered_agent_input": row["rendered_agent_input"] + "\npotency"}),
            ),
            "agent-visible forbidden term matches found",
        ),
    ],
)
def test_strict_validator_rejects_corrupted_releases(
    v1_release: Path,
    tmp_path: Path,
    name: str,
    mutate: Callable[[Path], str],
    contains: str,
) -> None:
    release = _copy_release(v1_release, tmp_path / name)
    mutate(release)
    _assert_invalid(release, contains)


def _duplicate_agent_hash_cross_split(release: Path) -> str:
    tasks = _load_tasks(release)
    source = tasks["train"][0]["agent_visible_hash"]
    tasks["test"][0]["agent_visible_hash"] = source
    _write_tasks(release, tasks)
    return str(tasks["test"][0]["task_id"])


def _failing_constraint_count(result: object) -> int:
    outcomes = getattr(result, "hard_outcomes", [])
    return len({outcome.constraint.id for outcome in outcomes if not outcome.passed})


def _set_near_miss_to_multi_violation_input(release: Path) -> str:
    tasks = _load_tasks(release)
    for rows in tasks.values():
        for row in rows:
            if row.get("task_type") != "repair_near_miss":
                continue
            if _append_distinct_hard_guard(row):
                _write_tasks(release, tasks)
                return str(row["task_id"])
            spec = SpecModel.model_validate(jsonio.read_json(release / "specs" / f"{row['spec_id']}.json"))
            constraints = (
                TaskConstraintsModel.model_validate(row["task_constraints"])
                if isinstance(row.get("task_constraints"), dict)
                else None
            )
            effective = build_effective_spec(spec, constraints)
            evaluator = ConstraintEvaluator(effective)
            candidates = [
                task["input"]["smiles"]
                for split_rows in tasks.values()
                for task in split_rows
                if task.get("task_type") == "repair_multi_violation"
                and isinstance(task.get("input"), dict)
                and isinstance(task["input"].get("smiles"), str)
            ]
            candidates.extend(["CCCCCCCCCCCCCCCC", "C", "N"])
            for candidate in candidates:
                result = evaluator.evaluate(candidate)
                if result.valid and hard_violation_units(result) >= 2 and _failing_constraint_count(result) >= 2:
                    row["input"]["smiles"] = candidate
                    _write_tasks(release, tasks)
                    return str(row["task_id"])
    raise AssertionError("No repair_near_miss task could be made multi-violation")


def _append_distinct_hard_guard(row: dict) -> bool:
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    input_result = evidence.get("input_verifier_result") if isinstance(evidence.get("input_verifier_result"), dict) else {}
    witness_result = evidence.get("witness_verifier_result") if isinstance(evidence.get("witness_verifier_result"), dict) else {}
    input_props = input_result.get("properties") if isinstance(input_result.get("properties"), dict) else {}
    witness_props = witness_result.get("properties") if isinstance(witness_result.get("properties"), dict) else {}
    for prop in ("MW", "TPSA", "logP", "HBA", "HBD", "ROTB"):
        if prop not in input_props or prop not in witness_props:
            continue
        input_value = float(input_props[prop])
        witness_value = float(witness_props[prop])
        difference = abs(input_value - witness_value)
        if difference <= 0.001:
            continue
        half_width = max(0.001, difference / 3.0)
        addition = {
            "id": "negative_control_distinct_hard_guard",
            "type": "hard",
            "check": "property_bounds",
            "params": {
                "mode": "all",
                "bounds": {prop: {"min": witness_value - half_width, "max": witness_value + half_width}},
            },
        }
        constraints = row.setdefault("task_constraints", {})
        additions = constraints.setdefault("additions", [])
        additions.append(addition)
        return True
    return False


def _reduce_multi_violation_to_one_distinct_failure(release: Path) -> str:
    tasks = _load_tasks(release)
    for rows in tasks.values():
        for row in rows:
            if row.get("task_type") != "repair_multi_violation":
                continue
            original_constraints = row.pop("task_constraints", None)
            spec = SpecModel.model_validate(jsonio.read_json(release / "specs" / f"{row['spec_id']}.json"))
            evaluator = ConstraintEvaluator(spec)
            result = evaluator.evaluate(row["input"]["smiles"])
            witness_result = evaluator.evaluate(row["evidence"]["feasible_witness_smiles"])
            if result.valid and not result.hard_pass and witness_result.hard_pass and _failing_constraint_count(result) == 1:
                _write_tasks(release, tasks)
                return str(row["task_id"])
            if original_constraints is not None:
                row["task_constraints"] = original_constraints
    raise AssertionError("No repair_multi_violation task could be reduced to one distinct failure")


def test_strict_validator_rejects_boundary_group_missing_pass_side(v1_release: Path, tmp_path: Path) -> None:
    release = _copy_release(v1_release, tmp_path / "boundary_missing_pass")
    tasks = _load_tasks(release)
    for split, rows in tasks.items():
        for index, row in enumerate(rows):
            evidence = row.get("evidence") or {}
            if row.get("task_type") == "boundary_precision" and row.get("expected_action") == "ACCEPT":
                rows.pop(index)
                _write_tasks(release, tasks)
                _assert_invalid(release, "missing pass/fail contrast")
                return
    raise AssertionError("No boundary pass task found")


def test_strict_validator_rejects_boundary_group_missing_fail_side(v1_release: Path, tmp_path: Path) -> None:
    release = _copy_release(v1_release, tmp_path / "boundary_missing_fail")
    tasks = _load_tasks(release)
    for split, rows in tasks.items():
        for index, row in enumerate(rows):
            if row.get("task_type") == "boundary_precision" and row.get("expected_action") == "REJECT":
                rows.pop(index)
                _write_tasks(release, tasks)
                _assert_invalid(release, "missing pass/fail contrast")
                return
    raise AssertionError("No boundary fail task found")


def test_strict_validator_rejects_bundle_crossing_splits(v1_release: Path, tmp_path: Path) -> None:
    release = _copy_release(v1_release, tmp_path / "bundle_cross_split")
    tasks = _load_tasks(release)
    moved = dict(tasks["train"][0])
    moved["split"] = "test"
    tasks["test"].append(moved)
    _write_tasks(release, tasks)
    _assert_invalid(release, "bundle crosses splits")
