from __future__ import annotations

import json
from pathlib import Path

import pytest

from specguard_chem.config import (
    FailureItem,
    FailureVector,
    SpecModel,
    ProjectPaths,
    list_available_specs,
    list_available_suites,
    load_spec,
    load_tasks_for_suite,
    migrate_spec_v1_to_v2,
    select_tasks,
    validate_unique_task_ids,
)
from specguard_chem.runner.protocols import ConstraintEvaluator


def test_spec_loads_and_contains_constraints() -> None:
    spec = load_spec("spec_v1_basic")
    assert spec.id == "spec_v1_basic"
    assert spec.version == 2
    assert any(constraint.type == "hard" for constraint in spec.constraints)
    checks = {constraint.check for constraint in spec.constraints}
    assert "property_bounds" in checks
    assert "alert_set_absent" in checks
    assert "property_bounds_all" not in checks


def test_task_selection_filters_protocol() -> None:
    tasks = load_tasks_for_suite("basic_plain")
    l1_tasks = select_tasks(tasks, protocol="L1", limit=2)
    assert len(l1_tasks) <= 2
    assert all(task.protocol == "L1" for task in l1_tasks)
    assert all(task.expected for task in tasks)
    assert all(task.expected_action for task in tasks)
    assert all(task.budgets is not None for task in tasks)
    assert all(task.budgets.max_steps >= 1 for task in tasks)


def test_interrupts_suite_available() -> None:
    suites = list_available_suites()
    assert "interrupts" in suites
    assert "interrupt_resume" in suites
    assert "smiles_invariance" in suites
    assert "boundary_precision" in suites
    assert "basic_plain" in suites


def test_spec_catalog_contains_multiple_families_and_splits() -> None:
    spec_ids = list_available_specs()
    assert len(spec_ids) >= 10

    specs = [load_spec(spec_id) for spec_id in spec_ids]
    assert len({spec.family for spec in specs}) >= 5
    assert {"train", "dev", "test"}.issubset({spec.spec_split for spec in specs})


def test_failure_vector_validation() -> None:
    fv = FailureVector(
        hard_fails=[FailureItem(id="demo", detail="oops")],
        soft_misses=[],
        margins=[],
        round=1,
    )
    payload = fv.model_dump()
    assert payload["hard_fails"][0]["id"] == "demo"


def test_validate_unique_task_ids_raises_on_duplicates(tmp_path: Path) -> None:
    suites_dir = tmp_path / "tasks" / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)
    (suites_dir / "suite_a.jsonl").write_text(
        json.dumps(
            {
                "task_id": "dup_id",
                "suite": "suite_a",
                "protocol": "L1",
                "prompt": "p",
                "input": {},
                "spec_id": "spec_v1_basic",
                "scoring": {"primary": "spec_compliance"},
                "expected": "PASS",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (suites_dir / "suite_b.jsonl").write_text(
        json.dumps(
            {
                "task_id": "dup_id",
                "suite": "suite_b",
                "protocol": "L1",
                "prompt": "p",
                "input": {},
                "spec_id": "spec_v1_basic",
                "scoring": {"primary": "spec_compliance"},
                "expected": "PASS",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = ProjectPaths(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        specs_dir=tmp_path / "data" / "specs",
        suites_dir=suites_dir,
    )
    with pytest.raises(ValueError, match="Duplicate task_id"):
        validate_unique_task_ids(paths=paths)


def test_spec_v2_validation_rejects_extra_constraint_fields() -> None:
    payload = {
        "id": "spec_v2_invalid",
        "version": 2,
        "constraints": [
            {
                "id": "c1",
                "type": "hard",
                "check": "property_bounds",
                "params": {
                    "mode": "all",
                    "bounds": {"logP": {"min": 0.0, "max": 5.0}},
                },
                "unexpected": "nope",
            }
        ],
        "behaviour": {"interrupt_policy": "confirm_then_continue"},
    }
    with pytest.raises(ValueError):
        SpecModel.model_validate(payload)


def test_migrate_v1_to_v2_preserves_evaluation_behavior() -> None:
    v1 = {
        "id": "spec_test",
        "version": 1,
        "constraints": [
            {
                "id": "ro5",
                "type": "hard",
                "check": "property_bounds_all",
                "params": {"bounds": {"logP": [0, 3], "MW": [50, 500]}},
            },
            {
                "id": "alerts",
                "type": "soft",
                "check": "substructure_absent",
                "params": {"alert_set": "PAINS_A"},
                "weight": 0.5,
            },
        ],
        "behaviour": {"interrupt_policy": "confirm_then_continue"},
    }
    migrated = migrate_spec_v1_to_v2(v1)
    expected_v2 = {
        "id": "spec_test",
        "version": 2,
        "constraints": [
            {
                "id": "ro5",
                "type": "hard",
                "check": "property_bounds",
                "params": {
                    "mode": "all",
                    "bounds": {
                        "logP": {"min": 0.0, "max": 3.0},
                        "MW": {"min": 50.0, "max": 500.0},
                    },
                },
            },
            {
                "id": "alerts",
                "type": "soft",
                "check": "alert_set_absent",
                "params": {"alert_set": "PAINS_A"},
                "weight": 0.5,
            },
        ],
        "behaviour": {"interrupt_policy": "confirm_then_continue"},
    }

    spec_migrated = SpecModel.model_validate(migrated)
    spec_expected = SpecModel.model_validate(expected_v2)
    evaluator_migrated = ConstraintEvaluator(spec_migrated)
    evaluator_expected = ConstraintEvaluator(spec_expected)

    smiles = "O=C1C=CC(=O)C=C1"
    result_migrated = evaluator_migrated.evaluate(smiles)
    result_expected = evaluator_expected.evaluate(smiles)

    assert result_migrated.hard_pass == result_expected.hard_pass
    assert result_migrated.property_margins == result_expected.property_margins
    assert result_migrated.alerts == result_expected.alerts


def test_equivalent_to_input_policy_defaults_are_materialized() -> None:
    spec = SpecModel.model_validate(
        {
            "id": "spec_equiv_defaults",
            "version": 2,
            "constraints": [
                {
                    "id": "same_identity",
                    "type": "hard",
                    "check": "equivalent_to_input",
                    "params": {"policy": "tautomer_canonical_no_stereo_inchi"},
                }
            ],
            "behaviour": {"interrupt_policy": "confirm_then_continue"},
        }
    )
    params = spec.constraints[0].params
    assert params["policy"] == "tautomer_canonical_no_stereo_inchi"
    assert params["require_stereo"] is False
    assert params["tautomer_invariant"] is True
    assert params["normalize"] == "rdkit_cleanup_plus_tautomer_canon"
