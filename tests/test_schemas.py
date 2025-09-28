from __future__ import annotations

from specguard_chem.config import (
    FailureItem,
    FailureVector,
    list_available_suites,
    load_spec,
    load_tasks_for_suite,
    select_tasks,
)


def test_spec_loads_and_contains_constraints() -> None:
    spec = load_spec("spec_v1_basic")
    assert spec.id == "spec_v1_basic"
    assert any(constraint.type == "hard" for constraint in spec.constraints)


def test_task_selection_filters_protocol() -> None:
    tasks = load_tasks_for_suite("basic")
    l1_tasks = select_tasks(tasks, protocol="L1", limit=2)
    assert len(l1_tasks) <= 2
    assert all(task.protocol == "L1" for task in l1_tasks)


def test_interrupts_suite_available() -> None:
    suites = list_available_suites()
    assert "interrupts" in suites


def test_failure_vector_validation() -> None:
    fv = FailureVector(
        hard_fails=[FailureItem(id="demo", detail="oops")],
        soft_misses=[],
        margins=[],
        round=1,
    )
    payload = fv.model_dump()
    assert payload["hard_fails"][0]["id"] == "demo"
