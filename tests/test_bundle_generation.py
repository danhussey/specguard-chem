from __future__ import annotations


def test_bundle_references_are_consistent(
    v1_tasks: list[dict], v1_bundles: list[dict], v1_manifest: dict
) -> None:
    task_ids = {task["task_id"] for task in v1_tasks}
    bundle_ids = {bundle["bundle_id"] for bundle in v1_bundles}
    assert all(task.get("bundle_id") in bundle_ids for task in v1_tasks)
    assert all(task.get("bundle_id") for task in v1_tasks)
    for bundle in v1_bundles:
        assert len(bundle["task_ids"]) >= 3
        assert set(bundle["task_ids"]).issubset(task_ids)
        assert bundle["spec_id"]
        assert bundle["split"] in {"train", "dev", "test"}
    assert v1_manifest["num_bundles"] == len(v1_bundles)


def test_required_task_types_exist(v1_tasks: list[dict]) -> None:
    task_types = {task["task_type"] for task in v1_tasks}
    assert {
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
    }.issubset(task_types)
