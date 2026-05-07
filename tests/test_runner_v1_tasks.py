from __future__ import annotations

from specguard_chem.config import SpecModel, TaskModel
from specguard_chem.runner.runner import TaskRunner
from specguard_chem.utils import jsonio


def test_runner_handles_one_v1_task_per_type(v1_release, v1_tasks: list[dict]) -> None:
    selected = {}
    for task in v1_tasks:
        selected.setdefault(task["task_type"], task)
    assert len(selected) == 10

    def spec_loader(spec_id: str) -> SpecModel:
        return SpecModel.model_validate(jsonio.read_json(v1_release / "specs" / f"{spec_id}.json"))

    runner = TaskRunner("heuristic", seed=7)
    records = runner.run_tasks(
        [TaskModel.model_validate(task) for task in selected.values()],
        spec_loader=spec_loader,
    )
    assert len(records) == 10
    assert {record.expected_action for record in records}.issuperset({"ACCEPT", "REJECT", "ABSTAIN"})
    assert all(record.final_decision in {"ACCEPT", "REJECT", "ABSTAIN"} for record in records)
