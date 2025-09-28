from __future__ import annotations

from pathlib import Path

from specguard_chem.runner.runner import TaskRunner


def test_runner_smoke_creates_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    runner = TaskRunner("heuristic", seed=13)
    results = runner.run_suite("basic", protocol="L1", limit=1, run_dir=run_dir)
    assert len(results) == 1
    record = results[0]
    assert record.task_id
    assert run_dir.joinpath("trace.jsonl").exists()
    assert len(record.rounds) >= 1
    assert record.spec_score >= 0.0
