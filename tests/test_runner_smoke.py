from __future__ import annotations

from pathlib import Path

from specguard_chem.runner.runner import TaskRunner


def test_runner_smoke_creates_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    runner = TaskRunner("heuristic", seed=13)
    results = runner.run_suite("basic_plain", protocol="L1", limit=1, run_dir=run_dir)
    assert len(results) == 1
    record = results[0]
    assert record.task_id
    assert run_dir.joinpath("trace.jsonl").exists()
    leaderboard = run_dir.joinpath("leaderboard.tsv")
    assert leaderboard.exists()
    header = leaderboard.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("suite\ttask_id\t")
    assert len(record.rounds) >= 1
    assert record.spec_score >= 0.0


def test_runner_can_filter_by_spec_split(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_split"
    runner = TaskRunner("heuristic", seed=13)
    results = runner.run_suite(
        "basic_plain",
        protocol="L1",
        spec_split="dev",
        run_dir=run_dir,
    )
    assert len(results) == 0
    assert run_dir.joinpath("trace.jsonl").exists()
    assert run_dir.joinpath("leaderboard.tsv").exists()
