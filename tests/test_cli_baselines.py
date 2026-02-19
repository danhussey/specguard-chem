from __future__ import annotations

import json
from pathlib import Path

from click.exceptions import Exit
import pytest

from specguard_chem.cli import compare_baselines, run_baselines


def test_run_baselines_writes_summary_artifact(tmp_path: Path) -> None:
    out_dir = tmp_path / "baselines"
    run_baselines(
        suite="basic_plain",
        spec_split="train",
        limit=1,
        out_dir=out_dir,
        seed=11,
    )

    summaries = list(out_dir.rglob("baseline_summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["suite"] == "basic_plain"
    assert payload["spec_split"] == "train"
    assert payload["limit"] == 1
    assert len(payload["runs"]) == 3

    names = {run["name"] for run in payload["runs"]}
    assert names == {
        "heuristic_non_tool_l2",
        "heuristic_tool_l3",
        "abstention_guard_l2",
    }

    for run in payload["runs"]:
        run_path = Path(run["run_path"])
        report_path = Path(run["report_path"])
        assert run_path.joinpath("trace.jsonl").exists()
        assert run_path.joinpath("leaderboard.tsv").exists()
        assert report_path.exists()


def test_compare_baselines_aggregates_multiple_summaries(tmp_path: Path) -> None:
    root = tmp_path / "baseline_inputs"
    root.mkdir(parents=True, exist_ok=True)
    summary_a = root / "a" / "baseline_summary.json"
    summary_b = root / "b" / "baseline_summary.json"
    summary_a.parent.mkdir(parents=True, exist_ok=True)
    summary_b.parent.mkdir(parents=True, exist_ok=True)

    payload_a = {
        "generated_at": "2026-02-19T00:00:00",
        "suite": "basic_plain",
        "spec_split": "train",
        "runs": [
            {
                "name": "heuristic_non_tool_l2",
                "model": "heuristic",
                "protocol": "L2",
                "num_tasks": 10,
                "accept_rate": 0.2,
                "hard_violation_rate": 0.6,
                "abstention_utility": -8.0,
            },
            {
                "name": "heuristic_tool_l3",
                "model": "open_source_example",
                "protocol": "L3",
                "num_tasks": 10,
                "accept_rate": 0.5,
                "hard_violation_rate": 0.3,
                "abstention_utility": -4.0,
            },
        ],
    }
    payload_b = {
        "generated_at": "2026-02-19T01:00:00",
        "suite": "basic_plain",
        "spec_split": "test",
        "runs": [
            {
                "name": "heuristic_non_tool_l2",
                "model": "heuristic",
                "protocol": "L2",
                "num_tasks": 10,
                "accept_rate": 0.4,
                "hard_violation_rate": 0.4,
                "abstention_utility": -6.0,
            },
            {
                "name": "abstention_guard_l2",
                "model": "abstention_guard",
                "protocol": "L2",
                "num_tasks": 10,
                "accept_rate": 0.6,
                "hard_violation_rate": 0.2,
                "abstention_utility": -3.0,
            },
        ],
    }
    summary_a.write_text(json.dumps(payload_a), encoding="utf-8")
    summary_b.write_text(json.dumps(payload_b), encoding="utf-8")

    output = tmp_path / "baseline_compare.json"
    compare_baselines(summary_paths=[root], output=output)

    assert output.exists()
    combined = json.loads(output.read_text(encoding="utf-8"))
    assert combined["n_rows"] == 4
    assert combined["group_by"] == ["name"]
    assert len(combined["aggregate"]) == 3

    by_name = {row["name"]: row for row in combined["aggregate"]}
    assert by_name["heuristic_non_tool_l2"]["mean_accept_rate"] == pytest.approx(0.3)
    assert by_name["heuristic_non_tool_l2"]["mean_hard_violation_rate"] == pytest.approx(
        0.5
    )
    assert by_name["heuristic_non_tool_l2"]["mean_abstention_utility"] == pytest.approx(-7.0)


def test_compare_baselines_supports_group_by_dimensions(tmp_path: Path) -> None:
    root = tmp_path / "baseline_inputs"
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "batch" / "baseline_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-02-19T00:00:00",
        "suite": "basic_plain",
        "spec_split": "train",
        "runs": [
            {
                "name": "heuristic_non_tool_l2",
                "model": "heuristic",
                "protocol": "L2",
                "num_tasks": 10,
                "accept_rate": 0.2,
                "hard_violation_rate": 0.6,
                "abstention_utility": -8.0,
            },
            {
                "name": "heuristic_non_tool_l2",
                "model": "heuristic",
                "protocol": "L2",
                "num_tasks": 10,
                "accept_rate": 0.5,
                "hard_violation_rate": 0.3,
                "abstention_utility": -5.0,
            },
        ],
    }
    summary.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "baseline_compare_grouped.json"
    compare_baselines(
        summary_paths=[root],
        group_by="name,spec_split",
        output=output,
    )

    combined = json.loads(output.read_text(encoding="utf-8"))
    assert combined["group_by"] == ["name", "spec_split"]
    assert len(combined["aggregate"]) == 1
    aggregate = combined["aggregate"][0]
    assert aggregate["name"] == "heuristic_non_tool_l2"
    assert aggregate["spec_split"] == "train"
    assert aggregate["mean_accept_rate"] == pytest.approx(0.35)
    assert aggregate["mean_hard_violation_rate"] == pytest.approx(0.45)


def test_compare_baselines_rejects_unknown_group_by_dimension(tmp_path: Path) -> None:
    root = tmp_path / "baseline_inputs"
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "baseline_summary.json"
    payload = {
        "generated_at": "2026-02-19T00:00:00",
        "suite": "basic_plain",
        "spec_split": "train",
        "runs": [
            {
                "name": "heuristic_non_tool_l2",
                "model": "heuristic",
                "protocol": "L2",
                "num_tasks": 10,
                "accept_rate": 0.2,
                "hard_violation_rate": 0.6,
                "abstention_utility": -8.0,
            }
        ],
    }
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exit):
        compare_baselines(
            summary_paths=[root],
            group_by="name,unknown_field",
        )
