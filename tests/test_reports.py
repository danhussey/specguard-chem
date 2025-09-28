from __future__ import annotations

from pathlib import Path

import json
import pytest

from specguard_chem.scoring.reports import summarise, load_trace, write_report


def test_summarise_computes_extended_metrics(tmp_path: Path) -> None:
    records = [
        {
            "task_id": "t1",
            "spec_score": 1.2,
            "hard_pass": True,
            "rounds": [{"action": "propose"}],
            "edit_distance": 2,
            "decision": "accept",
            "final_confidence": 0.8,
        },
        {
            "task_id": "t2",
            "spec_score": 0.9,
            "hard_pass": False,
            "rounds": [{"action": "propose"}, {"action": "propose"}],
            "edit_distance": None,
            "decision": "reject",
            "final_confidence": 0.3,
        },
        {
            "task_id": "t3",
            "spec_score": 1.0,
            "hard_pass": True,
            "rounds": [{"action": "propose"}],
            "edit_distance": 1,
            "decision": "abstain",
            "final_confidence": None,
        },
    ]

    summary = summarise(records)

    assert summary["num_tasks"] == 3
    assert summary["avg_spec_score"] > 1.0
    assert 0 <= summary["hard_violation_rate"] <= 1
    assert summary["abstain_rate"] == pytest.approx(1 / 3)
    assert summary["avg_rounds"] == pytest.approx(4 / 3)
    assert summary["avg_edit_distance"] == pytest.approx((2 + 1) / 2)
    assert summary["accept_rate"] == pytest.approx(1 / 3)
    assert summary["avg_confidence"] is not None
    assert summary["brier_score"] is not None
    assert summary["ece"] is not None
    assert summary["abstention_utility"] < 0


def test_write_report_persists_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace_path = run_dir / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": "x",
                    "spec_score": 1.0,
                    "hard_pass": True,
                    "rounds": [],
                    "edit_distance": 0,
                    "decision": "accept",
                    "final_confidence": 0.9,
                }
            )
            for _ in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    target = write_report(run_dir)
    payload = json.loads(target.read_text())
    assert "summary" in payload
    assert payload["summary"]["num_tasks"] == 2

    loaded_trace = load_trace(run_dir)
    assert len(loaded_trace) == 2
