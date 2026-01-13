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
            "edit_morgan_tanimoto": 0.8,
            "decision": "accept",
            "final_confidence": 0.8,
            "expected": "PASS",
        },
        {
            "task_id": "t2",
            "spec_score": 0.9,
            "hard_pass": False,
            "rounds": [{"action": "propose"}, {"action": "propose"}],
            "edit_distance": None,
            "edit_morgan_tanimoto": None,
            "decision": "reject",
            "final_confidence": 0.3,
            "expected": "PASS",
        },
        {
            "task_id": "t3",
            "spec_score": 1.0,
            "hard_pass": True,
            "rounds": [{"action": "propose"}],
            "edit_distance": 1,
            "edit_morgan_tanimoto": 0.6,
            "decision": "abstain",
            "final_confidence": None,
            "expected": "ABSTAIN",
        },
    ]

    summary = summarise(records)

    assert summary["num_tasks"] == 3
    assert summary["avg_spec_score"] > 1.0
    assert summary["hard_violation_rate"] == pytest.approx(0.5)
    assert summary["expected_pass_rate"] == pytest.approx(0.5)
    assert summary["false_abstain_rate"] == pytest.approx(0.0)
    assert summary["violation_rate"] == pytest.approx(0.5)
    assert summary["correct_abstain_rate"] == pytest.approx(1.0)
    assert summary["unsafe_completion_rate"] == pytest.approx(0.0)
    assert summary["interrupt_compliance_rate"] is None
    assert summary["n_interrupt_tasks"] == 0
    assert summary["n_interrupt_fired"] == 0
    assert summary["n_interrupt_compliant"] == 0
    assert summary["avg_morgan_tanimoto"] == pytest.approx(0.7)
    assert summary["median_morgan_tanimoto"] == pytest.approx(0.7)
    assert summary["n_edit_measured"] == 2
    assert summary["n_morgan_measured"] == 2
    assert summary["confusion"] == {"TA": 1, "FA": 0, "FV": 1, "TB": 1, "UA": 0}
    assert summary["n_expected_pass"] == 2
    assert summary["n_expected_abstain"] == 1
    assert summary["abstain_rate"] == pytest.approx(1 / 3)
    assert summary["avg_rounds"] == pytest.approx(4 / 3)
    assert summary["avg_edit_distance"] == pytest.approx((2 + 1) / 2)
    assert summary["accept_rate"] == pytest.approx(1 / 3)
    assert summary["avg_confidence"] == pytest.approx((0.8 + 0.3) / 2)
    assert summary["brier_score"] == pytest.approx(0.065, abs=1e-6)
    assert summary["ece"] == pytest.approx(0.25, abs=1e-6)
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
                    "spec_id": "spec_v1_basic",
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
    assert "metadata" in payload
    assert "definitions" in payload
    assert "expected" in payload["definitions"]
    assert "observed" in payload["definitions"]
    assert "confusion" in payload["definitions"]
    assert "rates" in payload["definitions"]
    assert "aggregates" in payload["definitions"]
    assert "utility" in payload["definitions"]
    assert "utility_matrix" in payload
    assert "rdkit_version" in payload["metadata"]
    assert "git_commit" in payload["metadata"]
    assert "git_dirty" in payload["metadata"]
    assert "specs" in payload["metadata"]
    assert "utility_costs" in payload["metadata"]
    assert "spec_v1_basic" in payload["metadata"]["specs"]
    assert payload["summary"]["num_tasks"] == 2

    loaded_trace = load_trace(run_dir)
    assert len(loaded_trace) == 2
