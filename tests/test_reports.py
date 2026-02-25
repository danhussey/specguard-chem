from __future__ import annotations

import json
from pathlib import Path

import pytest

from specguard_chem.scoring.reports import load_trace, summarise, write_report


def test_summarise_computes_extended_metrics(tmp_path: Path) -> None:
    records = [
        {
            "task_id": "t1",
            "spec_score": 1.2,
            "hard_pass": True,
            "rounds": [{"action": "propose"}],
            "edit_distance": 2,
            "edit_morgan_tanimoto": 0.8,
            "final_edit_cost_brics": 2,
            "trajectory_edit_distance": 4,
            "trajectory_edit_cost_brics": 3,
            "decision": "accept",
            "final_p_hard_pass": 0.8,
            "expected": "PASS",
        },
        {
            "task_id": "t2",
            "spec_score": 0.9,
            "hard_pass": False,
            "rounds": [{"action": "propose"}, {"action": "propose"}],
            "edit_distance": None,
            "edit_morgan_tanimoto": None,
            "final_edit_cost_brics": None,
            "trajectory_edit_distance": None,
            "trajectory_edit_cost_brics": None,
            "decision": "reject",
            "final_p_hard_pass": 0.3,
            "expected": "PASS",
        },
        {
            "task_id": "t3",
            "spec_score": 1.0,
            "hard_pass": True,
            "rounds": [{"action": "propose"}],
            "edit_distance": 1,
            "edit_morgan_tanimoto": 0.6,
            "final_edit_cost_brics": 1,
            "trajectory_edit_distance": 2,
            "trajectory_edit_cost_brics": 1,
            "decision": "abstain",
            "final_p_hard_pass": None,
            "expected": "ABSTAIN",
        },
    ]

    summary = summarise(records)

    assert summary["num_tasks"] == 3
    assert summary["avg_spec_score"] > 1.0
    assert summary["hard_violation_rate"] == pytest.approx(0.5)
    assert summary["pass_at_steps"] == [
        {"step_budget": 1, "pass_rate": pytest.approx(0.5)},
        {"step_budget": 2, "pass_rate": pytest.approx(0.5)},
    ]
    assert summary["avg_steps_to_accept"] == pytest.approx(1.0)
    assert summary["avg_verify_calls_to_accept"] == pytest.approx(0.0)
    assert summary["l3_avg_verify_calls_used"] is None
    assert summary["l3_avg_verify_calls_used_expected_accept"] is None
    assert summary["verify_usage_rate_on_L3"] is None
    assert summary["accept_rate_by_protocol"] == {"unknown": pytest.approx(1 / 3)}
    assert summary["hard_violation_rate_by_protocol"] == {"unknown": pytest.approx(0.5)}
    assert summary["expected_pass_rate"] == pytest.approx(0.5)
    assert summary["false_abstain_rate"] == pytest.approx(0.0)
    assert summary["violation_rate"] == pytest.approx(0.5)
    assert summary["correct_abstain_rate"] == pytest.approx(1.0)
    assert summary["unsafe_completion_rate"] == pytest.approx(0.0)
    assert summary["reject_on_abstain_expected_rate"] == pytest.approx(0.0)
    assert summary["interrupt_compliance_rate"] is None
    assert summary["n_interrupt_tasks"] == 0
    assert summary["n_interrupt_fired"] == 0
    assert summary["n_interrupt_compliant"] == 0
    assert summary["n_resume_tasks"] == 0
    assert summary["n_resume_fired"] == 0
    assert summary["resume_token_ok_rate"] is None
    assert summary["resume_success_rate"] is None
    assert summary["avg_extra_steps_after_interrupt"] is None
    assert summary["avg_morgan_tanimoto"] == pytest.approx(0.7)
    assert summary["median_morgan_tanimoto"] == pytest.approx(0.7)
    assert summary["n_edit_measured"] == 2
    assert summary["n_morgan_measured"] == 2
    assert summary["avg_final_edit_cost_brics"] == pytest.approx(1.5)
    assert summary["avg_trajectory_edit_distance"] == pytest.approx(3.0)
    assert summary["avg_trajectory_edit_cost_brics"] == pytest.approx(2.0)
    assert summary["n_final_edit_cost_brics_measured"] == 2
    assert summary["n_trajectory_edit_distance_measured"] == 2
    assert summary["n_trajectory_edit_cost_brics_measured"] == 2
    assert summary["confusion"] == {
        "ACCEPT": {"ACCEPT": 1, "REJECT": 1, "ABSTAIN": 0},
        "ABSTAIN": {"ACCEPT": 0, "REJECT": 0, "ABSTAIN": 1},
        "REJECT": {"ACCEPT": 0, "REJECT": 0, "ABSTAIN": 0},
    }
    assert summary["legacy_confusion"] == {"TA": 1, "FA": 0, "FV": 1, "TB": 1, "UA": 0}
    assert summary["n_expected_pass"] == 2
    assert summary["n_expected_accept"] == 2
    assert summary["n_expected_abstain"] == 1
    assert summary["n_expected_reject"] == 0
    assert summary["abstain_rate"] == pytest.approx(1 / 3)
    assert summary["avg_rounds"] == pytest.approx(4 / 3)
    assert summary["avg_edit_distance"] == pytest.approx((2 + 1) / 2)
    assert summary["accept_rate"] == pytest.approx(1 / 3)
    assert summary["avg_steps_used"] == pytest.approx(4 / 3)
    assert summary["avg_proposals_used"] == pytest.approx(4 / 3)
    assert summary["avg_verify_calls_used"] == pytest.approx(0.0)
    assert summary["avg_p_hard_pass"] == pytest.approx((0.8 + 0.3) / 2)
    assert summary["brier_score"] == pytest.approx(0.065, abs=1e-6)
    assert summary["ece"] == pytest.approx(0.25, abs=1e-6)
    assert summary["abstention_utility"] == pytest.approx(-2.0)
    assert summary["n_agent_outputs"] == 4
    assert summary["schema_error_rate"] == pytest.approx(0.0)
    assert summary["invalid_action_rate"] == pytest.approx(0.0)
    assert summary["invalid_tool_call_rate"] == pytest.approx(0.0)
    assert summary["spec_family_breakdown"]["unknown"]["num_tasks"] == 3
    assert summary["spec_split_breakdown"]["unknown"]["num_tasks"] == 3
    assert summary["soft_compliance_rate_given_hard_pass"] == pytest.approx(1.0)
    assert summary["weighted_soft_score_given_hard_pass"] == pytest.approx(1.0)
    assert summary["n_invariance_groups"] == 0
    assert summary["n_invariance_groups_evaluable"] == 0
    assert summary["n_invariance_groups_incomplete"] == 0
    assert summary["n_invariance_tasks"] == 0
    assert summary["invariance_failure_rate"] is None
    assert summary["invariance_group_inconsistency_rate"] is None
    assert summary["invariance_failure_rate_by_subfamily"] == {}
    assert summary["invariance_counts_by_subfamily"] == {}
    assert summary["n_boundary_precision_tasks"] == 0
    assert summary["boundary_precision_failure_rate"] is None
    assert summary["boundary_precision_pass_rate"] is None
    assert len(summary["risk_coverage_curve"]["expected_accept"]) == 21
    assert summary["risk_coverage_curve"]["expected_accept"][0] == {
        "threshold": 0.0,
        "coverage": pytest.approx(1.0),
        "risk": pytest.approx(0.5),
    }
    assert summary["risk_coverage_curve"]["expected_accept"][10] == {
        "threshold": 0.5,
        "coverage": pytest.approx(0.5),
        "risk": pytest.approx(0.0),
    }
    assert summary["cost_coverage_curve"]["expected_accept"][10] == {
        "threshold": 0.5,
        "coverage": pytest.approx(0.5),
        "expected_cost": pytest.approx(0.5),
    }
    assert summary["risk_coverage_curve"]["expected_abstain"][0] == {
        "threshold": 0.0,
        "coverage": pytest.approx(1.0),
        "risk": pytest.approx(1.0),
    }
    baseline = next(
        row
        for row in summary["utility_sensitivity"]
        if row["C_ACCEPT_INFEASIBLE"] == 10.0
        and row["C_REJECT_FEASIBLE"] == 2.0
        and row["C_ABSTAIN_FEASIBLE"] == 1.0
    )
    assert baseline["utility"] == pytest.approx(summary["abstention_utility"])


def test_decision_confusion_and_utility_cover_all_action_decision_pairs() -> None:
    records = [
        {
            "task_id": "ea_accept_fd_accept",
            "spec_score": 1.0,
            "hard_pass": True,
            "rounds": [],
            "expected_action": "ACCEPT",
            "final_decision": "ACCEPT",
        },
        {
            "task_id": "ea_accept_fd_reject",
            "spec_score": 0.0,
            "hard_pass": False,
            "rounds": [],
            "expected_action": "ACCEPT",
            "final_decision": "REJECT",
        },
        {
            "task_id": "ea_accept_fd_abstain",
            "spec_score": 0.0,
            "hard_pass": False,
            "rounds": [],
            "expected_action": "ACCEPT",
            "final_decision": "ABSTAIN",
        },
        {
            "task_id": "ea_abstain_fd_accept",
            "spec_score": 0.0,
            "hard_pass": True,
            "rounds": [],
            "expected_action": "ABSTAIN",
            "final_decision": "ACCEPT",
        },
        {
            "task_id": "ea_abstain_fd_reject",
            "spec_score": 0.0,
            "hard_pass": False,
            "rounds": [],
            "expected_action": "ABSTAIN",
            "final_decision": "REJECT",
        },
        {
            "task_id": "ea_abstain_fd_abstain",
            "spec_score": 0.0,
            "hard_pass": False,
            "rounds": [],
            "expected_action": "ABSTAIN",
            "final_decision": "ABSTAIN",
        },
    ]

    summary = summarise(records)

    assert summary["confusion"] == {
        "ACCEPT": {"ACCEPT": 1, "REJECT": 1, "ABSTAIN": 1},
        "ABSTAIN": {"ACCEPT": 1, "REJECT": 1, "ABSTAIN": 1},
        "REJECT": {"ACCEPT": 0, "REJECT": 0, "ABSTAIN": 0},
    }
    assert summary["abstention_utility"] == pytest.approx(-14.0)
    assert "legacy_confusion" not in summary


def test_pass_at_steps_uses_recorded_step_and_verify_counts() -> None:
    records = [
        {
            "task_id": "a",
            "protocol": "L3",
            "spec_score": 1.0,
            "hard_pass": True,
            "expected_action": "ACCEPT",
            "final_decision": "ACCEPT",
            "steps_used": 3,
            "verify_calls_used": 2,
            "proposals_used": 1,
            "rounds": [{"action": "tool_call"}, {"action": "tool_call"}, {"action": "propose"}],
        },
        {
            "task_id": "b",
            "protocol": "L3",
            "spec_score": 0.0,
            "hard_pass": False,
            "expected_action": "ACCEPT",
            "final_decision": "REJECT",
            "steps_used": 2,
            "verify_calls_used": 1,
            "proposals_used": 1,
            "rounds": [{"action": "tool_call"}, {"action": "propose"}],
        },
    ]
    summary = summarise(records)

    assert summary["pass_at_steps"] == [
        {"step_budget": 1, "pass_rate": pytest.approx(0.0)},
        {"step_budget": 2, "pass_rate": pytest.approx(0.0)},
        {"step_budget": 3, "pass_rate": pytest.approx(0.5)},
    ]
    assert summary["avg_steps_to_accept"] == pytest.approx(3.0)
    assert summary["avg_verify_calls_to_accept"] == pytest.approx(2.0)
    assert summary["l3_avg_verify_calls_used"] == pytest.approx(1.5)
    assert summary["l3_avg_verify_calls_used_expected_accept"] == pytest.approx(1.5)
    assert summary["verify_usage_rate_on_L3"] == pytest.approx(1.0)


def test_resume_metrics_are_reported() -> None:
    records = [
        {
            "task_id": "r1",
            "spec_score": 1.0,
            "hard_pass": True,
            "expected_action": "ACCEPT",
            "final_decision": "ACCEPT",
            "rounds": [{"interrupt": True, "action": "propose"}],
            "interrupt_expected": True,
            "interrupt_result": {"compliance": True},
            "resume_expected": True,
            "resume_token_ok": True,
            "resume_success": True,
            "extra_steps_after_interrupt": 1,
        },
        {
            "task_id": "r2",
            "spec_score": 0.0,
            "hard_pass": False,
            "expected_action": "ACCEPT",
            "final_decision": "REJECT",
            "rounds": [{"interrupt": True, "action": "propose"}],
            "interrupt_expected": True,
            "interrupt_result": {"compliance": False},
            "resume_expected": True,
            "resume_token_ok": False,
            "resume_success": False,
            "extra_steps_after_interrupt": 2,
        },
    ]

    summary = summarise(records)
    assert summary["n_resume_tasks"] == 2
    assert summary["n_resume_fired"] == 2
    assert summary["resume_token_ok_rate"] == pytest.approx(0.5)
    assert summary["resume_success_rate"] == pytest.approx(0.5)
    assert summary["avg_extra_steps_after_interrupt"] == pytest.approx(1.5)


def test_gaming_resistance_metrics_are_reported() -> None:
    records = [
        {
            "task_id": "inv_a",
            "spec_score": 1.0,
            "hard_pass": True,
            "expected_action": "ACCEPT",
            "final_decision": "ACCEPT",
            "task_family": "smiles_invariance",
            "invariance_group_id": "g1",
            "invariance_subfamily": "stereo",
            "rounds": [],
        },
        {
            "task_id": "inv_b",
            "spec_score": 0.0,
            "hard_pass": False,
            "expected_action": "ACCEPT",
            "final_decision": "REJECT",
            "task_family": "smiles_invariance",
            "invariance_group_id": "g1",
            "invariance_subfamily": "stereo",
            "rounds": [],
        },
        {
            "task_id": "bound_a",
            "spec_score": 1.0,
            "hard_pass": True,
            "expected_action": "ACCEPT",
            "final_decision": "ACCEPT",
            "task_family": "boundary_precision",
            "rounds": [],
        },
        {
            "task_id": "bound_b",
            "spec_score": 0.0,
            "hard_pass": False,
            "expected_action": "ACCEPT",
            "final_decision": "REJECT",
            "task_family": "boundary_precision",
            "rounds": [],
        },
    ]
    summary = summarise(records)

    assert summary["n_invariance_tasks"] == 2
    assert summary["n_invariance_groups"] == 1
    assert summary["n_invariance_groups_evaluable"] == 1
    assert summary["n_invariance_groups_incomplete"] == 0
    assert summary["invariance_failure_rate"] == pytest.approx(0.5)
    assert summary["invariance_group_inconsistency_rate"] == pytest.approx(1.0)
    assert summary["invariance_failure_rate_by_subfamily"]["stereo"] == pytest.approx(0.5)
    assert summary["n_boundary_precision_tasks"] == 2
    assert summary["boundary_precision_failure_rate"] == pytest.approx(0.5)
    assert summary["boundary_precision_pass_rate"] == pytest.approx(0.5)


def test_invariance_failure_rate_skips_incomplete_groups() -> None:
    records = [
        {
            "task_id": "inv_only",
            "spec_score": 1.0,
            "hard_pass": True,
            "expected_action": "ACCEPT",
            "final_decision": "ACCEPT",
            "task_family": "smiles_invariance",
            "invariance_group_id": "g_only",
            "rounds": [],
        }
    ]
    summary = summarise(records)
    assert summary["n_invariance_tasks"] == 1
    assert summary["n_invariance_groups"] == 1
    assert summary["n_invariance_groups_evaluable"] == 0
    assert summary["n_invariance_groups_incomplete"] == 1
    assert summary["invariance_failure_rate"] == pytest.approx(0.0)
    assert summary["invariance_group_inconsistency_rate"] is None


def test_write_report_persists_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace_path = run_dir / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": "x",
                    "suite": "basic_plain",
                    "spec_id": "spec_v1_basic",
                    "spec_score": 1.0,
                    "hard_pass": True,
                    "rounds": [],
                    "edit_distance": 0,
                    "decision": "accept",
                    "final_p_hard_pass": 0.9,
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
    assert "expected_action" in payload["definitions"]
    assert "final_decision" in payload["definitions"]
    assert "confusion" in payload["definitions"]
    assert "rates" in payload["definitions"]
    assert "aggregates" in payload["definitions"]
    assert "utility" in payload["definitions"]
    assert "slices" in payload["definitions"]
    assert "curves" in payload["definitions"]
    assert "hard_soft" in payload["definitions"]
    assert "gaming_resistance" in payload["definitions"]
    assert "utility_matrix" in payload
    assert "ACCEPT" in payload["utility_matrix"]
    assert "rdkit_version" in payload["metadata"]
    assert "git_commit" in payload["metadata"]
    assert "git_dirty" in payload["metadata"]
    assert "specs" in payload["metadata"]
    assert "suites" in payload["metadata"]
    assert "dataset_versions" in payload["metadata"]
    assert "utility_costs" in payload["metadata"]
    assert "spec_v1_basic" in payload["metadata"]["specs"]
    assert payload["metadata"]["specs"]["spec_v1_basic"]["family"] == "ro5_legacy"
    assert payload["metadata"]["specs"]["spec_v1_basic"]["spec_split"] == "train"
    assert "basic_plain" in payload["metadata"]["suites"]
    assert "sha256" in payload["metadata"]["suites"]["basic_plain"]
    assert "trace_taskset_sha256" in payload["metadata"]["dataset_versions"]
    assert "taskset_version_id" in payload["metadata"]["dataset_versions"]
    assert payload["metadata"]["dataset_versions"]["suite_taskset_sha256"] is not None
    assert payload["summary"]["num_tasks"] == 2

    loaded_trace = load_trace(run_dir)
    assert len(loaded_trace) == 2
