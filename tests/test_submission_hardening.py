from __future__ import annotations

from specguard_chem.audits.submission import (
    challenge_summary,
    detailed_baseline_metrics,
    render_metric_definitions,
)
from specguard_chem.dataset.validate_v1 import load_release_tasks_by_split


def test_difficulty_tags_are_structural_metadata(v1_tasks: list[dict]) -> None:
    assert any(task.get("difficulty_tags") for task in v1_tasks)
    for task in v1_tasks:
        assert task.get("challenge_slice") == any(
            tag != "mixed_hard_soft_tradeoff" for tag in task.get("difficulty_tags", [])
        )
        rendered = task.get("rendered_agent_input", "")
        for tag in task.get("difficulty_tags", []):
            assert tag not in rendered


def test_challenge_summary_uses_task_metadata(v1_release) -> None:
    tasks_by_split = load_release_tasks_by_split(v1_release)
    summary = challenge_summary(tasks_by_split)
    assert summary["num_test_tasks"] == len(tasks_by_split["test"])
    assert summary["num_challenge_tasks"] <= summary["num_test_tasks"]
    assert set(summary["difficulty_tag_counts"]).issubset(
        {
            "tight_property_boundary",
            "multi_constraint_violation",
            "minimal_edit_required",
            "low_margin_feasible",
            "reject_near_miss",
            "abstain_explicit_contradiction",
            "invariance_equivalent_representation",
            "tool_required_by_protocol",
            "interrupt_state_required",
            "high_similarity_guard",
            "mixed_hard_soft_tradeoff",
        }
    )


def test_detailed_baseline_metrics_exposes_denominators() -> None:
    records = [
        {
            "expected_action": "ACCEPT",
            "final_decision": "ACCEPT",
            "hard_pass": True,
            "steps_used": 1,
            "rounds": [{"action": "propose", "evaluation": {"hard_pass": True, "properties": {"MW": 100}}}],
        },
        {
            "expected_action": "REJECT",
            "final_decision": "ACCEPT",
            "hard_pass": True,
            "steps_used": 1,
            "rounds": [{"action": "propose", "evaluation": {"hard_pass": True, "properties": {"MW": 100}}}],
        },
        {
            "expected_action": "ABSTAIN",
            "final_decision": "ABSTAIN",
            "hard_pass": False,
            "steps_used": 1,
            "rounds": [{"action": "abstain"}],
        },
    ]
    metrics = detailed_baseline_metrics(records)
    assert metrics["overall_task_success"] == 2 / 3
    assert metrics["task_inconsistent_accept_rate"] == 0.5
    assert metrics["task_inconsistent_accept_denominator"] == 2
    assert metrics["pass_at_1_denominator"] == 1
    assert metrics["molecule_acceptance_rate"] == 2 / 3


def test_metric_definitions_demote_internal_accept_rate() -> None:
    text = render_metric_definitions()
    assert "molecule_acceptance_rate" in text
    assert "formerly the internal accept_rate" in text
    assert "not task success" in text
