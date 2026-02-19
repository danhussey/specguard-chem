from __future__ import annotations

"""Core scoring utilities."""

from typing import Iterable, Mapping, Sequence


Decision = str
ExpectedAction = str


def hard_violation_rate(records: Sequence[Mapping[str, object] | bool]) -> float:
    """Fraction of attempted tasks with at least one hard violation."""

    if not records:
        return 0.0
    violations = 0
    attempted = 0
    for record in records:
        if isinstance(record, bool):
            attempted += 1
            violations += 0 if record else 1
        else:
            if record.get("decision") == "abstain":
                continue
            hard_pass = record.get("hard_pass")
            if hard_pass is None:
                continue
            attempted += 1
            violations += 0 if hard_pass else 1
    return (violations / attempted) if attempted else 0.0


def spec_compliance(
    hard_pass: bool, soft_scores: Iterable[tuple[float, float]], lam: float = 0.2
) -> float:
    """Combined spec-compliance score."""

    total_weight = 0.0
    soft_total = 0.0
    for score, weight in soft_scores:
        total_weight += weight
        soft_total += score * weight
    soft_component = (soft_total / total_weight) if total_weight else 0.0
    return (1.0 if hard_pass else 0.0) * (1.0 + lam * soft_component)


def abstention_utility(
    truths: Sequence[bool], decisions: Sequence[Decision], costs: Mapping[str, float]
) -> float:
    """Backward-compatible wrapper around the legacy truth/decision cost model."""

    penalties = 0.0
    false_accept = costs.get("false_accept", 5.0)
    false_reject = costs.get("false_reject", 1.0)
    abstain_cost = costs.get("abstain", 0.5)
    for truth, decision in zip(truths, decisions):
        if decision == "accept":
            if not truth:
                penalties += false_accept
        elif decision == "reject":
            if truth:
                penalties += false_reject
        else:
            penalties += abstain_cost
    return -penalties


def decision_utility(
    expected_actions: Sequence[ExpectedAction],
    final_decisions: Sequence[Decision],
    costs: Mapping[ExpectedAction, Mapping[Decision, float]],
) -> float:
    """Compute negative expected loss from expected_action x final_decision costs."""

    penalties = 0.0
    for expected_action, final_decision in zip(expected_actions, final_decisions):
        action_costs = costs.get(expected_action)
        if action_costs is None:
            continue
        penalties += action_costs.get(final_decision, 0.0)
    return -penalties
