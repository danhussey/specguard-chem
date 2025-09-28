from __future__ import annotations

"""Core scoring utilities."""

from typing import Iterable, Mapping, Sequence


Decision = str


def hard_violation_rate(records: Sequence[Mapping[str, object] | bool]) -> float:
    """Fraction of tasks with at least one hard violation."""

    if not records:
        return 0.0
    violations = 0
    for record in records:
        if isinstance(record, bool):
            violations += 0 if record else 1
        else:
            hard_pass = record.get("hard_pass")
            if hard_pass is None:
                continue
            violations += 0 if hard_pass else 1
    return violations / len(records)


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
    return (1.0 if hard_pass else 0.0) + lam * soft_component


def abstention_utility(
    truths: Sequence[bool], decisions: Sequence[Decision], costs: Mapping[str, float]
) -> float:
    """Compute negative expected loss under the abstention cost model."""

    penalties = 0.0
    c_fa = costs.get("false_accept", 5.0)
    c_fr = costs.get("false_reject", 1.0)
    c_abstain = costs.get("abstain", 0.5)
    for truth, decision in zip(truths, decisions):
        if decision == "accept":
            if not truth:
                penalties += c_fa
        elif decision == "reject":
            if truth:
                penalties += c_fr
        else:  # treat everything else as abstain
            penalties += c_abstain
    return -penalties
