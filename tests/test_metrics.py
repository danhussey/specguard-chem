from __future__ import annotations

from specguard_chem.scoring.metrics import decision_utility


DEFAULT_COSTS = {
    "ACCEPT": {"ACCEPT": 0.0, "ABSTAIN": 1.0, "REJECT": 2.0},
    "ABSTAIN": {"ABSTAIN": 0.0, "REJECT": 1.0, "ACCEPT": 10.0},
}


def test_decision_utility_prefers_accept_for_feasible_tasks() -> None:
    accept = decision_utility(["ACCEPT"], ["ACCEPT"], DEFAULT_COSTS)
    abstain = decision_utility(["ACCEPT"], ["ABSTAIN"], DEFAULT_COSTS)
    reject = decision_utility(["ACCEPT"], ["REJECT"], DEFAULT_COSTS)
    assert accept > abstain > reject


def test_decision_utility_prefers_abstain_for_infeasible_tasks() -> None:
    abstain = decision_utility(["ABSTAIN"], ["ABSTAIN"], DEFAULT_COSTS)
    reject = decision_utility(["ABSTAIN"], ["REJECT"], DEFAULT_COSTS)
    accept = decision_utility(["ABSTAIN"], ["ACCEPT"], DEFAULT_COSTS)
    assert abstain > reject > accept
