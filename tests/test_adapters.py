from __future__ import annotations

from specguard_chem.models.heuristic_mutator import HeuristicMutatorAdapter
from specguard_chem.models.abstention_guard import AbstentionGuardAdapter


def _request(round_id: int = 1, **overrides):
    req = {
        "task": {"input": {"smiles": "c1ccccc1"}},
        "round": round_id,
        "tools": [],
        "failure_vector": None,
    }
    req.update(overrides)
    return req


def test_heuristic_mutator_switches_scaffold_on_pains():
    adapter = HeuristicMutatorAdapter()
    request = _request(
        round_id=2,
        failure_vector={
            "hard_fails": [{"id": "pains_block"}],
            "soft_misses": [],
            "margins": [],
        },
    )
    response = adapter.step(request)
    assert response["action"] == "propose"
    assert response["smiles"] == "COc1ccccc1O"


def test_abstention_guard_abstains_near_margin():
    adapter = AbstentionGuardAdapter(margin_threshold=0.3)
    request = _request(
        round_id=2,
        failure_vector={
            "hard_fails": [],
            "soft_misses": [{"id": "tpsa_pref", "delta": 0.1}],
            "margins": [{"id": "logP", "distance_to_bound": 0.2}],
        },
    )
    response = adapter.step(request)
    assert response["action"] == "abstain"
    assert "reason" in response


def test_abstention_guard_proposes_when_comfortable():
    adapter = AbstentionGuardAdapter(margin_threshold=0.1)
    request = _request(
        round_id=2,
        failure_vector={
            "hard_fails": [],
            "soft_misses": [],
            "margins": [{"id": "logP", "distance_to_bound": 0.5}],
        },
    )
    response = adapter.step(request)
    assert response["action"] == "propose"
    assert response["smiles"]
