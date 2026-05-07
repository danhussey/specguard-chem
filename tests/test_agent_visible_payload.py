from __future__ import annotations

from copy import deepcopy

from specguard_chem.dataset.oracles import stable_json_hash


def test_agent_visible_hash_excludes_oracle_fields(v1_tasks: list[dict]) -> None:
    task = v1_tasks[0]
    payload = deepcopy(task["agent_visible_payload"])
    assert task["rendered_agent_input"]
    assert task["agent_visible_hash"] == stable_json_hash(payload)

    changed_expected = deepcopy(task)
    changed_expected["expected_action"] = "REJECT"
    assert stable_json_hash(payload) == task["agent_visible_hash"]

    changed_evidence = deepcopy(task)
    changed_evidence["evidence"]["feasible_witness_smiles"] = "C"
    assert stable_json_hash(payload) == task["agent_visible_hash"]

    payload["budgets"]["max_steps"] = payload["budgets"]["max_steps"] + 1
    assert stable_json_hash(payload) != task["agent_visible_hash"]


def test_no_agent_visible_hash_duplicate_across_splits(v1_tasks: list[dict]) -> None:
    owners: dict[str, set[str]] = {}
    for task in v1_tasks:
        owners.setdefault(task["agent_visible_hash"], set()).add(task["split"])
    assert all(len(splits) == 1 for splits in owners.values())
