from __future__ import annotations

from specguard_chem.config import TaskModel, load_spec
from specguard_chem.dataset import build_corpus_records, generate_tasks_from_corpus
from specguard_chem.models.heuristic_mutator import HeuristicMutatorAdapter
from specguard_chem.models.abstention_guard import AbstentionGuardAdapter
from specguard_chem.models.corpus_search import CorpusSearchAdapter
from specguard_chem.models.local_mutation import LocalMutationAdapter
from specguard_chem.runner.runner import TaskRunner


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


def test_non_llm_adapters_are_deterministic_for_same_seed():
    spec_payload = load_spec("spec_v1_basic").model_dump(mode="json")
    request = _request(
        round_id=2,
        task={
            "task_id": "deterministic",
            "task_family": "repair_near_miss",
            "input": {"smiles": "CCN"},
            "protocol": "L2",
            "spec_id": "spec_v1_basic",
        },
        spec=spec_payload,
        failure_vector={"hard_fail_ids": ["logP_bounds"]},
    )
    corpus_a = CorpusSearchAdapter(seed=17).step(request)
    corpus_b = CorpusSearchAdapter(seed=17).step(request)
    mutation_a = LocalMutationAdapter(seed=17).step(request)
    mutation_b = LocalMutationAdapter(seed=17).step(request)

    assert corpus_a["smiles"] == corpus_b["smiles"]
    assert mutation_a["smiles"] == mutation_b["smiles"]


def test_non_llm_adapters_respect_runner_budgets():
    task = TaskModel.model_validate(
        {
            "task_id": "budget_non_llm",
            "suite": "unit",
            "protocol": "L2",
            "prompt": "repair",
            "input": {"smiles": "CCN"},
            "spec_id": "spec_v1_basic",
            "scoring": {"primary": "spec_compliance"},
            "task_family": "repair_near_miss",
            "budgets": {
                "max_steps": 1,
                "max_proposals": 1,
                "max_verify_calls": 0,
                "max_total_verifier_calls": 1,
            },
            "expected": "PASS",
        }
    )

    corpus_runner = TaskRunner("corpus_search", seed=7)
    local_runner = TaskRunner("local_mutation", seed=7)
    corpus_record = corpus_runner.run_tasks([task], suite="unit", protocol="L2")[0]
    local_record = local_runner.run_tasks([task], suite="unit", protocol="L2")[0]

    assert corpus_record.steps_used <= 1
    assert local_record.steps_used <= 1
    assert corpus_record.proposals_used <= 1
    assert local_record.proposals_used <= 1


def test_non_llm_adapters_solve_near_miss_task():
    corpus = build_corpus_records(seed=19, max_molecules=220, reaction_depth=1)
    tasks = generate_tasks_from_corpus(
        corpus_records=corpus,
        specs=[load_spec("spec_v1_basic")],
        target_tasks=80,
        seed=19,
        suite_name="near_miss_unit",
    )
    near_miss = next(
        task for task in tasks if task.get("task_family") == "repair_near_miss"
    )
    model_task = TaskModel.model_validate(near_miss)

    corpus_runner = TaskRunner("corpus_search", seed=7)
    local_runner = TaskRunner("local_mutation", seed=7)
    corpus_record = corpus_runner.run_tasks([model_task], suite="unit", protocol="L2")[0]
    local_record = local_runner.run_tasks([model_task], suite="unit", protocol="L2")[0]

    assert corpus_record.hard_pass is True
    assert local_record.hard_pass is True
