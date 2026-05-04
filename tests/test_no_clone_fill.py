from __future__ import annotations

from collections import Counter

from specguard_chem.config import load_spec
from specguard_chem.dataset import build_corpus_records, generate_tasks_from_corpus


def test_no_clone_fill_offsets_in_legacy_generator() -> None:
    corpus = build_corpus_records(seed=7, max_molecules=120, reaction_depth=1)
    tasks = generate_tasks_from_corpus(
        corpus_records=corpus,
        specs=[load_spec("spec_v1_basic")],
        target_tasks=500,
        seed=7,
        suite_name="clone_check",
    )
    assert len(tasks) <= 500
    assert all("100000" not in task["task_id"] for task in tasks)
    assert len({task["task_id"] for task in tasks}) == len(tasks)


def test_no_unmarked_agent_visible_duplicates_and_manifest_counts(
    v1_tasks: list[dict], v1_manifest: dict
) -> None:
    hashes = Counter(task["agent_visible_hash"] for task in v1_tasks)
    duplicates = {digest for digest, count in hashes.items() if count > 1}
    for digest in duplicates:
        paired = [task for task in v1_tasks if task["agent_visible_hash"] == digest]
        assert all(task.get("intentional_pair") for task in paired)
    assert v1_manifest["requested_tasks"] == 40
    assert v1_manifest["generated_tasks"] == len(v1_tasks)
    assert "generation_shortfall" in v1_manifest
    assert "generation_shortfall_reason" in v1_manifest
