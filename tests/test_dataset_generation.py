from __future__ import annotations

from pathlib import Path

from specguard_chem.config import load_spec
from specguard_chem.dataset import (
    build_corpus_records,
    compute_corpus_sha256,
    generate_tasks_from_corpus,
    load_corpus_records,
    validate_dataset_records,
    write_corpus_records,
)


def test_build_corpus_is_deterministic() -> None:
    records_a = build_corpus_records(seed=11, max_molecules=180, reaction_depth=1)
    records_b = build_corpus_records(seed=11, max_molecules=180, reaction_depth=1)

    assert len(records_a) == len(records_b)
    assert len(records_a) >= 120
    assert compute_corpus_sha256(records_a) == compute_corpus_sha256(records_b)
    assert records_a[0]["canonical_smiles"] == records_b[0]["canonical_smiles"]


def test_write_and_load_corpus_roundtrip(tmp_path: Path) -> None:
    records = build_corpus_records(seed=5, max_molecules=80, reaction_depth=1)
    target = tmp_path / "corpus.parquet"
    written = write_corpus_records(target, records)

    assert written.exists()
    loaded = load_corpus_records(target)
    assert len(loaded) == len(records)
    assert compute_corpus_sha256(loaded) == compute_corpus_sha256(records)


def test_generate_tasks_and_validate_invariants() -> None:
    corpus = build_corpus_records(seed=3, max_molecules=220, reaction_depth=1)
    tasks = generate_tasks_from_corpus(
        corpus_records=corpus,
        specs=[load_spec("spec_v1_basic")],
        target_tasks=160,
        seed=3,
        suite_name="generated_test",
    )
    assert len(tasks) == 160
    assert len({task["task_id"] for task in tasks}) == 160

    result = validate_dataset_records(tasks)
    assert result["valid"] is True
    assert result["num_errors"] == 0
    assert result["family_counts"]["feasible_propose"] > 0
    assert result["family_counts"]["repair_near_miss"] > 0
    assert result["family_counts"]["repair_multi_violation"] > 0
    assert result["family_counts"]["contradiction_abstain"] > 0
    assert result["family_counts"]["smiles_invariance"] > 0
    assert result["family_counts"]["boundary_precision"] > 0
    assert result["family_counts"]["tool_forced_l3"] > 0


def test_validator_detects_bad_witness() -> None:
    corpus = build_corpus_records(seed=9, max_molecules=120, reaction_depth=1)
    tasks = generate_tasks_from_corpus(
        corpus_records=corpus,
        specs=[load_spec("spec_v1_basic")],
        target_tasks=40,
        seed=9,
        suite_name="generated_bad_witness",
    )
    for task in tasks:
        if task.get("expected_action") == "ACCEPT":
            task["evidence"]["feasible_witness_smiles"] = "INVALID"
            break
    result = validate_dataset_records(tasks)
    assert result["valid"] is False
    assert result["num_errors"] >= 1


def test_validator_detects_broken_invariance_input() -> None:
    corpus = build_corpus_records(seed=15, max_molecules=180, reaction_depth=1)
    tasks = generate_tasks_from_corpus(
        corpus_records=corpus,
        specs=[load_spec("spec_v1_basic")],
        target_tasks=120,
        seed=15,
        suite_name="generated_bad_invariance",
    )
    for task in tasks:
        if task.get("task_family") == "smiles_invariance":
            task["input"]["smiles"] = "CCCC"
            break
    result = validate_dataset_records(tasks)
    assert result["valid"] is False
    assert any("invariance" in message for message in result["errors"])
