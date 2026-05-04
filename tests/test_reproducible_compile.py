from __future__ import annotations

from specguard_chem.benchmark.compiler import compile_benchmark_release
from specguard_chem.dataset.validate_v1 import load_release_tasks_by_split


def _tasks(path):
    by_split = load_release_tasks_by_split(path)
    return [task for rows in by_split.values() for task in rows]


def test_same_seed_reproduces_ids_hashes_and_checksums(tmp_path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    compile_benchmark_release(
        benchmark_id="sgchem_v1.0",
        out_dir=out_a,
        seed=7,
        target_bundles=8,
        min_tasks=20,
        max_tasks=80,
        anonymous=True,
    )
    compile_benchmark_release(
        benchmark_id="sgchem_v1.0",
        out_dir=out_b,
        seed=7,
        target_bundles=8,
        min_tasks=20,
        max_tasks=80,
        anonymous=True,
    )
    assert [task["task_id"] for task in _tasks(out_a)] == [task["task_id"] for task in _tasks(out_b)]
    assert [task["agent_visible_hash"] for task in _tasks(out_a)] == [task["agent_visible_hash"] for task in _tasks(out_b)]
    assert (out_a / "checksums" / "sha256sums.txt").read_text(encoding="utf-8") == (
        out_b / "checksums" / "sha256sums.txt"
    ).read_text(encoding="utf-8")


def test_different_seed_changes_some_tasks(tmp_path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    compile_benchmark_release(
        benchmark_id="sgchem_v1.0",
        out_dir=out_a,
        seed=7,
        target_bundles=8,
        min_tasks=20,
        max_tasks=80,
        anonymous=True,
    )
    compile_benchmark_release(
        benchmark_id="sgchem_v1.0",
        out_dir=out_b,
        seed=8,
        target_bundles=8,
        min_tasks=20,
        max_tasks=80,
        anonymous=True,
    )
    assert [task["agent_visible_hash"] for task in _tasks(out_a)] != [
        task["agent_visible_hash"] for task in _tasks(out_b)
    ]
