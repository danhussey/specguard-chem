from __future__ import annotations

from pathlib import Path

from specguard_chem.benchmark.sweep import load_baseline_matrix, run_benchmark_sweep


def test_baseline_tracks_are_separated() -> None:
    matrix = load_baseline_matrix(Path("baselines/paper_baselines.yaml"))
    by_name = {entry.name: entry for entry in matrix}
    assert by_name["always_accept"].track == "primary_closed_book"
    assert by_name["always_reject"].track == "primary_closed_book"
    assert by_name["always_abstain"].track == "primary_closed_book"
    assert by_name["corpus_retrieval_upper_bound"].track == "retrieval_upper_bound"
    assert by_name["verify_first"].track == "tool_enabled"


def test_sanity_baselines_have_expected_failure_modes(v1_release: Path, tmp_path: Path) -> None:
    aggregate = run_benchmark_sweep(
        benchmark_dir=v1_release,
        split="test",
        baselines_path=Path("baselines/paper_baselines.yaml"),
        out_dir=tmp_path / "runs",
        seed=7,
        limit=30,
        n_bootstrap=10,
    )
    rows = {row["name"]: row for row in aggregate["baselines"]}
    always_accept = rows["always_accept"]["metrics"]
    always_reject = rows["always_reject"]["metrics"]
    always_abstain = rows["always_abstain"]["metrics"]
    assert always_accept["task_inconsistent_accept_rate"] is not None
    assert always_accept["correct_reject_rate"] < 1.0
    assert always_reject["accept_rate"] < 1.0
    assert always_abstain["accept_rate"] == 0.0
    assert rows["corpus_retrieval_upper_bound"]["track"] == "retrieval_upper_bound"
