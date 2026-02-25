from __future__ import annotations

from pathlib import Path

from specguard_chem.benchmark.paper import make_paper_artifacts
from specguard_chem.benchmark.release import freeze_benchmark_release
from specguard_chem.benchmark.sweep import run_benchmark_sweep


def test_run_benchmark_sweep_and_paper_artifacts(tmp_path: Path) -> None:
    release_dir = tmp_path / "sgchem_v0.1"
    freeze_benchmark_release(
        benchmark_id="sgchem_v0.1",
        out_dir=release_dir,
        target_tasks=50,
        seed=7,
    )
    baselines_path = tmp_path / "baselines.yaml"
    baselines_path.write_text(
        (
            "baselines:\n"
            "  - name: heuristic\n"
            "    model: heuristic\n"
            "  - name: corpus_search\n"
            "    model: corpus_search\n"
        ),
        encoding="utf-8",
    )
    sweep_dir = tmp_path / "sweep"
    aggregate = run_benchmark_sweep(
        benchmark_dir=release_dir,
        split="test",
        baselines_path=baselines_path,
        out_dir=sweep_dir,
        seed=7,
        limit=25,
    )
    assert aggregate["n_baselines"] == 2
    assert (sweep_dir / "aggregate.json").exists()
    for row in aggregate["baselines"]:
        assert (sweep_dir / row["run_dir"] / "trace.jsonl").exists()
        assert (sweep_dir / row["report_path"]).exists()

    paper_dir = tmp_path / "paper"
    result = make_paper_artifacts(out_dir=paper_dir, runs_dir=sweep_dir)
    assert Path(result["summary_path"]).exists()
    assert any((paper_dir / "figures").glob("*.png"))
    assert any((paper_dir / "tables").glob("*.csv"))

