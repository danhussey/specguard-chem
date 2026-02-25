from __future__ import annotations

from pathlib import Path

import pytest

from specguard_chem.benchmark.release import (
    freeze_benchmark_release,
    validate_release_directory,
)


@pytest.fixture(scope="module")
def frozen_release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("frozen_release")
    out_dir = root / "sgchem_v0.1"
    freeze_benchmark_release(
        benchmark_id="sgchem_v0.1",
        out_dir=out_dir,
        target_tasks=50,
        seed=7,
    )
    return out_dir


def test_freeze_benchmark_writes_expected_layout(frozen_release: Path) -> None:
    required_paths = [
        frozen_release / "MANIFEST.json",
        frozen_release / "README.md",
        frozen_release / "specs" / "spec_catalog.json",
        frozen_release / "corpus",
        frozen_release / "tasks" / "train.jsonl",
        frozen_release / "tasks" / "dev.jsonl",
        frozen_release / "tasks" / "test.jsonl",
        frozen_release / "checksums" / "sha256sums.txt",
    ]
    for required in required_paths:
        assert required.exists(), f"missing {required}"

    validation = validate_release_directory(frozen_release)
    assert validation["valid"] is True
    assert validation["num_errors"] == 0


def test_freeze_benchmark_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "release_a"
    out_b = tmp_path / "release_b"
    freeze_benchmark_release(
        benchmark_id="sgchem_v0.1",
        out_dir=out_a,
        target_tasks=50,
        seed=7,
    )
    freeze_benchmark_release(
        benchmark_id="sgchem_v0.1",
        out_dir=out_b,
        target_tasks=50,
        seed=7,
    )
    checksums_a = (out_a / "checksums" / "sha256sums.txt").read_text(encoding="utf-8")
    checksums_b = (out_b / "checksums" / "sha256sums.txt").read_text(encoding="utf-8")
    assert checksums_a == checksums_b
