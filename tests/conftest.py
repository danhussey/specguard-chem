from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from specguard_chem.benchmark.compiler import compile_benchmark_release
from specguard_chem.dataset.validate_v1 import (
    load_release_bundles_by_split,
    load_release_tasks_by_split,
)
from specguard_chem.utils import jsonio


@pytest.fixture(scope="session")
def v1_release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("sgchem_v1") / "release"
    compile_benchmark_release(
        benchmark_id="sgchem_v1.0",
        out_dir=out_dir,
        seed=7,
        target_bundles=12,
        min_tasks=40,
        max_tasks=90,
        anonymous=True,
    )
    return out_dir


@pytest.fixture()
def v1_tasks(v1_release: Path) -> list[dict[str, Any]]:
    by_split = load_release_tasks_by_split(v1_release)
    return [task for rows in by_split.values() for task in rows]


@pytest.fixture()
def v1_bundles(v1_release: Path) -> list[dict[str, Any]]:
    by_split = load_release_bundles_by_split(v1_release)
    return [bundle for rows in by_split.values() for bundle in rows]


@pytest.fixture()
def v1_manifest(v1_release: Path) -> dict[str, Any]:
    return jsonio.read_json(v1_release / "MANIFEST.json")
