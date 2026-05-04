from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from specguard_chem.utils import jsonio


def _load_preflight():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preflight_neurips_ed_artifact.py"
    spec = importlib.util.spec_from_file_location("preflight_neurips_ed_artifact", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neurips_artifact_preflight_updates_manifest_and_passes(v1_release: Path, tmp_path: Path) -> None:
    release = tmp_path / "release"
    shutil.copytree(v1_release, release)
    summary = _load_preflight().run_preflight(release)
    assert summary["valid"] is True
    assert summary["anonymous_scan_passed"] is True
    assert summary["croissant_local_validation_passed"] is True
    manifest = jsonio.read_json(release / "MANIFEST.json")
    assert manifest["neurips_ed_preflight"]["anonymous_scan_passed"] is True
    assert manifest["dataset_url"].startswith("PENDING")
    checksums = (release / "checksums" / "sha256sums.txt").read_text(encoding="utf-8")
    assert "MANIFEST.json" in checksums
