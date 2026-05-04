from __future__ import annotations

import subprocess
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_hosting_packet_files_are_present_and_double_blind_safe() -> None:
    if not (ROOT / "hosting").exists():
        pytest.skip("hosting instructions are maintained outside the anonymous artifact archive")
    required = [
        ROOT / "hosting" / "README_for_upload.md",
        ROOT / "hosting" / "DATASET_CARD.md",
        ROOT / "hosting" / "HUGGINGFACE_DATASET_CARD.md",
        ROOT / "hosting" / "upload_manifest.md",
        ROOT / "hosting" / "post_upload_checklist.md",
    ]
    for path in required:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert "SpecGuard-Chem sgchem_v1.0" in text
        assert "does not evaluate biological activity" in text or path.name in {
            "README_for_upload.md",
            "upload_manifest.md",
            "post_upload_checklist.md",
        }
        assert "PENDING_ANONYMOUS_HOSTED_URL" not in text
        assert "".join(("/", "Users", "/")) not in text
        assert "".join(("github.com/", "dan", "hus", "sey")) not in text


def test_paper_skeleton_has_required_sections() -> None:
    required = [
        ROOT / "paper" / "specguard_chem_neurips2026.md",
        ROOT / "paper" / "sections" / "00_abstract.md",
        ROOT / "paper" / "sections" / "08_results.md",
        ROOT / "paper" / "sections" / "10_limitations_and_safety.md",
        ROOT / "paper" / "appendix" / "prompt_leakage.md",
        ROOT / "paper_v1" / "abstract_variants.md",
        ROOT / "paper_v1" / "artifact_links.md",
    ]
    for path in required:
        assert path.exists(), path
    text = "\n".join(path.read_text(encoding="utf-8") for path in required)
    manifest = json.loads((ROOT / "benchmarks" / "releases" / "sgchem_v1.0" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert f"{manifest['num_bundles']} bundles" in text
    assert f"{manifest['num_tasks']} tasks" in text
    assert f"{manifest['splits']['test']['tasks']} test tasks" in text
    assert "molecule_acceptance_rate=" in text
    assert "unsafe_accept_rate=" in text
    assert "does not evaluate biological activity" in text


def test_paper_consistency_rc_mode_passes() -> None:
    if not (ROOT / "runs" / "paper_sweeps" / "sgchem_v1.0_test" / "aggregate.json").exists():
        pytest.skip("paper consistency baseline comparison requires generated run aggregate")
    completed = subprocess.run(
        ["python", "scripts/check_paper_consistency.py", "--mode", "rc"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout
