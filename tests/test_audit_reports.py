from __future__ import annotations

from specguard_chem.dataset.validate_v1 import validate_release_v1


def test_expected_audit_reports_exist_and_match_manifest(v1_release, v1_manifest) -> None:
    expected = [
        "task_inventory.csv",
        "task_inventory_summary.md",
        "agent_visible_duplicate_report.md",
        "split_leakage_report.md",
        "oracle_validation_report.md",
        "safety_scope_report.md",
        "curation_summary.md",
    ]
    for name in expected:
        path = v1_release / "audits" / name
        assert path.exists()
        assert path.stat().st_size > 0
    summary = (v1_release / "audits" / "task_inventory_summary.md").read_text(encoding="utf-8")
    assert f"total tasks: {v1_manifest['num_tasks']}" in summary
    assert f"total bundles: {v1_manifest['num_bundles']}" in summary
    validation = validate_release_v1(v1_release, strict=True)
    assert validation["num_errors"] == v1_manifest["strict_validation"]["num_errors"]
