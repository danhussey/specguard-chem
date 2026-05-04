from __future__ import annotations


def test_manifest_counts_match_release_files(v1_release, v1_tasks, v1_bundles, v1_manifest) -> None:
    assert (v1_release / "MANIFEST.json").exists()
    assert v1_manifest["benchmark_id"] == "sgchem_v1.0"
    assert v1_manifest["release_type"] == "oracle_compiled_gold"
    assert v1_manifest["generator"] == "bundle_compiler_v1"
    assert v1_manifest["num_tasks"] == len(v1_tasks)
    assert v1_manifest["num_bundles"] == len(v1_bundles)
    assert v1_manifest["strict_validation"]["valid"] is True
    for key in (
        "tasks_per_expected_action",
        "tasks_per_protocol",
        "tasks_per_task_type",
        "leakage_checks",
        "safety_scope_checks",
        "requested_tasks",
        "generated_tasks",
    ):
        assert key in v1_manifest


def test_checksum_file_covers_manifest_and_release_artifacts(v1_release) -> None:
    checksum_text = (v1_release / "checksums" / "sha256sums.txt").read_text(encoding="utf-8")
    assert "MANIFEST.json" in checksum_text
    assert "tasks/train.jsonl" in checksum_text
    assert "bundles/train.jsonl" in checksum_text
    assert "croissant.json" in checksum_text
