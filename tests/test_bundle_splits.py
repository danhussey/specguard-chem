from __future__ import annotations

from collections import defaultdict

from specguard_chem.dataset.splits import split_leakage_summary, split_proportions_within_tolerance
from specguard_chem.dataset.validate_v1 import load_release_bundles_by_split, load_release_tasks_by_split


def test_bundle_and_group_ids_do_not_cross_splits(v1_release) -> None:
    tasks_by_split = load_release_tasks_by_split(v1_release)
    bundles_by_split = load_release_bundles_by_split(v1_release)
    summary = split_leakage_summary(tasks_by_split, bundles_by_split)
    assert summary["bundle_cross_split"] == 0
    assert summary["invariance_cross_split"] == 0
    assert summary["boundary_cross_split"] == 0
    assert summary["interrupt_cross_split"] == 0
    assert summary["agent_visible_cross_split"] == 0
    assert split_proportions_within_tolerance(summary["split_bundle_counts"])


def test_no_bundle_crosses_train_dev_test(v1_bundles: list[dict]) -> None:
    owners: dict[str, set[str]] = defaultdict(set)
    for bundle in v1_bundles:
        owners[bundle["bundle_id"]].add(bundle["split"])
    assert all(len(splits) == 1 for splits in owners.values())
