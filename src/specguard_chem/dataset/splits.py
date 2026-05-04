from __future__ import annotations

"""Deterministic bundle-aware split assignment for sgchem_v1."""

import hashlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping

RELEASE_SPLITS: tuple[str, ...] = ("train", "dev", "test")
DEFAULT_SPLIT_PROPORTIONS: Mapping[str, float] = {
    "train": 0.60,
    "dev": 0.20,
    "test": 0.20,
}


def _split_hash(bundle_id: str, seed: int) -> str:
    payload = f"{seed}:{bundle_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_bundle_splits(
    bundles: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    proportions: Mapping[str, float] = DEFAULT_SPLIT_PROPORTIONS,
) -> Dict[str, str]:
    """Assign whole bundles to splits with deterministic approximate proportions."""

    rows = sorted(
        [dict(bundle) for bundle in bundles],
        key=lambda item: (_split_hash(str(item.get("bundle_id", "")), seed), str(item.get("bundle_id", ""))),
    )
    total = len(rows)
    if total == 0:
        return {}

    train_count = int(round(total * float(proportions.get("train", 0.60))))
    dev_count = int(round(total * float(proportions.get("dev", 0.20))))
    train_count = max(0, min(train_count, total))
    dev_count = max(0, min(dev_count, total - train_count))
    test_count = total - train_count - dev_count
    counts = {"train": train_count, "dev": dev_count, "test": test_count}

    split_by_bundle: Dict[str, str] = {}
    cursor = 0
    for split in RELEASE_SPLITS:
        for row in rows[cursor : cursor + counts[split]]:
            split_by_bundle[str(row["bundle_id"])] = split
        cursor += counts[split]
    return split_by_bundle


def split_tasks_by_bundle(
    tasks: Iterable[Mapping[str, Any]],
    split_by_bundle: Mapping[str, str],
) -> Dict[str, list[Dict[str, Any]]]:
    by_split: Dict[str, list[Dict[str, Any]]] = {split: [] for split in RELEASE_SPLITS}
    for task in tasks:
        bundle_id = str(task.get("bundle_id", ""))
        split = split_by_bundle.get(bundle_id, "test")
        row = dict(task)
        row["split"] = split
        by_split.setdefault(split, []).append(row)
    for split in RELEASE_SPLITS:
        by_split[split].sort(key=lambda item: str(item.get("task_id", "")))
    return by_split


def split_bundles(
    bundles: Iterable[Mapping[str, Any]],
    split_by_bundle: Mapping[str, str],
) -> Dict[str, list[Dict[str, Any]]]:
    by_split: Dict[str, list[Dict[str, Any]]] = {split: [] for split in RELEASE_SPLITS}
    for bundle in bundles:
        bundle_id = str(bundle.get("bundle_id", ""))
        split = split_by_bundle.get(bundle_id, "test")
        row = dict(bundle)
        row["split"] = split
        by_split.setdefault(split, []).append(row)
    for split in RELEASE_SPLITS:
        by_split[split].sort(key=lambda item: str(item.get("bundle_id", "")))
    return by_split


def split_leakage_summary(
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    bundles_by_split: Mapping[str, list[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Return cross-split overlap counts for strict validation and audit reports."""

    def _owners_for(field: str, rows_by_split: Mapping[str, list[Mapping[str, Any]]]) -> Dict[str, set[str]]:
        owners: Dict[str, set[str]] = defaultdict(set)
        for split, rows in rows_by_split.items():
            for row in rows:
                value = row.get(field)
                if isinstance(value, str) and value:
                    owners[value].add(split)
        return owners

    bundle_owners = _owners_for("bundle_id", tasks_by_split)
    agent_hash_owners = _owners_for("agent_visible_hash", tasks_by_split)

    group_owners: Dict[str, Dict[str, set[str]]] = {
        "invariance_group_id": defaultdict(set),
        "boundary_group_id": defaultdict(set),
        "interrupt_group_id": defaultdict(set),
    }
    canonical_input_spec: Dict[str, set[str]] = defaultdict(set)
    canonical_witness_spec: Dict[str, set[str]] = defaultdict(set)
    scaffold_owners: Dict[str, set[str]] = defaultdict(set)

    for split, rows in tasks_by_split.items():
        for row in rows:
            evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
            for field in group_owners:
                value = evidence.get(field)
                if isinstance(value, str) and value:
                    group_owners[field][value].add(split)
            generation = row.get("generation") if isinstance(row.get("generation"), dict) else {}
            scaffold = generation.get("scaffold_hash") or row.get("scaffold_hash")
            if isinstance(scaffold, str) and scaffold:
                scaffold_owners[scaffold].add(split)
            input_block = row.get("input") if isinstance(row.get("input"), dict) else {}
            input_smiles = input_block.get("smiles") or input_block.get("candidate_smiles")
            spec_key = row.get("spec_instance_hash") or row.get("spec_id")
            if isinstance(input_smiles, str) and input_smiles:
                canonical_input_spec[f"{spec_key}::{input_smiles}"].add(split)
            witness = evidence.get("feasible_witness_canonical_smiles") or evidence.get("feasible_witness_smiles")
            if isinstance(witness, str) and witness:
                canonical_witness_spec[f"{spec_key}::{witness}"].add(split)

    bundle_file_owners = _owners_for("bundle_id", bundles_by_split)

    def _cross(owners: Mapping[str, set[str]]) -> list[str]:
        return sorted(key for key, splits in owners.items() if len(splits) > 1)

    split_counts = {
        split: len(tasks_by_split.get(split, [])) for split in RELEASE_SPLITS
    }
    bundle_counts = {
        split: len(bundles_by_split.get(split, [])) for split in RELEASE_SPLITS
    }
    total_bundles = sum(bundle_counts.values())
    bundle_props = {
        split: (bundle_counts[split] / total_bundles if total_bundles else 0.0)
        for split in RELEASE_SPLITS
    }

    return {
        "bundle_cross_split": len(_cross(bundle_owners)),
        "bundle_file_cross_split": len(_cross(bundle_file_owners)),
        "agent_visible_cross_split": len(_cross(agent_hash_owners)),
        "invariance_cross_split": len(_cross(group_owners["invariance_group_id"])),
        "boundary_cross_split": len(_cross(group_owners["boundary_group_id"])),
        "interrupt_cross_split": len(_cross(group_owners["interrupt_group_id"])),
        "canonical_input_spec_cross_split": len(_cross(canonical_input_spec)),
        "canonical_witness_spec_cross_split": len(_cross(canonical_witness_spec)),
        "scaffold_cross_split": len(_cross(scaffold_owners)),
        "bundle_overlap_examples": _cross(bundle_owners)[:20],
        "agent_visible_overlap_examples": _cross(agent_hash_owners)[:20],
        "canonical_input_spec_examples": _cross(canonical_input_spec)[:20],
        "canonical_witness_spec_examples": _cross(canonical_witness_spec)[:20],
        "scaffold_overlap_examples": _cross(scaffold_owners)[:20],
        "split_task_counts": split_counts,
        "split_bundle_counts": bundle_counts,
        "split_bundle_proportions": bundle_props,
    }


def split_proportions_within_tolerance(
    bundle_counts: Mapping[str, int],
    *,
    proportions: Mapping[str, float] = DEFAULT_SPLIT_PROPORTIONS,
    tolerance: float = 0.15,
) -> bool:
    total = sum(int(value) for value in bundle_counts.values())
    if total == 0:
        return False
    for split in RELEASE_SPLITS:
        observed = int(bundle_counts.get(split, 0)) / total
        expected = float(proportions.get(split, 0.0))
        if abs(observed - expected) > tolerance:
            return False
    return True


def count_splits(values: Iterable[str]) -> Dict[str, int]:
    counts = Counter(values)
    return {split: int(counts.get(split, 0)) for split in RELEASE_SPLITS}
