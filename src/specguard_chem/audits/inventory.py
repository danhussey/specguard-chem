from __future__ import annotations

"""Task inventory audit tables."""

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def inventory_rows(tasks_by_split: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
            rows.append(
                {
                    "split": split,
                    "task_id": task.get("task_id"),
                    "bundle_id": task.get("bundle_id"),
                    "task_type": task.get("task_type") or task.get("task_family"),
                    "expected_action": task.get("expected_action"),
                    "protocol": task.get("protocol"),
                    "spec_id": task.get("spec_id"),
                    "oracle_type": task.get("oracle_type"),
                    "agent_visible_hash": task.get("agent_visible_hash"),
                    "boundary_group_id": evidence.get("boundary_group_id"),
                    "invariance_group_id": evidence.get("invariance_group_id"),
                    "interrupt_group_id": evidence.get("interrupt_group_id"),
                }
            )
    rows.sort(key=lambda item: (str(item["split"]), str(item["task_id"])))
    return rows


def inventory_summary(
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    bundles_by_split: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = inventory_rows(tasks_by_split)
    task_type_counts = Counter(str(row["task_type"]) for row in rows)
    expected_counts = Counter(str(row["expected_action"]) for row in rows)
    protocol_counts = Counter(str(row["protocol"]) for row in rows)
    spec_counts = Counter(str(row["spec_id"]) for row in rows)
    oracle_counts = Counter(str(row["oracle_type"]) for row in rows)
    task_split_counts = {split: len(tasks_by_split.get(split, [])) for split in ("train", "dev", "test")}
    bundle_split_counts = {split: len(bundles_by_split.get(split, [])) for split in ("train", "dev", "test")}
    boundary_groups = {str(row["boundary_group_id"]) for row in rows if row.get("boundary_group_id")}
    invariance_groups = {str(row["invariance_group_id"]) for row in rows if row.get("invariance_group_id")}
    interrupt_groups = {str(row["interrupt_group_id"]) for row in rows if row.get("interrupt_group_id")}
    hash_counts = Counter(str(row["agent_visible_hash"]) for row in rows if row.get("agent_visible_hash"))
    duplicate_hashes = sum(1 for count in hash_counts.values() if count > 1)
    return {
        "total_tasks": len(rows),
        "total_bundles": sum(bundle_split_counts.values()),
        "tasks_per_split": task_split_counts,
        "bundles_per_split": bundle_split_counts,
        "tasks_per_task_type": dict(sorted(task_type_counts.items())),
        "tasks_per_expected_action": dict(sorted(expected_counts.items())),
        "tasks_per_protocol": dict(sorted(protocol_counts.items())),
        "tasks_per_spec": dict(sorted(spec_counts.items())),
        "tasks_per_oracle_type": dict(sorted(oracle_counts.items())),
        "boundary_groups": len(boundary_groups),
        "invariance_groups": len(invariance_groups),
        "interrupt_groups": len(interrupt_groups),
        "duplicate_hashes": duplicate_hashes,
    }


def write_inventory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "task_id",
        "bundle_id",
        "task_type",
        "expected_action",
        "protocol",
        "spec_id",
        "oracle_type",
        "agent_visible_hash",
        "boundary_group_id",
        "invariance_group_id",
        "interrupt_group_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_inventory_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Task Inventory Summary",
        "",
        f"total tasks: {summary.get('total_tasks', 0)}",
        f"total bundles: {summary.get('total_bundles', 0)}",
    ]
    for label in (
        "tasks_per_split",
        "bundles_per_split",
        "tasks_per_task_type",
        "tasks_per_expected_action",
        "tasks_per_protocol",
        "tasks_per_spec",
        "tasks_per_oracle_type",
    ):
        lines.extend(["", label.replace("_", " ") + ":"])
        payload = summary.get(label)
        if isinstance(payload, dict):
            for key, value in payload.items():
                lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            f"boundary groups: {summary.get('boundary_groups', 0)}",
            f"invariance groups: {summary.get('invariance_groups', 0)}",
            f"interrupt groups: {summary.get('interrupt_groups', 0)}",
            f"duplicate hashes: {summary.get('duplicate_hashes', 0)}",
        ]
    )
    return "\n".join(lines) + "\n"
