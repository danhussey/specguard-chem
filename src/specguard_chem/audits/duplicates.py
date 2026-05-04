from __future__ import annotations

"""Agent-visible duplicate audits."""

from collections import defaultdict
from typing import Any, Mapping


def duplicate_summary(tasks_by_split: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            digest = task.get("agent_visible_hash")
            if not isinstance(digest, str) or not digest:
                continue
            owners[digest].append(
                {
                    "split": split,
                    "task_id": task.get("task_id"),
                    "intentional_pair": bool(task.get("intentional_pair")),
                }
            )
    duplicate_hashes = {
        digest: rows for digest, rows in owners.items() if len(rows) > 1
    }
    cross_split = {
        digest: rows
        for digest, rows in duplicate_hashes.items()
        if len({str(row["split"]) for row in rows}) > 1
    }
    unmarked_within_split = {
        digest: rows
        for digest, rows in duplicate_hashes.items()
        if len({str(row["split"]) for row in rows}) == 1
        and not all(bool(row.get("intentional_pair")) for row in rows)
    }
    return {
        "duplicate_hashes": len(duplicate_hashes),
        "agent_visible_cross_split": len(cross_split),
        "unmarked_within_split_duplicates": len(unmarked_within_split),
        "cross_split_examples": dict(list(cross_split.items())[:20]),
        "within_split_examples": dict(list(unmarked_within_split.items())[:20]),
    }


def render_duplicate_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Agent-Visible Duplicate Report",
        "",
        f"duplicate hashes: {summary.get('duplicate_hashes', 0)}",
        f"cross-split duplicate hashes: {summary.get('agent_visible_cross_split', 0)}",
        f"unmarked within-split duplicate hashes: {summary.get('unmarked_within_split_duplicates', 0)}",
    ]
    examples = summary.get("cross_split_examples")
    if isinstance(examples, dict) and examples:
        lines.extend(["", "Cross-split examples:"])
        for digest, rows in examples.items():
            lines.append(f"- {digest}: {rows}")
    return "\n".join(lines) + "\n"
