from __future__ import annotations

"""Split leakage audit rendering."""

from typing import Any, Mapping

from ..dataset.splits import split_leakage_summary


def leakage_summary(
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    bundles_by_split: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    return split_leakage_summary(tasks_by_split, bundles_by_split)


def render_split_leakage_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Split Leakage Report",
        "",
        f"bundle overlap across splits: {summary.get('bundle_cross_split', 0)}",
        f"scaffold overlap across splits: {summary.get('scaffold_cross_split', 0)}",
        f"canonical input/spec overlap across splits: {summary.get('canonical_input_spec_cross_split', 0)}",
        f"witness/spec overlap across splits: {summary.get('canonical_witness_spec_cross_split', 0)}",
        f"agent-visible hash overlap across splits: {summary.get('agent_visible_cross_split', 0)}",
        f"invariance group leakage: {summary.get('invariance_cross_split', 0)}",
        f"boundary group leakage: {summary.get('boundary_cross_split', 0)}",
        f"interrupt group leakage: {summary.get('interrupt_cross_split', 0)}",
    ]
    for key, title in (
        ("bundle_overlap_examples", "Bundle overlap examples"),
        ("agent_visible_overlap_examples", "Agent-visible overlap examples"),
        ("canonical_input_spec_examples", "Canonical input/spec examples"),
        ("canonical_witness_spec_examples", "Witness/spec examples"),
        ("scaffold_overlap_examples", "Scaffold overlap examples"),
    ):
        examples = summary.get(key)
        if isinstance(examples, list) and examples:
            lines.extend(["", f"{title}:"])
            for item in examples[:20]:
                lines.append(f"- {item}")
    return "\n".join(lines) + "\n"
