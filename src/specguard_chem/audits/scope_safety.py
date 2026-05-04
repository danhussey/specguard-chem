from __future__ import annotations

"""Scope and safety text checks for agent-visible benchmark tasks."""

import re
from typing import Any, Mapping

FORBIDDEN_AGENT_VISIBLE_RE = re.compile(
    r"\b(disease|activity|potency|toxicity|efficacy|therapeutic|clinical|patient|synthesis route|dosage|target-binding)\b",
    re.IGNORECASE,
)


def scan_agent_visible_scope(tasks_by_split: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    scanned = 0
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            scanned += 1
            text = str(task.get("rendered_agent_input") or task.get("prompt") or "")
            for match in FORBIDDEN_AGENT_VISIBLE_RE.finditer(text):
                matches.append(
                    {
                        "split": split,
                        "task_id": task.get("task_id"),
                        "term": match.group(0),
                        "start": match.start(),
                    }
                )
    return {
        "agent_visible_forbidden_matches": len(matches),
        "matches": matches,
        "task_count_scanned": scanned,
        "medicinal_chemistry_allowed": True,
    }


def render_safety_scope_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Safety Scope Report",
        "",
        f"task count scanned: {summary.get('task_count_scanned', 0)}",
        f"agent-visible forbidden term matches: {summary.get('agent_visible_forbidden_matches', 0)}",
        "medicinal chemistry allowed framing confirmation: yes",
        "",
        "Out-of-scope claim terms are checked only in agent-visible task text.",
    ]
    matches = summary.get("matches") if isinstance(summary.get("matches"), list) else []
    if matches:
        lines.extend(["", "Matches:"])
        for row in matches[:50]:
            lines.append(f"- {row.get('split')} {row.get('task_id')}: {row.get('term')}")
    return "\n".join(lines) + "\n"
