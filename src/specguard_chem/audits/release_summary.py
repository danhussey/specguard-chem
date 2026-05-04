from __future__ import annotations

"""Write all compiled-release audit reports."""

from pathlib import Path
from typing import Any, Mapping

from .duplicates import duplicate_summary, render_duplicate_report
from .inventory import (
    inventory_rows,
    inventory_summary,
    render_inventory_summary,
    write_inventory_csv,
)
from .leakage import leakage_summary, render_split_leakage_report
from .oracle_validation import render_oracle_validation_report
from .scope_safety import render_safety_scope_report, scan_agent_visible_scope


def render_curation_summary(compilation: Mapping[str, Any]) -> str:
    lines = [
        "# Curation Summary",
        "",
        "curation mode: deterministic generated oracle-backed bundles",
        f"requested bundles: {compilation.get('requested_bundles', 0)}",
        f"generated bundles: {compilation.get('generated_bundles', 0)}",
        f"requested tasks: {compilation.get('requested_tasks', 0)}",
        f"generated tasks: {compilation.get('generated_tasks', 0)}",
        f"generation shortfall: {compilation.get('generation_shortfall', 0)}",
        f"generation shortfall reason: {compilation.get('generation_shortfall_reason')}",
    ]
    return "\n".join(lines) + "\n"


def write_audit_reports(
    *,
    release_dir: Path,
    benchmark_id: str,
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    bundles_by_split: Mapping[str, list[Mapping[str, Any]]],
    oracle_validation: Mapping[str, Any],
    compilation: Mapping[str, Any],
) -> dict[str, Any]:
    audits_dir = release_dir / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    rows = inventory_rows(tasks_by_split)
    inventory = inventory_summary(tasks_by_split, bundles_by_split)
    duplicates = duplicate_summary(tasks_by_split)
    leakage = leakage_summary(tasks_by_split, bundles_by_split)
    safety = scan_agent_visible_scope(tasks_by_split)

    write_inventory_csv(audits_dir / "task_inventory.csv", rows)
    (audits_dir / "task_inventory_summary.md").write_text(
        render_inventory_summary(inventory), encoding="utf-8"
    )
    duplicate_text = render_duplicate_report(duplicates)
    (audits_dir / "agent_visible_duplicate_report.md").write_text(
        duplicate_text, encoding="utf-8"
    )
    (audits_dir / f"{benchmark_id}_agent_visible_duplicate_report.md").write_text(
        duplicate_text, encoding="utf-8"
    )
    leakage_text = render_split_leakage_report(leakage)
    (audits_dir / "split_leakage_report.md").write_text(
        leakage_text, encoding="utf-8"
    )
    (audits_dir / f"{benchmark_id}_split_leakage_report.md").write_text(
        leakage_text, encoding="utf-8"
    )
    (audits_dir / "oracle_validation_report.md").write_text(
        render_oracle_validation_report(oracle_validation), encoding="utf-8"
    )
    (audits_dir / "safety_scope_report.md").write_text(
        render_safety_scope_report(safety), encoding="utf-8"
    )
    (audits_dir / "curation_summary.md").write_text(
        render_curation_summary(compilation), encoding="utf-8"
    )

    return {
        "inventory": inventory,
        "duplicates": duplicates,
        "leakage": leakage,
        "safety_scope": safety,
    }
