from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from specguard_chem.dataset.validate_v1 import load_release_bundles_by_split, load_release_tasks_by_split
from specguard_chem.audits.hardening import (
    denominator_summary,
    render_claim_readiness,
    render_denominator_table,
    render_manual_test_bundle_dossiers,
    render_reviewer_attack_report,
)
from specguard_chem.audits.inventory import inventory_rows, inventory_summary, render_inventory_summary, write_inventory_csv
from specguard_chem.audits.leakage import leakage_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    tasks_by_split = load_release_tasks_by_split(args.release)
    bundles_by_split = load_release_bundles_by_split(args.release)
    audits_dir = args.release / "audits"
    rows = inventory_rows(tasks_by_split)
    write_inventory_csv(audits_dir / "task_inventory.csv", rows)
    summary = inventory_summary(tasks_by_split, bundles_by_split)
    (audits_dir / "task_inventory_summary.md").write_text(
        render_inventory_summary(summary), encoding="utf-8"
    )
    denominator = denominator_summary(tasks_by_split, bundles_by_split)
    (audits_dir / "claim_readiness_report.md").write_text(
        render_claim_readiness(denominator), encoding="utf-8"
    )
    (audits_dir / "manual_test_bundle_dossiers.md").write_text(
        render_manual_test_bundle_dossiers(tasks_by_split, bundles_by_split),
        encoding="utf-8",
    )
    paper_tables = Path("paper_v1") / "tables"
    paper_tables.mkdir(parents=True, exist_ok=True)
    (paper_tables / "evaluation_denominators.md").write_text(
        render_denominator_table(denominator), encoding="utf-8"
    )
    leakage = leakage_summary(tasks_by_split, bundles_by_split)
    prompt_report = _read_report_flags(audits_dir / "model_prompt_leakage_report.md")
    scrambling_report = _read_report_flags(audits_dir / "oracle_scrambling_report.md")
    preflight_report = _read_report_flags(audits_dir / "neurips_ed_preflight_report.md")
    reviewer_attack_report = render_reviewer_attack_report(
        leakage=leakage,
        prompt_leakage=prompt_report,
        scrambling=scrambling_report,
        denominator=denominator,
        preflight=preflight_report,
    )
    (audits_dir / "reviewer_attack_report.md").write_text(
        reviewer_attack_report, encoding="utf-8"
    )
    root_audits = Path("audits")
    root_audits.mkdir(parents=True, exist_ok=True)
    (root_audits / "reviewer_attack_report.md").write_text(
        reviewer_attack_report, encoding="utf-8"
    )
    _write_checksums(args.release)
    return 0


def _read_report_flags(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return {
        "valid": "valid: true" in text or '"valid": true' in text,
        "one_command_reproduction_configured": "one-command reproduction: configured" in text,
        "croissant_local_validation_passed": "croissant_local_validation_passed: true" in text,
        "anonymous_scan_passed": "anonymous_scan_passed: true" in text,
    }


def _write_checksums(release: Path) -> None:
    rows = {}
    for path in sorted(release.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release).as_posix()
        if rel == "checksums/sha256sums.txt":
            continue
        rows[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = [f"{digest}  {rel}" for rel, digest in sorted(rows.items())]
    (release / "checksums" / "sha256sums.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
