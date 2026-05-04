from __future__ import annotations

import argparse
from pathlib import Path

from specguard_chem.dataset.validate_v1 import load_release_bundles_by_split, load_release_tasks_by_split
from specguard_chem.audits.inventory import inventory_rows, inventory_summary, render_inventory_summary, write_inventory_csv


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
