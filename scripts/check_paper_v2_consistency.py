from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "run_manifest.json",
    "environment.txt",
    "commands.log",
    "tables/full_offline_baseline_matrix_test.csv",
    "tables/main_table_representative_baselines.md",
    "tables/per_family_metrics_test.csv",
    "tables/wrapper_ablation_test.csv",
    "tables/protocol_ladder_test.csv",
    "tables/metric_winners_by_objective.md",
    "tables/bootstrap_ci_test_task_level.csv",
    "figures/per_family_action_accuracy_heatmap.pdf",
    "figures/wrapper_ablation_budget_curve.pdf",
    "figures/metric_rank_shift.pdf",
    "notes/wrapper_ablation_interpretation.md",
    "notes/protocol_ladder_interpretation.md",
    "notes/metric_sanity_interpretation.md",
    "notes/paper_insertion_memo.md",
    "validation/validate_dataset_strict.log",
    "validation/model_prompt_leakage_audit.log",
    "validation/oracle_scrambling_audit.log",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close_enough(a: str | float | None, b: str | float | None, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    for rel in REQUIRED_FILES:
        if not (args.results / rel).exists():
            errors.append(f"missing required file: {rel}")

    test_tasks = read_jsonl(args.release / "tasks" / "test.jsonl")
    if len(test_tasks) != 266:
        errors.append(f"test denominator is {len(test_tasks)}, expected 266")
    expected_counts: dict[str, int] = {}
    for task in test_tasks:
        expected_counts[str(task.get("expected_action"))] = expected_counts.get(str(task.get("expected_action")), 0) + 1

    normalized_path = args.results / "summaries" / "normalized_task_results.jsonl"
    if normalized_path.exists():
        normalized = read_jsonl(normalized_path)
        full_test = [
            row
            for row in normalized
            if row.get("source") == "full_offline_test" and row.get("adapter") == "well_engineered_wrapper"
        ]
        if len(full_test) != 266:
            errors.append(f"well_engineered_wrapper normalized test rows={len(full_test)}, expected 266")
    else:
        errors.append("missing normalized_task_results.jsonl")

    full_matrix = args.results / "tables" / "full_offline_baseline_matrix_test.csv"
    if full_matrix.exists() and normalized_path.exists():
        rows = read_csv(full_matrix)
        for row in rows:
            adapter = row["adapter"]
            subset = [
                item
                for item in read_jsonl(normalized_path)
                if item.get("source") == "full_offline_test" and item.get("adapter") == adapter
            ]
            if not subset:
                continue
            action_accuracy = sum(1 for item in subset if item.get("task_success")) / len(subset)
            if not close_enough(action_accuracy, row.get("action_accuracy")):
                errors.append(f"action_accuracy mismatch for {adapter}: table={row.get('action_accuracy')} raw={action_accuracy}")
            if adapter == "well_engineered_wrapper" and row.get("access_model") == "closed-book":
                errors.append("wrapper row is mixed into closed-book leaderboard")

    external_path = args.results / "tables" / "external_diagnostic_snapshot.csv"
    if external_path.exists():
        for row in read_csv(external_path):
            if row.get("adapter") != "skipped" and row.get("cache_mode") not in {"live", "replay", "skipped"}:
                errors.append(f"external diagnostic row lacks cache_mode label: {row}")

    figure_sources = args.results / "figure_sources.json"
    if figure_sources.exists():
        sources = json.loads(figure_sources.read_text(encoding="utf-8"))
        for stem, source in sources.items():
            if not (args.results / source).exists():
                errors.append(f"figure {stem} source CSV/table missing: {source}")
            if not (args.results / "figures" / f"{stem}.pdf").exists():
                errors.append(f"figure PDF missing: {stem}")
            if not (args.results / "figures" / f"{stem}.png").exists():
                errors.append(f"figure PNG missing: {stem}")
    else:
        errors.append("missing figure_sources.json")

    leakage_log = args.results / "validation" / "model_prompt_leakage_audit.log"
    if leakage_log.exists():
        text = leakage_log.read_text(encoding="utf-8", errors="ignore")
        if '"valid": true' not in text.lower():
            errors.append("model prompt leakage audit did not report valid=true")

    status = {"valid": not errors, "errors": errors, "expected_action_counts": expected_counts}
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
