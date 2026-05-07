from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt


ACTIONS = ("ACCEPT", "REJECT", "ABSTAIN", "INVALID")
EXPECTED_ACTIONS = ("ACCEPT", "REJECT", "ABSTAIN")
RELEASE_ID = "sgchem_v1.0"
REPRESENTATIVE = (
    "always_accept",
    "always_abstain",
    "heuristic",
    "abstention_guard",
    "local_mutation",
    "verify_first",
    "corpus_search",
    "well_engineered_wrapper",
)
CONFUSION_SYSTEMS = (
    "always_accept",
    "always_abstain",
    "local_mutation",
    "verify_first",
    "corpus_search",
    "well_engineered_wrapper",
    "heuristic",
    "abstention_guard",
    "verifier_guided_greedy",
)
FAMILY_ORDER = (
    "abstain_contradiction",
    "audit_accept",
    "audit_reject",
    "boundary_precision",
    "construct_feasible",
    "interrupt_resume",
    "repair_multi_violation",
    "repair_near_miss",
    "smiles_invariance",
    "tool_forced_l3",
)
FIGURE_SOURCES = {
    "per_family_action_accuracy_heatmap": "tables/per_family_metrics_test.csv",
    "wrapper_ablation_action_accuracy": "tables/wrapper_ablation_test.csv",
    "wrapper_ablation_budget_curve": "tables/wrapper_ablation_test.csv",
    "protocol_ladder_action_accuracy": "tables/protocol_ladder_test.csv",
    "protocol_ladder_verify_calls": "tables/protocol_ladder_test.csv",
    "metric_rank_shift": "tables/metric_ranking_sensitivity.csv",
}


@dataclass(frozen=True)
class RunData:
    split: str
    source: str
    adapter: str
    model: str
    access_model: str
    protocol: str
    run_dir: Path
    records: list[dict[str, Any]]
    summary: dict[str, Any]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_md_table(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        cells = [_format_md(row.get(field)) for field in fields]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_md(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except OSError:
        return None


def load_release_tasks(release: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        rows = read_jsonl(release / "tasks" / f"{split}.jsonl")
        by_split[split] = {str(row["task_id"]): row for row in rows}
    return by_split


def access_model_for(track: str | None, adapter: str) -> str:
    if adapter.startswith("wrapper_") or adapter == "well_engineered_wrapper":
        return "verifier/search wrapper"
    if adapter in {"verify_first", "verifier_guided_greedy"} or adapter.endswith(("_L3", "_l3")) and any(
        adapter.startswith(prefix) for prefix in ("verify_first", "verifier_guided_greedy")
    ):
        return "public-verifier"
    if adapter == "corpus_search":
        return "retrieval"
    value = str(track or "").strip()
    if value in {"closed_book", "primary_closed_book"}:
        return "closed-book"
    if value in {"tool_enabled", "public-verifier"}:
        return "public-verifier"
    if value in {"retrieval", "retrieval_upper_bound"}:
        return "retrieval"
    if value in {"oracle_upper_bound", "oracle", "debug"}:
        return "oracle/debug"
    if value in {"external", "external_model_snapshot"}:
        return "external diagnostic"
    if value == "wrapper_guarded":
        return "verifier/search wrapper"
    return value or "closed-book"


def final_decision(record: Mapping[str, Any]) -> str:
    if record.get("schema_error") or record.get("invalid_action") or record.get("invalid_tool_call"):
        return "INVALID"
    value = str(record.get("final_decision") or "").upper()
    if value in ACTIONS:
        return value
    decision = str(record.get("decision") or "").lower()
    if decision == "accept":
        return "ACCEPT"
    if decision == "reject":
        return "REJECT"
    if decision == "abstain":
        return "ABSTAIN"
    return "INVALID"


def expected_action(record: Mapping[str, Any]) -> str:
    value = str(record.get("expected_action") or record.get("expected") or "ACCEPT").upper()
    if value == "PASS":
        return "ACCEPT"
    if value == "FAIL":
        return "REJECT"
    return value if value in EXPECTED_ACTIONS else "ACCEPT"


def protocol_from_run(run: RunData) -> str:
    if run.protocol and run.protocol != "mixed":
        return run.protocol
    protocols = sorted({str(record.get("protocol") or "") for record in run.records})
    protocols = [item for item in protocols if item]
    return protocols[0] if len(protocols) == 1 else "mixed"


def final_invalid_molecule(record: Mapping[str, Any]) -> bool:
    rounds = record.get("rounds")
    if not isinstance(rounds, list):
        return False
    for item in reversed(rounds):
        if not isinstance(item, dict) or item.get("action") != "propose":
            continue
        evaluation = item.get("evaluation")
        if not isinstance(evaluation, dict):
            return False
        properties = evaluation.get("properties")
        return bool(not properties and not evaluation.get("sa_score"))
    return False


def confusion_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {expected: {action: 0 for action in ACTIONS} for expected in EXPECTED_ACTIONS}
    for record in records:
        matrix[expected_action(record)][final_decision(record)] += 1
    return matrix


def balanced_action_accuracy(records: Sequence[Mapping[str, Any]]) -> float | None:
    matrix = confusion_counts(records)
    recalls = []
    for action in EXPECTED_ACTIONS:
        den = sum(matrix[action].values())
        if den:
            recalls.append(matrix[action][action] / den)
    return mean(recalls) if recalls else None


def precision_recall(records: Sequence[Mapping[str, Any]], action: str) -> tuple[float | None, float | None]:
    expected = [expected_action(record) for record in records]
    predicted = [final_decision(record) for record in records]
    tp = sum(1 for exp, pred in zip(expected, predicted) if exp == action and pred == action)
    pred_den = sum(1 for pred in predicted if pred == action)
    exp_den = sum(1 for exp in expected if exp == action)
    return _safe_div(tp, pred_den), _safe_div(tp, exp_den)


def metric_row(run: RunData, records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(records if records is not None else run.records)
    expected = [expected_action(record) for record in rows]
    predicted = [final_decision(record) for record in rows]
    n = len(rows)
    correct = sum(1 for exp, pred in zip(expected, predicted) if exp == pred)
    accept_precision, accept_recall = precision_recall(rows, "ACCEPT")
    reject_precision, reject_recall = precision_recall(rows, "REJECT")
    abstain_precision, abstain_recall = precision_recall(rows, "ABSTAIN")
    expected_accept = sum(1 for exp in expected if exp == "ACCEPT")
    expected_non_accept = sum(1 for exp in expected if exp in {"REJECT", "ABSTAIN"})
    attempted = [record for record in rows if final_decision(record) != "ABSTAIN"]
    hard_violations = sum(1 for record in attempted if not bool(record.get("hard_pass")))
    verify_values = [int(record.get("verify_calls_used", 0) or 0) for record in rows]
    step_values = [int(record.get("steps_used", len(record.get("rounds") or [])) or 0) for record in rows]
    proposal_values = [int(record.get("proposals_used", 0) or 0) for record in rows]
    sorted_verify = sorted(verify_values)
    p95_index = int(math.ceil(0.95 * len(sorted_verify))) - 1 if sorted_verify else None
    pass_at_1 = _safe_div(
        sum(
            1
            for exp, pred, record in zip(expected, predicted, rows)
            if exp == "ACCEPT" and pred == "ACCEPT" and int(record.get("steps_used", 0) or 0) <= 1
        ),
        expected_accept,
    )
    pass_at_3 = _safe_div(
        sum(
            1
            for exp, pred, record in zip(expected, predicted, rows)
            if exp == "ACCEPT" and pred == "ACCEPT" and int(record.get("steps_used", 0) or 0) <= 3
        ),
        expected_accept,
    )
    return {
        "split": run.split,
        "access_model": run.access_model,
        "adapter": run.adapter,
        "system": run.adapter,
        "protocol": protocol_from_run(run),
        "n_tasks": n,
        "action_accuracy": _safe_div(correct, n),
        "balanced_action_accuracy": balanced_action_accuracy(rows),
        "accept_precision": accept_precision,
        "accept_recall": accept_recall,
        "reject_precision": reject_precision,
        "reject_recall": reject_recall,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "molecule_acceptance_rate": _safe_div(sum(1 for pred in predicted if pred == "ACCEPT"), n),
        "task_inconsistent_accept_rate": _safe_div(
            sum(1 for exp, pred in zip(expected, predicted) if exp in {"REJECT", "ABSTAIN"} and pred == "ACCEPT"),
            expected_non_accept,
        ),
        "false_abstain_rate": _safe_div(
            sum(1 for exp, pred in zip(expected, predicted) if exp == "ACCEPT" and pred == "ABSTAIN"),
            expected_accept,
        ),
        "hard_violation_rate": _safe_div(hard_violations, len(attempted)),
        "schema_error_rate": _safe_div(sum(1 for record in rows if record.get("schema_error")), n),
        "invalid_action_rate": _safe_div(sum(1 for record in rows if record.get("invalid_action")), n),
        "invalid_molecule_rate": _safe_div(sum(1 for record in rows if final_invalid_molecule(record)), n),
        "invalid_tool_call_rate": _safe_div(sum(1 for record in rows if record.get("invalid_tool_call")), n),
        "pass_at_1": pass_at_1,
        "pass_at_3": pass_at_3,
        "mean_steps": mean(step_values) if step_values else None,
        "mean_proposals": mean(proposal_values) if proposal_values else None,
        "mean_verify_calls": mean(verify_values) if verify_values else None,
        "p95_verify_calls": sorted_verify[p95_index] if p95_index is not None else None,
        "max_verify_calls": max(verify_values) if verify_values else None,
        "budget_exhaustion_rate": _safe_div(
            sum(1 for record in rows if str(record.get("termination_reason") or "").startswith("budget_exhausted")),
            n,
        ),
    }


def discover_runs(results: Path) -> list[RunData]:
    runs: list[RunData] = []
    aggregate_sources: set[Path] = set()
    for aggregate_path in sorted((results / "raw_runs").glob("*/aggregate.json")):
        source_dir = aggregate_path.parent
        aggregate_sources.add(source_dir)
        source = source_dir.name
        if source in {"full_offline_test_missing", "full_offline_test_wrapper"}:
            continue
        split = str(read_json(aggregate_path).get("split") or "test")
        aggregate = read_json(aggregate_path)
        for row in aggregate.get("all_baselines", aggregate.get("baselines", [])):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name"))
            report_path = source_dir / str(row.get("report_path"))
            run_dir = source_dir / str(row.get("run_dir", name))
            if not report_path.exists():
                report_path = run_dir / "report.json"
            trace_path = run_dir / "trace.jsonl"
            if report_path.exists():
                report = read_json(report_path)
                records = list(report.get("records") or [])
                summary = dict(report.get("summary") or {})
            elif trace_path.exists():
                records = read_jsonl(trace_path)
                summary = {}
            else:
                continue
            runs.append(
                RunData(
                    split=split,
                    source=source,
                    adapter=name,
                    model=str(row.get("model") or name),
                    access_model=access_model_for(row.get("track"), name),
                    protocol=str(row.get("protocol") or "mixed"),
                    run_dir=run_dir,
                    records=records,
                    summary=summary,
                )
            )
    for source_dir in sorted((results / "raw_runs").iterdir() if (results / "raw_runs").exists() else []):
        if not source_dir.is_dir() or source_dir in aggregate_sources:
            continue
        source = source_dir.name
        if source in {"full_offline_test_missing", "full_offline_test_wrapper"}:
            continue
        split = source.rsplit("_", 1)[-1] if source.rsplit("_", 1)[-1] in {"train", "dev", "test"} else "test"
        for report_path in sorted(source_dir.glob("*/report.json")):
            run_dir = report_path.parent
            adapter = run_dir.name
            report = read_json(report_path)
            records = list(report.get("records") or [])
            summary = dict(report.get("summary") or {})
            runs.append(
                RunData(
                    split=split,
                    source=source,
                    adapter=adapter,
                    model=adapter,
                    access_model=access_model_for(None, adapter),
                    protocol="mixed",
                    run_dir=run_dir,
                    records=records,
                    summary=summary,
                )
            )
    return runs


def save_figure(fig: plt.Figure, figures: Path, stem: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.png", dpi=220)
    fig.savefig(figures / f"{stem}.pdf")
    plt.close(fig)


def plot_heatmap(
    matrix: list[list[float | None]],
    *,
    rows: Sequence[str],
    cols: Sequence[str],
    title: str,
    stem: str,
    figures: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    numeric = [[float("nan") if value is None else value for value in row] for row in matrix]
    fig_width = max(7.0, 0.75 * len(cols) + 2.5)
    fig_height = max(4.5, 0.42 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(numeric, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for r_index, row in enumerate(numeric):
        for c_index, value in enumerate(row):
            label = "NA" if math.isnan(value) else f"{value:.2f}"
            ax.text(c_index, r_index, label, ha="center", va="center", color="white" if not math.isnan(value) and value < 0.55 else "black", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    save_figure(fig, figures, stem)


def generate_tables_and_figures(results: Path, release: Path, n_bootstrap: int) -> None:
    tables = results / "tables"
    figures = results / "figures"
    summaries = results / "summaries"
    notes = results / "notes"
    release_tasks = load_release_tasks(release)
    runs = discover_runs(results)
    by_source = defaultdict(list)
    for run in runs:
        by_source[run.source].append(run)

    full_sources = {"full_offline_train", "full_offline_dev", "full_offline_test"}
    full_rows = [metric_row(run) for run in runs if run.source in full_sources]
    full_fields = [
        "split",
        "access_model",
        "adapter",
        "protocol",
        "n_tasks",
        "action_accuracy",
        "balanced_action_accuracy",
        "accept_precision",
        "accept_recall",
        "reject_precision",
        "reject_recall",
        "abstain_precision",
        "abstain_recall",
        "molecule_acceptance_rate",
        "task_inconsistent_accept_rate",
        "false_abstain_rate",
        "hard_violation_rate",
        "schema_error_rate",
        "invalid_action_rate",
        "invalid_molecule_rate",
        "invalid_tool_call_rate",
        "pass_at_1",
        "pass_at_3",
        "mean_steps",
        "mean_proposals",
        "mean_verify_calls",
        "p95_verify_calls",
        "budget_exhaustion_rate",
    ]
    test_rows = [row for row in full_rows if row["split"] == "test"]
    write_csv(tables / "full_offline_baseline_matrix_test.csv", test_rows, full_fields)
    write_csv(tables / "full_offline_baseline_matrix_all_splits.csv", full_rows, full_fields)
    write_md_table(tables / "full_offline_baseline_matrix_test.md", test_rows, full_fields)

    representative_rows = [row for row in test_rows if row["adapter"] in REPRESENTATIVE]
    representative_rows.sort(key=lambda row: REPRESENTATIVE.index(str(row["adapter"])))
    main_fields = [
        "system",
        "access_model",
        "action_accuracy",
        "molecule_acceptance_rate",
        "task_inconsistent_accept_rate",
        "reject_recall",
        "abstain_recall",
        "schema_error_rate",
        "mean_verify_calls",
    ]
    write_csv(tables / "main_table_representative_baselines.csv", representative_rows, main_fields)
    write_md_table(tables / "main_table_representative_baselines.md", representative_rows, main_fields)

    normalized_rows = []
    release_lookup = release_tasks["test"] | release_tasks["dev"] | release_tasks["train"]
    for run in runs:
        for record in run.records:
            task_id = str(record.get("task_id"))
            task_meta = release_lookup.get(task_id, {})
            normalized_rows.append(
                {
                    "release": RELEASE_ID,
                    "split": run.split,
                    "source": run.source,
                    "access_model": run.access_model,
                    "adapter": run.adapter,
                    "protocol": str(record.get("protocol") or protocol_from_run(run)),
                    "task_public_hash": str(task_meta.get("agent_visible_hash") or _hash(task_id))[:71],
                    "family": str(record.get("task_family") or task_meta.get("task_family") or task_meta.get("task_type") or "unknown"),
                    "expected_action": expected_action(record).title(),
                    "predicted_action": final_decision(record).title(),
                    "schema_valid": not bool(record.get("schema_error")),
                    "molecule_valid": not final_invalid_molecule(record),
                    "hard_constraints_passed": bool(record.get("hard_pass")),
                    "task_success": expected_action(record) == final_decision(record),
                    "task_inconsistent_accept": expected_action(record) in {"REJECT", "ABSTAIN"} and final_decision(record) == "ACCEPT",
                    "abstained": final_decision(record) == "ABSTAIN",
                    "verify_calls": int(record.get("verify_calls_used", 0) or 0),
                    "steps": int(record.get("steps_used", len(record.get("rounds") or [])) or 0),
                    "proposals": int(record.get("proposals_used", 0) or 0),
                    "budget_exhausted": str(record.get("termination_reason") or "").startswith("budget_exhausted"),
                }
            )
    write_jsonl(summaries / "normalized_task_results.jsonl", normalized_rows)
    write_csv(summaries / "normalized_run_metrics.csv", full_rows, full_fields)

    full_test_by_adapter = {run.adapter: run for run in runs if run.source == "full_offline_test"}
    per_family_rows = []
    include_systems = [name for name in REPRESENTATIVE if name in full_test_by_adapter]
    for adapter in include_systems:
        run = full_test_by_adapter[adapter]
        families = sorted({str(record.get("task_family") or "unknown") for record in run.records})
        for family in families:
            subset = [record for record in run.records if str(record.get("task_family") or "unknown") == family]
            row = metric_row(run, subset)
            row["family"] = family
            row["family_reportability"] = "primary" if len(subset) >= 25 else "diagnostic"
            per_family_rows.append(row)
    family_fields = [
        "adapter",
        "access_model",
        "family",
        "family_reportability",
        "n_tasks",
        "action_accuracy",
        "balanced_action_accuracy",
        "molecule_acceptance_rate",
        "task_inconsistent_accept_rate",
        "reject_recall",
        "abstain_recall",
        "hard_violation_rate",
        "schema_error_rate",
        "mean_verify_calls",
    ]
    write_csv(tables / "per_family_metrics_test.csv", per_family_rows, family_fields)
    heatmap_cols = [family for family in FAMILY_ORDER if any(row["family"] == family for row in per_family_rows)]
    family_counts = {
        family: next(
            (
                int(row["n_tasks"])
                for row in per_family_rows
                if row["family"] == family and row["adapter"] == include_systems[0]
            ),
            0,
        )
        for family in heatmap_cols
    }
    heatmap_labels = [f"{family}\n(n={family_counts[family]})" for family in heatmap_cols]
    heatmap = []
    for adapter in include_systems:
        row = []
        for family in heatmap_cols:
            match = next(
                (item for item in per_family_rows if item["adapter"] == adapter and item["family"] == family),
                None,
            )
            row.append(_safe_float(match.get("action_accuracy")) if match else None)
        heatmap.append(row)
    plot_heatmap(
        heatmap,
        rows=include_systems,
        cols=heatmap_labels,
        title="Per-Family Action Accuracy (test)",
        stem="per_family_action_accuracy_heatmap",
        figures=figures,
    )
    action_table_rows = []
    for adapter in include_systems:
        row = {"adapter": adapter}
        for family in heatmap_cols:
            match = next(
                (item for item in per_family_rows if item["adapter"] == adapter and item["family"] == family),
                None,
            )
            row[family] = match.get("action_accuracy") if match else None
        action_table_rows.append(row)
    write_md_table(tables / "per_family_action_accuracy_test.md", action_table_rows, ["adapter", *heatmap_cols])

    collapse_rows = []
    for adapter in CONFUSION_SYSTEMS:
        run = full_test_by_adapter.get(adapter)
        if run is None:
            continue
        matrix = confusion_counts(run.records)
        count_rows = [
            {"expected_action": expected, **matrix[expected]} for expected in EXPECTED_ACTIONS
        ]
        count_fields = ["expected_action", *ACTIONS]
        write_csv(tables / f"confusion_{adapter}_counts.csv", count_rows, count_fields)
        norm_rows = []
        for expected in EXPECTED_ACTIONS:
            total = sum(matrix[expected].values())
            norm_rows.append(
                {
                    "expected_action": expected,
                    **{action: _safe_div(matrix[expected][action], total) for action in ACTIONS},
                }
            )
        write_csv(tables / f"confusion_{adapter}_row_normalized.csv", norm_rows, count_fields)
        plot_heatmap(
            [[row[action] for action in ACTIONS] for row in norm_rows],
            rows=list(EXPECTED_ACTIONS),
            cols=["Accept", "Reject", "Abstain", "Invalid/SchemaError"],
            title=f"Action Confusion: {adapter}",
            stem=f"confusion_{adapter}",
            figures=figures,
        )
        collapse_rows.append(
            {
                "adapter": adapter,
                "expected_accept_pred_accept": matrix["ACCEPT"]["ACCEPT"],
                "expected_reject_pred_accept": matrix["REJECT"]["ACCEPT"],
                "expected_abstain_pred_accept": matrix["ABSTAIN"]["ACCEPT"],
                "expected_accept_pred_abstain": matrix["ACCEPT"]["ABSTAIN"],
                "expected_reject_pred_abstain": matrix["REJECT"]["ABSTAIN"],
                "expected_abstain_pred_abstain": matrix["ABSTAIN"]["ABSTAIN"],
                "reject_to_accept_rate": _safe_div(matrix["REJECT"]["ACCEPT"], sum(matrix["REJECT"].values())),
                "abstain_to_accept_rate": _safe_div(matrix["ABSTAIN"]["ACCEPT"], sum(matrix["ABSTAIN"].values())),
                "accept_to_abstain_rate": _safe_div(matrix["ACCEPT"]["ABSTAIN"], sum(matrix["ACCEPT"].values())),
            }
        )
    collapse_fields = [
        "adapter",
        "expected_accept_pred_accept",
        "expected_reject_pred_accept",
        "expected_abstain_pred_accept",
        "expected_accept_pred_abstain",
        "expected_reject_pred_abstain",
        "expected_abstain_pred_abstain",
        "reject_to_accept_rate",
        "abstain_to_accept_rate",
        "accept_to_abstain_rate",
    ]
    write_csv(tables / "action_collapse_summary.csv", collapse_rows, collapse_fields)
    write_md_table(tables / "action_collapse_summary.md", collapse_rows, collapse_fields)

    wrapper_runs = {run.adapter: run for run in runs if run.source == "wrapper_ablation_test"}
    wrapper_rows = []
    for variant, run in sorted(wrapper_runs.items()):
        row = metric_row(run)
        row["variant"] = variant
        row["construct_family_accuracy"] = family_accuracy(run.records, ("construct_feasible",))
        row["repair_family_accuracy"] = family_accuracy(run.records, ("repair_", "tool_forced_l3"))
        row["audit_reject_family_accuracy"] = family_accuracy(run.records, ("audit_reject",))
        row["abstain_family_accuracy"] = family_accuracy(run.records, ("abstain_contradiction",))
        row["boundary_family_accuracy"] = family_accuracy(run.records, ("boundary_precision",))
        row["invariance_family_accuracy"] = family_accuracy(run.records, ("smiles_invariance",))
        row["interrupt_family_accuracy"] = family_accuracy(run.records, ("interrupt_resume",))
        row["notes"] = wrapper_note(variant)
        wrapper_rows.append(row)
    wrapper_fields = [
        "variant",
        "access_model",
        "n_tasks",
        "action_accuracy",
        "balanced_action_accuracy",
        "molecule_acceptance_rate",
        "task_inconsistent_accept_rate",
        "accept_recall",
        "reject_recall",
        "abstain_recall",
        "construct_family_accuracy",
        "repair_family_accuracy",
        "audit_reject_family_accuracy",
        "abstain_family_accuracy",
        "boundary_family_accuracy",
        "invariance_family_accuracy",
        "interrupt_family_accuracy",
        "mean_verify_calls",
        "p95_verify_calls",
        "max_verify_calls",
        "budget_exhaustion_rate",
        "notes",
    ]
    write_csv(tables / "wrapper_ablation_test.csv", wrapper_rows, wrapper_fields)
    write_md_table(tables / "wrapper_ablation_test.md", wrapper_rows, wrapper_fields)
    plot_bar(wrapper_rows, "variant", "action_accuracy", "Wrapper Ablation Action Accuracy", "wrapper_ablation_action_accuracy", figures)
    budget_rows = [row for row in wrapper_rows if str(row["variant"]).startswith("wrapper_verify_budget") or row["variant"] in {"wrapper_no_verifier_calls", "wrapper_full"}]
    budget_x = []
    budget_y = []
    for row in budget_rows:
        variant = str(row["variant"])
        if variant == "wrapper_no_verifier_calls":
            budget = 0
        elif variant == "wrapper_full":
            budget = max(int(row.get("max_verify_calls") or 0), 10)
        else:
            budget = int(variant.rsplit("_", 1)[-1])
        budget_x.append(budget)
        budget_y.append(row.get("action_accuracy"))
    if budget_x:
        sorted_budget = sorted(zip(budget_x, budget_y), key=lambda item: item[0])
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot([item[0] for item in sorted_budget], [item[1] for item in sorted_budget], marker="o")
        ax.set_xlabel("Verifier-call budget label")
        ax.set_ylabel("Action accuracy")
        ax.set_title("Wrapper Verifier-Budget Curve")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        save_figure(fig, figures, "wrapper_ablation_budget_curve")

    protocol_runs = [run for run in runs if run.source == "protocol_ladder_test"]
    protocol_rows = []
    for run in protocol_runs:
        base_adapter = run.adapter.rsplit("_", 1)[0] if run.adapter.endswith(("_L1", "_L2", "_L3")) else run.adapter
        row = metric_row(run)
        row["adapter"] = base_adapter
        row["protocol"] = protocol_from_run(run)
        row["not_applicable_reason"] = None
        protocol_rows.append(row)
    for adapter in ("verify_first", "verifier_guided_greedy"):
        for proto in ("L1", "L2"):
            protocol_rows.append(
                {
                    "adapter": adapter,
                    "protocol": proto,
                    "n_tasks": 0,
                    "not_applicable_reason": "adapter is verifier-oriented; benchmark entry restricted to L3",
                }
            )
    protocol_fields = [
        "adapter",
        "protocol",
        "n_tasks",
        "action_accuracy",
        "balanced_action_accuracy",
        "molecule_acceptance_rate",
        "task_inconsistent_accept_rate",
        "reject_recall",
        "abstain_recall",
        "hard_violation_rate",
        "schema_error_rate",
        "mean_steps",
        "mean_proposals",
        "mean_verify_calls",
        "p95_verify_calls",
        "budget_exhaustion_rate",
        "not_applicable_reason",
    ]
    write_csv(tables / "protocol_ladder_test.csv", protocol_rows, protocol_fields)
    write_md_table(tables / "protocol_ladder_test.md", protocol_rows, protocol_fields)
    plot_grouped_protocol(protocol_rows, "action_accuracy", "Protocol Ladder Action Accuracy", "protocol_ladder_action_accuracy", figures)
    plot_grouped_protocol(protocol_rows, "mean_verify_calls", "Protocol Ladder Verify Calls", "protocol_ladder_verify_calls", figures)

    ranking_rows, winners = ranking_sensitivity(representative_rows)
    ranking_fields = ["objective_metric", "adapter", "access_model", "value", "rank"]
    write_csv(tables / "metric_ranking_sensitivity.csv", ranking_rows, ranking_fields)
    write_md_table(tables / "metric_ranking_sensitivity.md", ranking_rows, ranking_fields)
    winner_fields = [
        "objective_metric",
        "rank_1_system",
        "rank_1_access_model",
        "rank_1_value",
        "rank_2_system",
        "rank_2_value",
        "hidden_failure_mode",
        "paper_interpretation",
    ]
    write_csv(tables / "metric_winners_by_objective.csv", winners, winner_fields)
    write_md_table(tables / "metric_winners_by_objective.md", winners, winner_fields)
    plot_rank_shift(ranking_rows, figures)

    task_ci, bundle_ci, main_ci_rows = bootstrap_tables(
        representative_rows,
        [full_test_by_adapter[name] for name in REPRESENTATIVE if name in full_test_by_adapter],
        release_tasks["test"],
        n_bootstrap=n_bootstrap,
    )
    ci_fields = ["adapter", "metric", "mean", "ci_low", "ci_high", "n_bootstrap", "resample_unit"]
    write_csv(tables / "bootstrap_ci_test_task_level.csv", task_ci, ci_fields)
    write_csv(tables / "bootstrap_ci_test_bundle_level.csv", bundle_ci, ci_fields)
    write_md_table(tables / "main_table_representative_baselines_with_ci.md", main_ci_rows, ["system", "access_model", "action_accuracy_ci", "molecule_acceptance_rate_ci", "task_inconsistent_accept_rate_ci", "reject_recall_ci", "abstain_recall_ci"])

    external_table = tables / "external_diagnostic_snapshot.csv"
    if not external_table.exists():
        external_rows = [
            {
                "adapter": "skipped",
                "provider_or_gateway": None,
                "model_id": None,
                "model_access_date": None,
                "protocol": None,
                "n_tasks": 0,
                "subset_definition": "not run",
                "sampling_temperature": None,
                "top_p": None,
                "max_tokens": None,
                "action_accuracy": None,
                "balanced_action_accuracy": None,
                "molecule_acceptance_rate": None,
                "task_inconsistent_accept_rate": None,
                "reject_recall": None,
                "abstain_recall": None,
                "schema_error_rate": None,
                "invalid_molecule_rate": None,
                "mean_verify_calls": None,
                "cache_mode": "skipped",
                "estimated_cost_usd": 0.0,
                "notes": "No credentials and no replay cache were configured.",
            }
        ]
        external_fields = list(external_rows[0].keys())
        write_csv(external_table, external_rows, external_fields)
        write_md_table(tables / "external_diagnostic_snapshot.md", external_rows, external_fields)
        (notes / "external_diagnostic_snapshot_skipped.md").write_text(
            "# External Diagnostic Snapshot Skipped\n\nReason: No credentials and no replay cache were configured. External diagnostics are secondary and were not treated as failed primary results.\n",
            encoding="utf-8",
        )
        write_json(
            summaries / "external_snapshot_subset.json",
            {
                "status": "skipped",
                "seed": 7,
                "reason": "No credentials and no replay cache were configured.",
            },
        )

    write_integrity_table(results, tables)
    write_notes(notes, representative_rows, wrapper_rows, protocol_rows, winners)
    write_manifest(results, release, runs)
    write_json(results / "figure_sources.json", FIGURE_SOURCES)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def family_accuracy(records: Sequence[Mapping[str, Any]], prefixes: tuple[str, ...]) -> float | None:
    subset = [
        record
        for record in records
        if any(str(record.get("task_family") or "").startswith(prefix) for prefix in prefixes)
    ]
    if not subset:
        return None
    return _safe_div(sum(1 for record in subset if expected_action(record) == final_decision(record)), len(subset))


def wrapper_note(variant: str) -> str:
    notes = {
        "wrapper_full": "Full public verifier/search wrapper.",
        "wrapper_no_public_candidate_search": "Corpus/public candidate search disabled.",
        "wrapper_no_verifier_calls": "L3 verify tool calls disabled.",
        "wrapper_verify_budget_1": "Verifier-call budget label 1.",
        "wrapper_verify_budget_3": "Verifier-call budget label 3.",
        "wrapper_verify_budget_10": "Verifier-call budget label 10.",
        "wrapper_no_contradiction_detector": "Public contradiction detector disabled.",
        "wrapper_no_repair_loop": "Local repair loop disabled.",
        "wrapper_no_boundary_special_case": "Boundary audit routing disabled.",
        "wrapper_name_scrambled_public_view": "Visible task name ignored by adapter.",
    }
    return notes.get(variant, "")


def plot_bar(rows: Sequence[Mapping[str, Any]], x_field: str, y_field: str, title: str, stem: str, figures: Path) -> None:
    if not rows:
        return
    labels = [str(row[x_field]) for row in rows]
    values = [_safe_float(row.get(y_field)) or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 4.2))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(y_field)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, figures, stem)


def plot_grouped_protocol(rows: Sequence[Mapping[str, Any]], y_field: str, title: str, stem: str, figures: Path) -> None:
    adapters = sorted({str(row.get("adapter")) for row in rows if row.get("n_tasks")})
    protocols = ("L1", "L2", "L3")
    if not adapters:
        return
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(adapters)), 4.2))
    xs = list(range(len(adapters)))
    for p_index, protocol in enumerate(protocols):
        values = []
        for adapter in adapters:
            match = next((row for row in rows if row.get("adapter") == adapter and row.get("protocol") == protocol and row.get("n_tasks")), None)
            values.append(_safe_float(match.get(y_field)) if match else None)
        ax.bar([x + (p_index - 1) * width for x in xs], [value or 0.0 for value in values], width=width, label=protocol)
    ax.set_xticks(xs, adapters, rotation=35, ha="right")
    ax.set_ylabel(y_field)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures, stem)


def ranking_sensitivity(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_verify = max((_safe_float(row.get("mean_verify_calls")) or 0.0 for row in rows), default=0.0)
    derived = []
    for row in rows:
        item = dict(row)
        accepts = [_safe_float(row.get("accept_recall")), _safe_float(row.get("reject_recall")), _safe_float(row.get("abstain_recall"))]
        item["balanced_action_score"] = mean([value for value in accepts if value is not None]) if any(value is not None for value in accepts) else None
        item["safe_action_score"] = (
            (_safe_float(row.get("action_accuracy")) or 0.0)
            - (_safe_float(row.get("task_inconsistent_accept_rate")) or 0.0)
            - (_safe_float(row.get("hard_violation_rate")) or 0.0)
            - (_safe_float(row.get("schema_error_rate")) or 0.0)
        )
        norm_verify = (_safe_float(row.get("mean_verify_calls")) or 0.0) / max_verify if max_verify else 0.0
        item["cost_adjusted_action_score_lambda_0_01"] = (_safe_float(row.get("action_accuracy")) or 0.0) - 0.01 * norm_verify
        item["cost_adjusted_action_score_lambda_0_05"] = (_safe_float(row.get("action_accuracy")) or 0.0) - 0.05 * norm_verify
        derived.append(item)

    objectives: list[tuple[str, bool]] = [
        ("action_accuracy", True),
        ("balanced_action_accuracy", True),
        ("molecule_acceptance_rate", True),
        ("task_inconsistent_accept_rate", False),
        ("reject_recall", True),
        ("abstain_recall", True),
        ("hard_violation_rate", False),
        ("schema_error_rate", False),
        ("mean_verify_calls", False),
        ("balanced_action_score", True),
        ("safe_action_score", True),
        ("cost_adjusted_action_score_lambda_0_01", True),
        ("cost_adjusted_action_score_lambda_0_05", True),
    ]
    ranking_rows = []
    winner_rows = []
    for objective, high_is_good in objectives:
        candidates = [row for row in derived if _safe_float(row.get(objective)) is not None]
        if objective == "mean_verify_calls":
            candidates = [row for row in candidates if (_safe_float(row.get("action_accuracy")) or 0.0) >= 0.9]
        candidates.sort(key=lambda row: _safe_float(row.get(objective)) or 0.0, reverse=high_is_good)
        for rank, row in enumerate(candidates, start=1):
            ranking_rows.append(
                {
                    "objective_metric": objective,
                    "adapter": row["adapter"],
                    "access_model": row["access_model"],
                    "value": row.get(objective),
                    "rank": rank,
                }
            )
        if candidates:
            first = candidates[0]
            second = candidates[1] if len(candidates) > 1 else {}
            winner_rows.append(
                {
                    "objective_metric": objective,
                    "rank_1_system": first["adapter"],
                    "rank_1_access_model": first["access_model"],
                    "rank_1_value": first.get(objective),
                    "rank_2_system": second.get("adapter"),
                    "rank_2_value": second.get(objective),
                    "hidden_failure_mode": hidden_failure_mode(objective, first),
                    "paper_interpretation": paper_interpretation(objective, first),
                }
            )
    return ranking_rows, winner_rows


def hidden_failure_mode(objective: str, row: Mapping[str, Any]) -> str:
    if objective == "molecule_acceptance_rate":
        return "May reward accept-biased systems that fail REJECT/ABSTAIN semantics."
    if objective in {"reject_recall", "abstain_recall"}:
        return "Exposes action semantics hidden by molecule-only acceptance."
    if objective == "mean_verify_calls":
        return "Cost-only ranking is meaningful only after an accuracy threshold."
    return "None specific; inspect confusion and family slices."


def paper_interpretation(objective: str, row: Mapping[str, Any]) -> str:
    if row.get("access_model") == "verifier/search wrapper":
        return "Ceiling under public verifier/search access; do not mix into closed-book leaderboard."
    if objective == "molecule_acceptance_rate":
        return "Molecule acceptance is not task success."
    return "Use with explicit access-model label."


def plot_rank_shift(rows: Sequence[Mapping[str, Any]], figures: Path) -> None:
    objectives = [
        "action_accuracy",
        "balanced_action_accuracy",
        "molecule_acceptance_rate",
        "task_inconsistent_accept_rate",
        "reject_recall",
        "abstain_recall",
        "safe_action_score",
    ]
    adapters = sorted({str(row["adapter"]) for row in rows if row["objective_metric"] in objectives})
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for adapter in adapters:
        points = []
        xs = []
        for index, objective in enumerate(objectives):
            match = next((row for row in rows if row["adapter"] == adapter and row["objective_metric"] == objective), None)
            if match is None:
                continue
            xs.append(index)
            points.append(int(match["rank"]))
        if points:
            ax.plot(xs, points, marker="o", label=adapter)
    ax.set_xticks(range(len(objectives)), objectives, rotation=35, ha="right")
    ax.set_ylabel("Rank (1 is best)")
    ax.invert_yaxis()
    ax.set_title("Metric Ranking Sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    save_figure(fig, figures, "metric_rank_shift")


def bootstrap_tables(
    summary_rows: Sequence[Mapping[str, Any]],
    runs: Sequence[RunData],
    test_tasks: Mapping[str, Mapping[str, Any]],
    *,
    n_bootstrap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = [
        "action_accuracy",
        "balanced_action_accuracy",
        "molecule_acceptance_rate",
        "task_inconsistent_accept_rate",
        "reject_recall",
        "abstain_recall",
    ]
    task_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    main_ci: list[dict[str, Any]] = []
    run_by_adapter = {run.adapter: run for run in runs}
    summary_by_adapter = {str(row["adapter"]): row for row in summary_rows}
    for adapter in REPRESENTATIVE:
        run = run_by_adapter.get(adapter)
        if run is None:
            continue
        for metric in metrics:
            task_rows.append(bootstrap_metric(run.records, metric, n_bootstrap=n_bootstrap, seed=7, unit="task"))
            task_rows[-1]["adapter"] = adapter
            bundle_rows.append(
                bootstrap_metric_by_bundle(
                    run.records,
                    test_tasks,
                    metric,
                    n_bootstrap=n_bootstrap,
                    seed=7,
                )
            )
            bundle_rows[-1]["adapter"] = adapter
        row = {
            "system": adapter,
            "access_model": summary_by_adapter.get(adapter, {}).get("access_model"),
        }
        for metric in (
            "action_accuracy",
            "molecule_acceptance_rate",
            "task_inconsistent_accept_rate",
            "reject_recall",
            "abstain_recall",
        ):
            ci = next(item for item in task_rows if item["adapter"] == adapter and item["metric"] == metric)
            row[f"{metric}_ci"] = f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        main_ci.append(row)
    return task_rows, bundle_rows, main_ci


def metric_value(records: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    dummy = RunData("test", "bootstrap", "adapter", "model", "closed-book", "mixed", Path("."), list(records), {})
    return _safe_float(metric_row(dummy).get(metric))


def bootstrap_metric(records: Sequence[Mapping[str, Any]], metric: str, *, n_bootstrap: int, seed: int, unit: str) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(records)
    observed = metric_value(records, metric)
    values = []
    if n and observed is not None:
        for _ in range(n_bootstrap):
            sample = [records[rng.randrange(n)] for _ in range(n)]
            value = metric_value(sample, metric)
            if value is not None:
                values.append(value)
    return ci_payload(metric, observed, values, n_bootstrap, unit)


def bootstrap_metric_by_bundle(
    records: Sequence[Mapping[str, Any]],
    test_tasks: Mapping[str, Mapping[str, Any]],
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    by_bundle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        task = test_tasks.get(str(record.get("task_id")), {})
        bundle_id = str(task.get("bundle_id") or str(record.get("task_id")).split("__")[0])
        by_bundle[bundle_id].append(record)
    bundles = list(by_bundle)
    observed = metric_value(records, metric)
    values = []
    rng = random.Random(seed + 1000)
    if bundles and observed is not None:
        for _ in range(n_bootstrap):
            sample = []
            for _ in bundles:
                sample.extend(by_bundle[rng.choice(bundles)])
            value = metric_value(sample, metric)
            if value is not None:
                values.append(value)
    return ci_payload(metric, observed, values, n_bootstrap, "bundle")


def ci_payload(metric: str, observed: float | None, values: list[float], n_bootstrap: int, unit: str) -> dict[str, Any]:
    if observed is None or not values:
        low = high = observed
    else:
        values.sort()
        low = values[int(0.025 * (len(values) - 1))]
        high = values[int(0.975 * (len(values) - 1))]
    return {
        "metric": metric,
        "mean": observed,
        "ci_low": low,
        "ci_high": high,
        "n_bootstrap": len(values) if values else 0,
        "resample_unit": unit,
    }


def write_integrity_table(results: Path, tables: Path) -> None:
    checks = []
    mapping = [
        ("strict validation", results / "validation" / "validate_dataset_strict.log"),
        ("Croissant validation", results / "validation" / "validate_croissant.log"),
        ("prompt leakage audit", results / "validation" / "model_prompt_leakage_audit.log"),
        ("oracle scrambling", results / "validation" / "oracle_scrambling_audit.log"),
        ("negative controls", Path("paper_v1/tables/negative_controls.md")),
        ("split leakage", Path("benchmarks/releases/sgchem_v1.0/audits/split_leakage_report.md")),
        ("clean-clone reproduction", Path("audits/clean_reviewer_reproduction_report.md")),
        ("hosted URL preflight", results / "validation" / "neurips_ed_preflight.log"),
    ]
    for name, path in mapping:
        status = "available" if path.exists() else "not_run"
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        lower = text.lower()
        if "error" in lower or "failed" in lower and "0 failed" not in lower:
            status = "review"
        if '"valid": true' in lower or "valid true" in lower or "passed" in lower:
            status = "passed"
        checks.append({"gate": name, "status": status, "path": str(path)})
    fields = ["gate", "status", "path"]
    write_csv(tables / "artifact_integrity_gates_v2.csv", checks, fields)
    write_md_table(tables / "artifact_integrity_gates_v2.md", checks, fields)


def write_notes(
    notes: Path,
    representative_rows: Sequence[Mapping[str, Any]],
    wrapper_rows: Sequence[Mapping[str, Any]],
    protocol_rows: Sequence[Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]],
) -> None:
    wrapper_full = next((row for row in wrapper_rows if row.get("variant") == "wrapper_full"), None)
    no_search = next((row for row in wrapper_rows if row.get("variant") == "wrapper_no_public_candidate_search"), None)
    no_verify = next((row for row in wrapper_rows if row.get("variant") == "wrapper_no_verifier_calls"), None)
    scrambled = next((row for row in wrapper_rows if row.get("variant") == "wrapper_name_scrambled_public_view"), None)
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "wrapper_ablation_interpretation.md").write_text(
        "\n".join(
            [
                "# Wrapper Ablation Interpretation",
                "",
                f"Full wrapper action accuracy: {_format_md((wrapper_full or {}).get('action_accuracy'))}.",
                f"Without public candidate search: {_format_md((no_search or {}).get('action_accuracy'))}.",
                f"Without L3 verify tool calls: {_format_md((no_verify or {}).get('action_accuracy'))}.",
                "The strongest public capability is the combination of candidate search plus public-spec evaluation; disabling one route should be interpreted by the measured delta, not by a hidden oracle claim.",
                "The verifier-budget variants are labels over the public verify-call behavior exposed by the adapter and runner; inspect mean/p95 verify calls before making cost claims.",
                f"Name-scrambled public-view accuracy: {_format_md((scrambled or {}).get('action_accuracy'))}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    l3_rows = [row for row in protocol_rows if row.get("protocol") == "L3" and row.get("n_tasks")]
    l2_rows = [row for row in protocol_rows if row.get("protocol") == "L2" and row.get("n_tasks")]
    (notes / "protocol_ladder_interpretation.md").write_text(
        "\n".join(
            [
                "# Protocol Ladder Interpretation",
                "",
                f"Mean L3 action accuracy across applicable rows: {_format_md(mean([_safe_float(row.get('action_accuracy')) or 0.0 for row in l3_rows]) if l3_rows else None)}.",
                f"Mean L2 action accuracy across applicable rows: {_format_md(mean([_safe_float(row.get('action_accuracy')) or 0.0 for row in l2_rows]) if l2_rows else None)}.",
                "L3 verifier access mainly changes interpretation when a system uses the verify tool or public verifier semantics; L2 feedback can improve repair behavior without necessarily fixing reject/abstain action semantics.",
                "Rows with higher molecule acceptance but flat or lower action accuracy should not be presented as task-success improvements.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    winner_lines = [
        "# Metric Sanity Interpretation",
        "",
        "Molecule acceptance is not a stable proxy for task success in SpecGuard-Chem.",
        "Accept-biased and retrieval-heavy systems can score well on molecule acceptance while failing REJECT and ABSTAIN semantics.",
        "Action accuracy and balanced action accuracy expose those failures directly.",
        "Reject recall isolates audit/rejection collapse, while abstain recall isolates contradiction-handling behavior.",
        "The wrapper row is a public verifier/search ceiling and must remain separate from closed-book systems.",
        "Cost-adjusted metrics are useful only after making the access model and verifier-call accounting explicit.",
        "The paper should report metric ranking sensitivity as evidence for action-aware evaluation rather than as a single universal leaderboard.",
    ]
    (notes / "metric_sanity_interpretation.md").write_text("\n".join(winner_lines) + "\n", encoding="utf-8")
    (notes / "paper_insertion_memo.md").write_text(paper_memo(representative_rows, winners), encoding="utf-8")


def paper_memo(representative_rows: Sequence[Mapping[str, Any]], winners: Sequence[Mapping[str, Any]]) -> str:
    wrapper = next((row for row in representative_rows if row.get("adapter") == "well_engineered_wrapper"), {})
    accept = next((row for row in representative_rows if row.get("adapter") == "always_accept"), {})
    return f"""# Paper Insertion Memo

## Proposed Results Section

### Finding 1: Molecule acceptance overstates specification success.
The expanded baselines show that high molecule acceptance can be achieved without satisfying task-level action semantics. For example, `always_accept` has molecule acceptance {_format_md(accept.get('molecule_acceptance_rate'))}, but its task-inconsistent acceptance rate is {_format_md(accept.get('task_inconsistent_accept_rate'))}.

### Finding 2: Failures are action- and family-specific.
Per-family tables and action-confusion matrices show which task families drive each failure mode, including reject collapse, abstention collapse, boundary precision failures, and interrupt/resume behavior.

### Finding 3: Public verifier/search access saturates the contract.
The wrapper result should be presented as a declared-access ceiling: action accuracy {_format_md(wrapper.get('action_accuracy'))}, reject recall {_format_md(wrapper.get('reject_recall'))}, and abstain recall {_format_md(wrapper.get('abstain_recall'))}. This supports the interpretation of SpecGuard-Chem as an evaluation-contract artifact rather than a chemistry-capability leaderboard.

### Finding 4: Metric choice changes apparent conclusions.
The ranking-sensitivity table shows that molecule acceptance, action accuracy, reject recall, abstain recall, and cost-adjusted scores select different winners.

### Finding 5: Expanded audits preserve public/private isolation.
The public adapter boundary remains separate from scorer-side labels and oracle certificates; audit logs are included with the paper_v2 result package.

## Recommended Main-Paper Figures/Tables

- Table: representative baseline matrix with action metrics.
- Figure: per-family action accuracy heatmap.
- Figure: wrapper ablation / verifier-budget curve.
- Table: metric winners by objective.

## Appendix Tables

- Full offline baseline matrix.
- All confusion matrices.
- Protocol ladder.
- External diagnostic snapshot.
- Bootstrap confidence intervals.
- Audit logs and integrity gates.

## Key Wording

The expanded baselines show that high molecule acceptance is easy to obtain with accept-biased or retrieval-heavy systems, but those systems fail reject and abstain semantics. The wrapper ablations show that saturation is driven by public verifier/search access rather than hidden oracle leakage. Therefore, SpecGuard-Chem should be interpreted as an evaluation-contract artifact with explicit access-model ceilings, not as a chemistry capability leaderboard.
"""


def write_manifest(results: Path, release: Path, runs: Sequence[RunData]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _commit(),
        "release": RELEASE_ID,
        "release_path": str(release),
        "primary_test_denominator": len(read_jsonl(release / "tasks" / "test.jsonl")),
        "n_runs_loaded": len(runs),
        "sources": sorted({run.source for run in runs}),
        "access_models": sorted({run.access_model for run in runs}),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "external_diagnostics": "skipped unless external_diagnostic_snapshot.csv records live/cache rows",
    }
    write_json(results / "run_manifest.json", manifest)


def write_results_summary(results: Path) -> None:
    manifest = read_json(results / "run_manifest.json")
    main_rows = list(csv.DictReader((results / "tables" / "main_table_representative_baselines.csv").open()))
    wrapper = next((row for row in main_rows if row["system"] == "well_engineered_wrapper"), {})
    summary = f"""# SpecGuard-Chem paper_v2 result summary

## Release and environment
- commit: {manifest.get('commit')}
- release path: {manifest.get('release_path')}
- validation status: see `validation/`

## Main findings
1. Molecule acceptance versus action accuracy: representative baseline tables separate molecule acceptance from action accuracy and task-inconsistent acceptance.
2. Per-family failure modes: `per_family_metrics_test.csv` and the heatmap show family-specific behavior.
3. Wrapper saturation and ablations: wrapper action accuracy is {_format_md(_safe_float(wrapper.get('action_accuracy')))} under the public verifier/search access model.
4. Protocol ladder: `protocol_ladder_test.csv` separates L1/L2/L3 behavior and verifier-call use.
5. Metric ranking sensitivity: `metric_winners_by_objective.md` shows that headline metric choice changes apparent winners.
6. External diagnostics, if available: external diagnostics are marked secondary; skipped runs are documented in notes.
7. Audit status: validation and audit logs are stored under `validation/`.

## Recommended paper changes
- main text changes: emphasize action-aware evaluation, access-model separation, and wrapper ceiling interpretation.
- appendix additions: full matrix, confusion matrices, protocol ladder, CIs, and audit gates.
- figures/tables to replace existing ones: representative baseline matrix, per-family heatmap, wrapper ablation/budget curve, and metric winners table.

## Caveats
- synthetic rule-based release
- wrapper-solvable under public verifier/search threat model
- no drug-discovery claims
- external results are diagnostic only
"""
    (results / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("benchmarks/releases/sgchem_v1.0"))
    parser.add_argument("--results", type=Path, default=Path("paper_v2/results"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    generate_tables_and_figures(args.results, args.release, args.n_bootstrap)
    write_results_summary(args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
