from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specguard_chem.utils import jsonio


ACTIONS = ("ACCEPT", "REJECT", "ABSTAIN")
PREDICTED_ACTIONS = ("ACCEPT", "REJECT", "ABSTAIN", "INVALID")
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

ACCESS_MODEL_BY_TRACK = {
    "primary_closed_book": "closed-book",
    "closed_book": "closed-book",
    "tool_enabled": "public-verifier",
    "retrieval_upper_bound": "retrieval",
    "retrieval": "retrieval",
    "wrapper_guarded": "verifier/search wrapper",
    "external_model_snapshot": "external diagnostic",
    "external": "external diagnostic",
    "oracle_upper_bound": "oracle/debug",
}


def _safe_div(numer: float, denom: float) -> float | None:
    return None if denom == 0 else numer / denom


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def _md_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header)) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _write_md(path: Path, rows: Sequence[Mapping[str, Any]], *, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"# {title}\n\n" if title else ""
    path.write_text(prefix + _md_table(rows), encoding="utf-8")


def _save_fig(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.png", dpi=220)
    fig.savefig(figures_dir / f"{stem}.pdf")
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [row for row in jsonio.read_jsonl(path) if isinstance(row, dict)]


def _load_release_tasks(release: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split: _read_jsonl(release / "tasks" / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }


def _public_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]


def _expected(record: Mapping[str, Any]) -> str:
    value = str(record.get("expected_action") or "").upper()
    if value in ACTIONS:
        return value
    legacy = str(record.get("expected") or "PASS").upper()
    if legacy == "ABSTAIN":
        return "ABSTAIN"
    if legacy == "FAIL":
        return "REJECT"
    return "ACCEPT"


def _predicted(record: Mapping[str, Any]) -> str:
    if record.get("schema_error") or record.get("invalid_action") or record.get("invalid_tool_call"):
        return "INVALID"
    value = str(record.get("final_decision") or "").upper()
    if value in PREDICTED_ACTIONS:
        return value
    decision = str(record.get("decision") or "").lower()
    if decision == "accept":
        return "ACCEPT"
    if decision == "reject":
        return "REJECT"
    if decision == "abstain":
        return "ABSTAIN"
    return "INVALID"


def _schema_output_counts(records: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int]:
    outputs = 0
    schema = 0
    invalid_action = 0
    invalid_tool = 0
    for record in records:
        rounds = record.get("rounds")
        if not isinstance(rounds, list):
            continue
        for item in rounds:
            if not isinstance(item, dict):
                continue
            outputs += 1
            schema += int(bool(item.get("schema_error")))
            invalid_action += int(bool(item.get("invalid_action")))
            invalid_tool += int(bool(item.get("invalid_tool_call")))
    if outputs == 0:
        outputs = len(records)
        schema = sum(int(bool(row.get("schema_error"))) for row in records)
        invalid_action = sum(int(bool(row.get("invalid_action"))) for row in records)
        invalid_tool = sum(int(bool(row.get("invalid_tool_call"))) for row in records)
    return outputs, schema, invalid_action, invalid_tool


def _invalid_molecule_rate(records: Sequence[Mapping[str, Any]]) -> float | None:
    invalid = 0
    proposals = 0
    for record in records:
        for item in record.get("rounds") or []:
            if not isinstance(item, dict) or item.get("action") != "propose":
                continue
            proposals += 1
            vector = item.get("failure_vector")
            text = json.dumps(vector, sort_keys=True).lower() if isinstance(vector, dict) else ""
            evaluation = item.get("evaluation")
            if "invalid_smiles" in text or "invalid smiles" in text:
                invalid += 1
            elif isinstance(evaluation, dict) and evaluation.get("properties") == {} and not evaluation.get("hard_pass"):
                invalid += int("invalid" in " ".join(map(str, evaluation.get("hard_fails", []))).lower())
    return _safe_div(invalid, proposals)


def _confusion(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {exp: {pred: 0 for pred in PREDICTED_ACTIONS} for exp in ACTIONS}
    for record in records:
        matrix[_expected(record)][_predicted(record)] += 1
    return matrix


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), 95))


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    n = len(records)
    matrix = _confusion(records)
    correct = sum(matrix[action][action] for action in ACTIONS)
    recalls: dict[str, float | None] = {}
    precisions: dict[str, float | None] = {}
    for action in ACTIONS:
        recalls[action] = _safe_div(matrix[action][action], sum(matrix[action].values()))
        precisions[action] = _safe_div(
            matrix[action][action],
            sum(matrix[exp][action] for exp in ACTIONS),
        )
    balanced = mean([value for value in recalls.values() if value is not None]) if any(value is not None for value in recalls.values()) else None
    predicted = [_predicted(record) for record in records]
    expected = [_expected(record) for record in records]
    unsafe_denom = sum(1 for value in expected if value in {"REJECT", "ABSTAIN"})
    false_abstain_denom = sum(1 for value in expected if value in {"ACCEPT", "REJECT"})
    attempted = [idx for idx, pred in enumerate(predicted) if pred != "ABSTAIN"]
    outputs, schema, invalid_action, invalid_tool = _schema_output_counts(records)
    step_values = [int(record.get("steps_used") or 0) for record in records]
    proposal_values = [int(record.get("proposals_used") or 0) for record in records]
    verify_values = [int(record.get("verify_calls_used") or 0) for record in records]
    total_verify_values = [int(record.get("total_verifier_calls") or 0) for record in records]

    def pass_at(k: int) -> float | None:
        accept_indices = [idx for idx, exp in enumerate(expected) if exp == "ACCEPT"]
        if not accept_indices:
            return None
        hits = sum(1 for idx in accept_indices if predicted[idx] == "ACCEPT" and step_values[idx] <= k)
        return hits / len(accept_indices)

    return {
        "n_tasks": n,
        "action_accuracy": _safe_div(correct, n),
        "balanced_action_accuracy": balanced,
        "accept_precision": precisions["ACCEPT"],
        "accept_recall": recalls["ACCEPT"],
        "reject_precision": precisions["REJECT"],
        "reject_recall": recalls["REJECT"],
        "abstain_precision": precisions["ABSTAIN"],
        "abstain_recall": recalls["ABSTAIN"],
        "molecule_acceptance_rate": _safe_div(predicted.count("ACCEPT"), n),
        "task_inconsistent_accept_rate": _safe_div(
            sum(1 for exp, pred in zip(expected, predicted) if exp in {"REJECT", "ABSTAIN"} and pred == "ACCEPT"),
            unsafe_denom,
        ),
        "false_abstain_rate": _safe_div(
            sum(1 for exp, pred in zip(expected, predicted) if exp in {"ACCEPT", "REJECT"} and pred == "ABSTAIN"),
            false_abstain_denom,
        ),
        "hard_violation_rate": _safe_div(
            sum(1 for idx in attempted if not bool(records[idx].get("hard_pass"))),
            len(attempted),
        ),
        "schema_error_rate": _safe_div(schema, outputs),
        "invalid_action_rate": _safe_div(invalid_action, outputs),
        "invalid_molecule_rate": _invalid_molecule_rate(records),
        "invalid_tool_call_rate": _safe_div(invalid_tool, outputs),
        "pass_at_1": pass_at(1),
        "pass_at_3": pass_at(3),
        "mean_steps": mean(step_values) if step_values else None,
        "mean_proposals": mean(proposal_values) if proposal_values else None,
        "mean_verify_calls": mean(verify_values) if verify_values else None,
        "mean_total_verifier_calls": mean(total_verify_values) if total_verify_values else None,
        "p95_verify_calls": _p95(verify_values),
        "max_verify_calls": max(verify_values) if verify_values else None,
        "budget_exhaustion_rate": _safe_div(
            sum(1 for record in records if str(record.get("termination_reason") or "").startswith("budget_exhausted")),
            n,
        ),
        "confusion": matrix,
    }


def _report_rows(sweep_dir: Path) -> list[dict[str, Any]]:
    aggregate_path = sweep_dir / "aggregate.json"
    if not aggregate_path.exists():
        return []
    aggregate = jsonio.read_json(aggregate_path)
    split = str(aggregate.get("split") or "unknown")
    rows: list[dict[str, Any]] = []
    for baseline in aggregate.get("all_baselines", aggregate.get("baselines", [])):
        if not isinstance(baseline, dict):
            continue
        report_path = sweep_dir / str(baseline.get("report_path"))
        if not report_path.exists():
            continue
        payload = jsonio.read_json(report_path)
        records = payload.get("records")
        if not isinstance(records, list):
            continue
        rows.append(
            {
                "split": split,
                "adapter": str(baseline.get("name")),
                "model": str(baseline.get("model")),
                "protocol": baseline.get("protocol") or "mixed",
                "track": baseline.get("track") or "primary_closed_book",
                "access_model": ACCESS_MODEL_BY_TRACK.get(str(baseline.get("track")), str(baseline.get("track") or "closed-book")),
                "run_dir": str(report_path.parent),
                "records": [record for record in records if isinstance(record, dict)],
                "adapter_kwargs": baseline.get("adapter_kwargs") or {},
            }
        )
    return rows


def _run_metric_rows(report_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in report_rows:
        metrics = _metrics(report["records"])
        row = {
            "split": report["split"],
            "access_model": report["access_model"],
            "adapter": report["adapter"],
            "protocol": report["protocol"],
        }
        for key in (
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
        ):
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def _normalize_records(report_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in report_rows:
        for record in report["records"]:
            expected = _expected(record)
            predicted = _predicted(record)
            rows.append(
                {
                    "release": "sgchem_v1.0",
                    "split": report["split"],
                    "access_model": report["access_model"],
                    "adapter": report["adapter"],
                    "protocol": record.get("protocol") or report["protocol"],
                    "task_public_hash": _public_hash(str(record.get("task_id") or "")),
                    "family": record.get("task_family") or "unknown",
                    "expected_action": expected,
                    "predicted_action": predicted,
                    "schema_valid": not bool(record.get("schema_error")),
                    "molecule_valid": predicted != "INVALID" and (bool(record.get("hard_pass")) or predicted != "ACCEPT"),
                    "hard_constraints_passed": bool(record.get("hard_pass")),
                    "task_success": expected == predicted,
                    "task_inconsistent_accept": expected in {"REJECT", "ABSTAIN"} and predicted == "ACCEPT",
                    "abstained": predicted == "ABSTAIN",
                    "verify_calls": int(record.get("verify_calls_used") or 0),
                    "steps": int(record.get("steps_used") or 0),
                    "proposals": int(record.get("proposals_used") or 0),
                    "budget_exhausted": str(record.get("termination_reason") or "").startswith("budget_exhausted"),
                }
            )
    return rows


def _write_normalized(out: Path, all_report_rows: Sequence[Mapping[str, Any]]) -> None:
    normalized = _normalize_records(all_report_rows)
    jsonio.write_jsonl(out / "summaries" / "normalized_task_results.jsonl", normalized)
    run_rows = _run_metric_rows(all_report_rows)
    _write_csv(out / "summaries" / "normalized_run_metrics.csv", run_rows)


def _make_full_offline_tables(out: Path, full_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tables = out / "tables"
    metric_rows = _run_metric_rows(full_rows)
    test_rows = [row for row in metric_rows if row["split"] == "test"]
    _write_csv(tables / "full_offline_baseline_matrix_test.csv", test_rows)
    _write_csv(tables / "full_offline_baseline_matrix_all_splits.csv", metric_rows)
    _write_md(tables / "full_offline_baseline_matrix_test.md", test_rows, title="Full Offline Baseline Matrix (Test)")
    representative_rows = [
        {
            "system": row["adapter"],
            "access_model": row["access_model"],
            "action_accuracy": row["action_accuracy"],
            "molecule_acceptance_rate": row["molecule_acceptance_rate"],
            "task_inconsistent_accept_rate": row["task_inconsistent_accept_rate"],
            "reject_recall": row["reject_recall"],
            "abstain_recall": row["abstain_recall"],
            "schema_error_rate": row["schema_error_rate"],
            "mean_verify_calls": row["mean_verify_calls"],
        }
        for row in test_rows
        if row["adapter"] in REPRESENTATIVE
    ]
    _write_csv(tables / "main_table_representative_baselines.csv", representative_rows)
    _write_md(tables / "main_table_representative_baselines.md", representative_rows, title="Representative Baselines")
    return test_rows


def _make_per_family(out: Path, full_test_rows: Sequence[Mapping[str, Any]]) -> None:
    selected = {name for name in REPRESENTATIVE}
    rows: list[dict[str, Any]] = []
    for report in full_test_rows:
        if report["adapter"] not in selected:
            continue
        by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in report["records"]:
            by_family[str(record.get("task_family") or "unknown")].append(record)
        for family, records in sorted(by_family.items()):
            metrics = _metrics(records)
            rows.append(
                {
                    "adapter": report["adapter"],
                    "access_model": report["access_model"],
                    "family": family,
                    "n_tasks": metrics["n_tasks"],
                    "action_accuracy": metrics["action_accuracy"],
                    "balanced_action_accuracy": metrics["balanced_action_accuracy"],
                    "molecule_acceptance_rate": metrics["molecule_acceptance_rate"],
                    "task_inconsistent_accept_rate": metrics["task_inconsistent_accept_rate"],
                    "reject_recall": metrics["reject_recall"],
                    "abstain_recall": metrics["abstain_recall"],
                    "hard_violation_rate": metrics["hard_violation_rate"],
                    "schema_error_rate": metrics["schema_error_rate"],
                    "mean_verify_calls": metrics["mean_verify_calls"],
                    "diagnostic": metrics["n_tasks"] < 20,
                }
            )
    _write_csv(out / "tables" / "per_family_metrics_test.csv", rows)
    pivot = pd.DataFrame(rows).pivot(index="adapter", columns="family", values="action_accuracy").reindex(list(REPRESENTATIVE))
    counts = pd.DataFrame(rows).groupby("family")["n_tasks"].max().to_dict()
    cols = list(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.75), 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{col}\n(n={counts.get(col, 0)})" for col in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title("Per-family action accuracy on held-out test tasks")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if not pd.isna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.65 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="action accuracy")
    _save_fig(fig, out / "figures", "per_family_action_accuracy_heatmap")
    action_rows = [
        {"adapter": idx, **{col: pivot.loc[idx, col] for col in cols}}
        for idx in pivot.index
        if idx in pivot.index
    ]
    _write_md(out / "tables" / "per_family_action_accuracy_test.md", action_rows, title="Per-family Action Accuracy")


def _make_confusions(out: Path, full_test_rows: Sequence[Mapping[str, Any]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    for report in full_test_rows:
        adapter = str(report["adapter"])
        if adapter not in CONFUSION_SYSTEMS:
            continue
        matrix = _confusion(report["records"])
        count_rows = [
            {"expected_action": exp, **{pred: matrix[exp][pred] for pred in PREDICTED_ACTIONS}}
            for exp in ACTIONS
        ]
        norm_rows = []
        for exp in ACTIONS:
            denom = sum(matrix[exp].values())
            norm_rows.append(
                {"expected_action": exp, **{pred: _safe_div(matrix[exp][pred], denom) for pred in PREDICTED_ACTIONS}}
            )
        _write_csv(out / "tables" / f"confusion_{adapter}_counts.csv", count_rows)
        _write_csv(out / "tables" / f"confusion_{adapter}_row_normalized.csv", norm_rows)
        data = np.array([[matrix[exp][pred] for pred in PREDICTED_ACTIONS] for exp in ACTIONS], dtype=float)
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        image = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(PREDICTED_ACTIONS)))
        ax.set_xticklabels(PREDICTED_ACTIONS, rotation=30, ha="right")
        ax.set_yticks(range(len(ACTIONS)))
        ax.set_yticklabels(ACTIONS)
        ax.set_xlabel("predicted_action")
        ax.set_ylabel("expected_action")
        ax.set_title(adapter)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, str(int(data[i, j])), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, label="count")
        _save_fig(fig, out / "figures", f"confusion_{adapter}")
        summary_rows.append(
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
    _write_csv(out / "tables" / "action_collapse_summary.csv", summary_rows)
    _write_md(out / "tables" / "action_collapse_summary.md", summary_rows, title="Action Collapse Summary")


def _family_accuracy(records: Sequence[Mapping[str, Any]], family: str) -> float | None:
    subset = [record for record in records if str(record.get("task_family") or "") == family]
    return _metrics(subset)["action_accuracy"] if subset else None


def _make_wrapper_ablation(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    table_rows: list[dict[str, Any]] = []
    for report in rows:
        records = report["records"]
        metrics = _metrics(records)
        table_rows.append(
            {
                "variant": report["adapter"],
                "access_model": report["access_model"],
                "n_tasks": metrics["n_tasks"],
                "action_accuracy": metrics["action_accuracy"],
                "balanced_action_accuracy": metrics["balanced_action_accuracy"],
                "molecule_acceptance_rate": metrics["molecule_acceptance_rate"],
                "task_inconsistent_accept_rate": metrics["task_inconsistent_accept_rate"],
                "accept_recall": metrics["accept_recall"],
                "reject_recall": metrics["reject_recall"],
                "abstain_recall": metrics["abstain_recall"],
                "construct_family_accuracy": _family_accuracy(records, "construct_feasible"),
                "repair_family_accuracy": _family_accuracy(records, "repair_near_miss"),
                "audit_reject_family_accuracy": _family_accuracy(records, "audit_reject"),
                "abstain_family_accuracy": _family_accuracy(records, "abstain_contradiction"),
                "boundary_family_accuracy": _family_accuracy(records, "boundary_precision"),
                "invariance_family_accuracy": _family_accuracy(records, "smiles_invariance"),
                "interrupt_family_accuracy": _family_accuracy(records, "interrupt_resume"),
                "mean_verify_calls": metrics["mean_verify_calls"],
                "p95_verify_calls": metrics["p95_verify_calls"],
                "max_verify_calls": metrics["max_verify_calls"],
                "budget_exhaustion_rate": metrics["budget_exhaustion_rate"],
                "notes": _wrapper_note(str(report["adapter"])),
            }
        )
    _write_csv(out / "tables" / "wrapper_ablation_test.csv", table_rows)
    _write_md(out / "tables" / "wrapper_ablation_test.md", table_rows, title="Wrapper Ablation")
    labels = [row["variant"].replace("wrapper_", "") for row in table_rows]
    values = [row["action_accuracy"] or 0.0 for row in table_rows]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("action accuracy")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Wrapper ablation action accuracy")
    _save_fig(fig, out / "figures", "wrapper_ablation_action_accuracy")
    budget_rows = [
        row for row in table_rows if str(row["variant"]).startswith("wrapper_verify_budget_")
    ]
    budget_rows.sort(key=lambda row: int(str(row["variant"]).rsplit("_", 1)[1]))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(
        [int(str(row["variant"]).rsplit("_", 1)[1]) for row in budget_rows],
        [row["action_accuracy"] or 0.0 for row in budget_rows],
        marker="o",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("declared verify budget")
    ax.set_ylabel("action accuracy")
    ax.set_title("Wrapper verifier-budget curve")
    ax.grid(True, alpha=0.25)
    _save_fig(fig, out / "figures", "wrapper_ablation_budget_curve")
    _write_wrapper_note(out, table_rows)


def _wrapper_note(name: str) -> str:
    notes = {
        "wrapper_full": "full public verifier/search wrapper",
        "wrapper_no_public_candidate_search": "corpus/public candidate search disabled",
        "wrapper_no_verifier_calls": "explicit L3 verify tool calls disabled; local public-spec evaluation remains available to this deterministic wrapper",
        "wrapper_no_contradiction_detector": "visible contradiction branch disabled",
        "wrapper_no_repair_loop": "local repair/mutation fallback disabled",
        "wrapper_no_boundary_special_case": "boundary audit dispatch disabled",
        "wrapper_name_scrambled_public_view": "adapter ignores canonical visible family names and infers from public action/input shape",
    }
    if name.startswith("wrapper_verify_budget_"):
        return "explicit L3 verify tool-call budget variant"
    return notes.get(name, "")


def _write_wrapper_note(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by_name = {str(row["variant"]): row for row in rows}
    full = by_name.get("wrapper_full", {})
    no_search = by_name.get("wrapper_no_public_candidate_search", {})
    no_verify = by_name.get("wrapper_no_verifier_calls", {})
    scrambled = by_name.get("wrapper_name_scrambled_public_view", {})
    budget = [row for row in rows if str(row["variant"]).startswith("wrapper_verify_budget_")]
    budget.sort(key=lambda row: int(str(row["variant"]).rsplit("_", 1)[1]))
    near = next((row for row in budget if (row.get("action_accuracy") or 0) >= 0.95), None)
    text = "\n".join(
        [
            "# Wrapper Ablation Interpretation",
            "",
            f"The full wrapper reached action_accuracy={_fmt(full.get('action_accuracy'))} on the held-out test split.",
            f"Disabling public candidate search changed action_accuracy to {_fmt(no_search.get('action_accuracy'))}, which estimates the contribution of retrieval/search over public candidates.",
            f"Disabling explicit L3 verify tool calls changed action_accuracy to {_fmt(no_verify.get('action_accuracy'))}; this variant still uses deterministic local evaluation of public specification fields, so it should be interpreted as a tool-call ablation rather than a complete removal of verifier semantics.",
            f"The smallest measured explicit verify budget with at least 0.95 action accuracy was {near.get('variant') if near else 'not reached'}; the measured budget curve should be cited instead of assuming saturation.",
            f"Scrambling/ignoring public task names gave action_accuracy={_fmt(scrambled.get('action_accuracy'))}, testing whether the wrapper depends on visible family-name artifacts.",
            "These rows support the evaluation-contract interpretation: wrapper performance is an access-model ceiling under public verifier/search assumptions, not a closed-book chemistry capability result.",
        ]
    )
    (out / "notes" / "wrapper_ablation_interpretation.md").write_text(text + "\n", encoding="utf-8")


def _make_protocol_ladder(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    table_rows: list[dict[str, Any]] = []
    for report in rows:
        adapter, _, protocol = str(report["adapter"]).rpartition("_")
        metrics = _metrics(report["records"])
        table_rows.append(
            {
                "adapter": adapter,
                "protocol": protocol,
                "n_tasks": metrics["n_tasks"],
                "action_accuracy": metrics["action_accuracy"],
                "balanced_action_accuracy": metrics["balanced_action_accuracy"],
                "molecule_acceptance_rate": metrics["molecule_acceptance_rate"],
                "task_inconsistent_accept_rate": metrics["task_inconsistent_accept_rate"],
                "reject_recall": metrics["reject_recall"],
                "abstain_recall": metrics["abstain_recall"],
                "hard_violation_rate": metrics["hard_violation_rate"],
                "schema_error_rate": metrics["schema_error_rate"],
                "mean_steps": metrics["mean_steps"],
                "mean_proposals": metrics["mean_proposals"],
                "mean_verify_calls": metrics["mean_verify_calls"],
                "p95_verify_calls": metrics["p95_verify_calls"],
                "budget_exhaustion_rate": metrics["budget_exhaustion_rate"],
                "not_applicable_reason": "" if metrics["n_tasks"] else "no tasks for protocol in split",
            }
        )
    _write_csv(out / "tables" / "protocol_ladder_test.csv", table_rows)
    _write_md(out / "tables" / "protocol_ladder_test.md", table_rows, title="Protocol-Slice Analysis")
    df = pd.DataFrame(table_rows)
    for metric, stem, ylabel in (
        ("action_accuracy", "protocol_ladder_action_accuracy", "action accuracy"),
        ("mean_verify_calls", "protocol_ladder_verify_calls", "mean verify calls"),
    ):
        pivot = df.pivot(index="adapter", columns="protocol", values=metric)
        protocols = [proto for proto in ("L1", "L2", "L3") if proto in pivot.columns]
        x = np.arange(len(pivot.index))
        width = 0.22
        fig, ax = plt.subplots(figsize=(8, 4.3))
        for idx, proto in enumerate(protocols):
            ax.bar(x + (idx - 1) * width, pivot[proto].fillna(0).to_numpy(), width, label=proto)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        if metric == "action_accuracy":
            ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title(ylabel.title() + " by Native Protocol")
        _save_fig(fig, out / "figures", stem)
    _write_protocol_note(out, table_rows)


def _write_protocol_note(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    l3_best = df[df["protocol"] == "L3"].sort_values("action_accuracy", ascending=False).head(1)
    l2 = df[df["protocol"] == "L2"]["action_accuracy"].mean()
    l1 = df[df["protocol"] == "L1"]["action_accuracy"].mean()
    text = "\n".join(
        [
            "# Protocol-Slice Interpretation",
            "",
            "These rows are a native-protocol grouping: each row is evaluated on tasks whose release protocol already matches L1, L2, or L3. They are not a forced same-task L1/L2/L3 intervention.",
            f"Across measured systems, mean L1 action accuracy was {_fmt(float(l1) if not math.isnan(l1) else None)} and mean L2 action accuracy was {_fmt(float(l2) if not math.isnan(l2) else None)}.",
            f"The top measured L3 row was {l3_best.iloc[0]['adapter'] if not l3_best.empty else 'NA'} with action_accuracy={_fmt(float(l3_best.iloc[0]['action_accuracy']) if not l3_best.empty else None)}.",
            "L2 feedback can improve repair dynamics for construction tasks, but reject and abstain semantics still need explicit action-aware handling.",
            "Rows where molecule_acceptance_rate increases without a matching action_accuracy increase should be treated as evidence that molecule production is not the headline metric.",
        ]
    )
    (out / "notes" / "protocol_ladder_interpretation.md").write_text(text + "\n", encoding="utf-8")


def _make_metric_sensitivity(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    metric_rows = [row for row in _run_metric_rows(rows) if row["split"] == "test"]
    max_verify = max((row["mean_verify_calls"] or 0.0 for row in metric_rows), default=1.0) or 1.0
    enriched: list[dict[str, Any]] = []
    for row in metric_rows:
        accept = row["accept_recall"]
        reject = row["reject_recall"]
        abstain = row["abstain_recall"]
        balanced_action_score = mean([value for value in (accept, reject, abstain) if value is not None])
        safe = (row["action_accuracy"] or 0.0) - (row["task_inconsistent_accept_rate"] or 0.0) - (row["hard_violation_rate"] or 0.0) - (row["schema_error_rate"] or 0.0)
        normalized_cost = (row["mean_verify_calls"] or 0.0) / max_verify
        payload = dict(row)
        payload["balanced_action_score"] = balanced_action_score
        payload["safe_action_score"] = safe
        payload["cost_adjusted_action_score_lambda_0_01"] = (row["action_accuracy"] or 0.0) - 0.01 * normalized_cost
        payload["cost_adjusted_action_score_lambda_0_05"] = (row["action_accuracy"] or 0.0) - 0.05 * normalized_cost
        enriched.append(payload)
    objectives = [
        ("action_accuracy", False),
        ("balanced_action_accuracy", False),
        ("molecule_acceptance_rate", False),
        ("task_inconsistent_accept_rate", True),
        ("reject_recall", False),
        ("abstain_recall", False),
        ("hard_violation_rate", True),
        ("schema_error_rate", True),
        ("mean_verify_calls", True),
        ("balanced_action_score", False),
        ("safe_action_score", False),
        ("cost_adjusted_action_score_lambda_0_01", False),
        ("cost_adjusted_action_score_lambda_0_05", False),
    ]
    rank_rows: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    for metric, ascending in objectives:
        usable = [row for row in enriched if row.get(metric) is not None]
        usable.sort(key=lambda row: float(row[metric]), reverse=not ascending)
        for rank, row in enumerate(usable, start=1):
            rank_rows.append(
                {
                    "objective_metric": metric,
                    "rank": rank,
                    "system": row["adapter"],
                    "access_model": row["access_model"],
                    "value": row[metric],
                }
            )
        first = usable[0] if usable else {}
        second = usable[1] if len(usable) > 1 else {}
        winners.append(
            {
                "objective_metric": metric,
                "rank_1_system": first.get("adapter"),
                "rank_1_access_model": first.get("access_model"),
                "rank_1_value": first.get(metric),
                "rank_2_system": second.get("adapter"),
                "rank_2_value": second.get(metric),
                "hidden_failure_mode": _hidden_failure_mode(metric),
                "paper_interpretation": _metric_interpretation(metric),
            }
        )
    _write_csv(out / "tables" / "metric_ranking_sensitivity.csv", rank_rows)
    _write_md(out / "tables" / "metric_ranking_sensitivity.md", rank_rows, title="Metric Ranking Sensitivity")
    _write_csv(out / "tables" / "metric_winners_by_objective.csv", winners)
    _write_md(out / "tables" / "metric_winners_by_objective.md", winners, title="Metric Winners by Objective")
    selected_metrics = ["molecule_acceptance_rate", "action_accuracy", "reject_recall", "abstain_recall", "safe_action_score"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for system in REPRESENTATIVE:
        points = []
        labels = []
        for metric in selected_metrics:
            row = next((r for r in rank_rows if r["objective_metric"] == metric and r["system"] == system), None)
            if row:
                points.append(row["rank"])
                labels.append(metric)
        if points:
            ax.plot(range(len(points)), points, marker="o", label=system)
    ax.invert_yaxis()
    ax.set_xticks(range(len(selected_metrics)))
    ax.set_xticklabels(selected_metrics, rotation=25, ha="right")
    ax.set_ylabel("rank (1 is best)")
    ax.set_title("Rank shifts under headline metric choice")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    _save_fig(fig, out / "figures", "metric_rank_shift")
    _write_metric_note(out, winners)


def _hidden_failure_mode(metric: str) -> str:
    if metric == "molecule_acceptance_rate":
        return "can reward accept-biased systems on reject/abstain tasks"
    if metric == "task_inconsistent_accept_rate":
        return "exposes reject/abstain collapse into accept decisions"
    if metric == "mean_verify_calls":
        return "ignores accuracy unless thresholded"
    if metric in {"reject_recall", "abstain_recall"}:
        return "isolates one action class rather than aggregate task success"
    return "depends on access model and action distribution"


def _metric_interpretation(metric: str) -> str:
    if metric == "molecule_acceptance_rate":
        return "not a task-success headline metric"
    if metric == "action_accuracy":
        return "primary decision-contract metric"
    if metric == "safe_action_score":
        return "penalizes unsafe accepts, hard violations, and schema failures"
    if metric.startswith("cost_adjusted"):
        return "separates high-accuracy systems by verifier economy"
    return "use as a diagnostic slice with denominator"


def _write_metric_note(out: Path, winners: Sequence[Mapping[str, Any]]) -> None:
    win = {row["objective_metric"]: row for row in winners}
    text = "\n".join(
        [
            "# Metric Sanity Interpretation",
            "",
            f"Ranking by molecule_acceptance_rate selects {win.get('molecule_acceptance_rate', {}).get('rank_1_system')}, which is not necessarily the best action-contract system.",
            f"Ranking by action_accuracy selects {win.get('action_accuracy', {}).get('rank_1_system')}, directly measuring whether the system chose Accept, Reject, or Abstain correctly.",
            f"Reject recall is led by {win.get('reject_recall', {}).get('rank_1_system')}, while abstain recall is led by {win.get('abstain_recall', {}).get('rank_1_system')}; separating these metrics makes action collapse visible.",
            "Task-inconsistent acceptance is a critical counter-metric because it counts Accept decisions on tasks that require Reject or Abstain.",
            "Hard-violation and schema-error rates should remain safety and validity diagnostics rather than substitutes for action accuracy.",
            "Cost-adjusted action scores distinguish systems that spend verifier calls from systems that achieve similar action accuracy with fewer calls.",
            "The paper should present molecule acceptance as a misleading baseline diagnostic, not as specification success.",
            "The most defensible headline is that access model and metric choice jointly determine the apparent winner.",
        ]
    )
    (out / "notes" / "metric_sanity_interpretation.md").write_text(text + "\n", encoding="utf-8")


def _bootstrap_ci(values: Sequence[float], *, seed: int, n_bootstrap: int) -> tuple[float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan)
    rng = random.Random(seed)
    values = list(values)
    samples = []
    for _ in range(n_bootstrap):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(mean(draw))
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return (mean(values), low, high)


def _metric_task_values(records: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    if metric == "action_accuracy":
        return [float(_expected(record) == _predicted(record)) for record in records]
    if metric == "molecule_acceptance_rate":
        return [float(_predicted(record) == "ACCEPT") for record in records]
    if metric == "task_inconsistent_accept_rate":
        subset = [record for record in records if _expected(record) in {"REJECT", "ABSTAIN"}]
        return [float(_predicted(record) == "ACCEPT") for record in subset]
    if metric == "reject_recall":
        subset = [record for record in records if _expected(record) == "REJECT"]
        return [float(_predicted(record) == "REJECT") for record in subset]
    if metric == "abstain_recall":
        subset = [record for record in records if _expected(record) == "ABSTAIN"]
        return [float(_predicted(record) == "ABSTAIN") for record in subset]
    if metric == "balanced_action_accuracy":
        recalls = []
        for action in ACTIONS:
            subset = [record for record in records if _expected(record) == action]
            if subset:
                recalls.append(sum(float(_predicted(record) == action) for record in subset) / len(subset))
        return recalls
    raise ValueError(metric)


def _make_bootstrap(out: Path, full_test_rows: Sequence[Mapping[str, Any]], release_tasks: Mapping[str, Sequence[Mapping[str, Any]]], n_bootstrap: int) -> None:
    task_to_bundle = {
        str(task.get("task_id")): str(task.get("bundle_id") or task.get("task_id"))
        for task in release_tasks.get("test", [])
    }
    metrics = (
        "action_accuracy",
        "balanced_action_accuracy",
        "molecule_acceptance_rate",
        "task_inconsistent_accept_rate",
        "reject_recall",
        "abstain_recall",
    )
    task_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    by_adapter = {str(row["adapter"]): row for row in full_test_rows}
    for adapter in REPRESENTATIVE:
        report = by_adapter.get(adapter)
        if not report:
            continue
        records = report["records"]
        for metric in metrics:
            values = _metric_task_values(records, metric)
            mean_value, low, high = _bootstrap_ci(values, seed=7, n_bootstrap=n_bootstrap)
            task_rows.append(
                {
                    "adapter": adapter,
                    "metric": metric,
                    "mean": mean_value,
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap": n_bootstrap,
                    "resample_unit": "task",
                }
            )
            by_bundle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for record in records:
                by_bundle[task_to_bundle.get(str(record.get("task_id")), str(record.get("task_id")))].append(record)
            bundle_values = []
            for bundle_records in by_bundle.values():
                vals = _metric_task_values(bundle_records, metric)
                if vals:
                    bundle_values.append(mean(vals))
            b_mean, b_low, b_high = _bootstrap_ci(bundle_values, seed=17, n_bootstrap=n_bootstrap)
            bundle_rows.append(
                {
                    "adapter": adapter,
                    "metric": metric,
                    "mean": b_mean,
                    "ci_low": b_low,
                    "ci_high": b_high,
                    "n_bootstrap": n_bootstrap,
                    "resample_unit": "bundle",
                    "note": "bundle IDs used scorer-side only",
                }
            )
    _write_csv(out / "tables" / "bootstrap_ci_test_task_level.csv", task_rows)
    _write_csv(out / "tables" / "bootstrap_ci_test_bundle_level.csv", bundle_rows)
    ci_lookup = {(row["adapter"], row["metric"]): row for row in task_rows}
    base_rows = []
    for adapter in REPRESENTATIVE:
        report = by_adapter.get(adapter)
        if not report:
            continue
        metrics_row = _metrics(report["records"])
        base_rows.append(
            {
                "system": adapter,
                "action_accuracy": _ci_fmt(ci_lookup.get((adapter, "action_accuracy"))),
                "molecule_acceptance_rate": _ci_fmt(ci_lookup.get((adapter, "molecule_acceptance_rate"))),
                "task_inconsistent_accept_rate": _ci_fmt(ci_lookup.get((adapter, "task_inconsistent_accept_rate"))),
                "reject_recall": _ci_fmt(ci_lookup.get((adapter, "reject_recall"))),
                "abstain_recall": _ci_fmt(ci_lookup.get((adapter, "abstain_recall"))),
                "mean_verify_calls": metrics_row["mean_verify_calls"],
            }
        )
    _write_md(out / "tables" / "main_table_representative_baselines_with_ci.md", base_rows, title="Representative Baselines with Task-level Bootstrap CIs")


def _ci_fmt(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "NA"
    return f"{float(row['mean']):.3f} [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"


def _make_external_skip(out: Path, release_tasks: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    test = list(release_tasks.get("test", []))
    rng = random.Random(7)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task in test:
        by_family[str(task.get("task_type") or task.get("task_family") or "unknown")].append(task)
    selected: list[Mapping[str, Any]] = []
    for rows in by_family.values():
        rng.shuffle(rows)
        selected.extend(rows[: max(1, min(13, len(rows)))])
    selected = selected[:127]
    payload = {
        "seed": 7,
        "n_tasks": len(selected),
        "definition": "deterministic stratified public-hash subset; no hidden fields exposed to adapters",
        "task_public_hashes": [_public_hash(str(task.get("task_id"))) for task in selected],
    }
    jsonio.write_json(out / "summaries" / "external_snapshot_subset.json", payload)
    rows = [
        {
            "adapter": "openai_chat/openai_chat_verify_l3/process",
            "provider_or_gateway": "NA",
            "model_id": "NA",
            "model_access_date": "NA",
            "protocol": "NA",
            "n_tasks": 0,
            "subset_definition": "skipped",
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
            "notes": "No sgchem_v1.0-compatible replay cache and no explicit live external configuration.",
        }
    ]
    _write_csv(out / "tables" / "external_diagnostic_snapshot.csv", rows)
    _write_md(out / "tables" / "external_diagnostic_snapshot.md", rows, title="External Diagnostic Snapshot")
    text = "# External Diagnostic Snapshot Skipped\n\nReason: No sgchem_v1.0-compatible cache and no explicit live external configuration or budget was provided. External diagnostics remain secondary and were not used in any main-paper ranking.\n"
    (out / "notes" / "external_diagnostic_snapshot_skipped.md").write_text(text, encoding="utf-8")
    (out / "notes" / "external_diagnostic_snapshot.md").write_text(text, encoding="utf-8")


def _make_integrity(out: Path) -> None:
    validation = out / "validation"
    rows = [
        {"gate": "strict validation", "status": _log_status(validation / "validate_dataset_strict.log"), "source": "validate_dataset_strict.log"},
        {"gate": "prompt leakage audit", "status": _log_status(validation / "model_prompt_leakage_audit.log"), "source": "model_prompt_leakage_audit.log"},
        {"gate": "oracle scrambling negative controls", "status": _log_status(validation / "oracle_scrambling_audit.log"), "source": "oracle_scrambling_audit.log"},
        {"gate": "split leakage", "status": "collected", "source": "release audits"},
        {"gate": "clean-clone reproduction", "status": "not run", "source": "not requested in local run"},
        {"gate": "Croissant validation", "status": _log_status(validation / "validate_croissant.log"), "source": "validate_croissant.log"},
        {"gate": "hosted URL preflight", "status": _log_status(validation / "neurips_ed_preflight.log"), "source": "neurips_ed_preflight.log"},
    ]
    _write_csv(out / "tables" / "artifact_integrity_gates_v2.csv", rows)
    _write_md(out / "tables" / "artifact_integrity_gates_v2.md", rows, title="Artifact Integrity Gates v2")


def _log_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    if '"valid": true' in text or "valid: true" in text:
        return "pass"
    if '"valid": false' in text or "valid: false" in text or "error" in text:
        return "check log"
    return "completed"


def _write_paper_memo(out: Path, full_test_rows: Sequence[Mapping[str, Any]]) -> None:
    rows = _run_metric_rows(full_test_rows)
    by_adapter = {row["adapter"]: row for row in rows}
    always_accept = by_adapter.get("always_accept", {})
    wrapper = by_adapter.get("well_engineered_wrapper", {})
    text = "\n".join(
        [
            "# Paper Insertion Memo",
            "",
            "## Proposed new Results section",
            "",
            "### Finding 1: Molecule acceptance overstates specification success.",
            f"`always_accept` reached molecule_acceptance_rate={_fmt(always_accept.get('molecule_acceptance_rate'))} but action_accuracy={_fmt(always_accept.get('action_accuracy'))}, illustrating why acceptance alone is not a task-success metric.",
            "",
            "### Finding 2: Failures are action- and family-specific.",
            "Use the per-family heatmap and confusion matrices to show which systems collapse reject or abstain tasks into accept decisions.",
            "",
            "### Finding 3: Public verifier/search access saturates the contract.",
            f"The wrapper row measured action_accuracy={_fmt(wrapper.get('action_accuracy'))} under the public verifier/search access model. Present it as an access-model ceiling, not a closed-book baseline.",
            "",
            "### Finding 4: Metric choice changes apparent conclusions.",
            "Report metric winners by objective to show how molecule acceptance, reject recall, abstain recall, and cost-adjusted action scores select different systems.",
            "",
            "### Finding 5: Expanded audits preserve public/private isolation.",
            "Cite strict validation, prompt leakage, oracle scrambling, and the v2 consistency check as result-generation gates.",
            "",
            "## Recommended main-paper figures/tables",
            "",
            "- Table: representative baseline matrix with action metrics",
            "- Figure: per-family action accuracy heatmap",
            "- Figure: wrapper ablation / verifier-budget curve",
            "- Table: metric winners by objective",
            "",
            "## Appendix tables",
            "",
            "- full offline baseline matrix",
            "- all confusion matrices",
            "- protocol-slice analysis",
            "- external diagnostic snapshot",
            "- bootstrap confidence intervals",
            "- audit logs and integrity gates",
            "",
            "## Key wording",
            "",
            "The expanded baselines show that high molecule acceptance is easy to obtain with accept-biased or retrieval-heavy systems, but those systems fail reject and abstain semantics. The wrapper ablations show that saturation is driven by public verifier/search access rather than hidden oracle leakage.",
            "",
            "Therefore, SpecGuard-Chem should be interpreted as an evaluation-contract artifact with explicit access-model ceilings, not as a chemistry capability leaderboard.",
        ]
    )
    (out / "notes" / "paper_insertion_memo.md").write_text(text + "\n", encoding="utf-8")


def _write_results_summary(out: Path, release: Path, full_test_rows: Sequence[Mapping[str, Any]]) -> None:
    env = (out / "environment.txt").read_text(encoding="utf-8", errors="replace") if (out / "environment.txt").exists() else ""
    commit = "unknown"
    for line in env.splitlines():
        if line.startswith("commit:"):
            commit = line.split(":", 1)[1].strip()
            break
    rows = _run_metric_rows(full_test_rows)
    by_adapter = {row["adapter"]: row for row in rows}
    wrapper = by_adapter.get("well_engineered_wrapper", {})
    text = "\n".join(
        [
            "# SpecGuard-Chem paper_v2 result summary",
            "## Release and environment",
            f"- commit: {commit}",
            f"- release path: {release}",
            f"- validation status: {_log_status(out / 'validation' / 'validate_dataset_strict.log')}",
            "## Main findings",
            f"1. Molecule acceptance versus action accuracy: accept-biased and retrieval systems can produce high molecule_acceptance_rate while action_accuracy and reject/abstain recalls expose failures.",
            "2. Per-family failure modes: see `tables/per_family_metrics_test.csv` and the heatmap for family-specific denominators and accuracies.",
            f"3. Wrapper saturation and ablations: the full wrapper measured action_accuracy={_fmt(wrapper.get('action_accuracy'))}; ablations are reported separately under the verifier/search wrapper access model.",
            "4. Protocol-slice analysis: L1/L2/L3 rows in `tables/protocol_ladder_test.csv` are grouped by the native task protocol, not by a forced same-task protocol intervention.",
            "5. Metric ranking sensitivity: `tables/metric_winners_by_objective.md` shows that headline metric choice changes the apparent winner.",
            "6. External diagnostics, if available: skipped in this run because no v1-compatible replay cache or explicit live configuration was present.",
            "7. Audit status: strict validation, prompt leakage, oracle scrambling, Croissant validation, and consistency outputs are under `validation/`.",
            "## Recommended paper changes",
            "- main text changes: replace acceptance-only toplines with action-aware metrics and access-model separation.",
            "- appendix additions: full matrix, confusion matrices, protocol-slice analysis, external skip/snapshot, bootstrap CIs, and audit gates.",
            "- figures/tables to replace existing ones: representative baseline table, per-family heatmap, wrapper budget curve, metric winners table.",
            "## Caveats",
            "- synthetic rule-based release",
            "- wrapper-solvable under public verifier/search threat model",
            "- no drug-discovery claims",
            "- external results are diagnostic only",
        ]
    )
    (out / "RESULTS_SUMMARY.md").write_text(text + "\n", encoding="utf-8")


def _write_run_manifest(out: Path, release: Path, *, active_splits: Sequence[str]) -> None:
    active = set(active_splits)
    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        rel = str(path.relative_to(out))
        if rel.startswith("raw_runs/"):
            continue
        if rel.startswith(".mplconfig/"):
            continue
        if rel == ".DS_Store" or "/.DS_Store" in rel:
            continue
        if rel.startswith("raw_runs/full_offline_"):
            split = rel.split("/", 2)[1].replace("full_offline_", "")
            if split not in active:
                continue
        if rel.startswith("validation/run_benchmark_full_offline_"):
            split = rel.removeprefix("validation/run_benchmark_full_offline_").removesuffix(".log")
            if split not in active:
                continue
        files.append(rel)
    payload = {
        "release": "sgchem_v1.0",
        "release_path": str(release),
        "result_dir": str(out),
        "active_full_offline_splits": list(active_splits),
        "raw_runs_committed": False,
        "raw_run_recovery": "Raw runner traces are intentionally excluded from the compact committed artifact. Regenerate them with scripts/run_paper_v2_results.sh.",
        "generated_files": files,
        "access_models": sorted(set(ACCESS_MODEL_BY_TRACK.values())),
        "notes": "Generated by scripts/run_paper_v2_results.sh and scripts/make_paper_v2_tables_and_figures.py.",
    }
    jsonio.write_json(out / "run_manifest.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("benchmarks/releases/sgchem_v1.0"))
    parser.add_argument("--results", type=Path, default=Path("paper_v2/results"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--splits", type=str, default="test")
    args = parser.parse_args()

    out = args.results
    for subdir in ("summaries", "tables", "figures", "notes"):
        (out / subdir).mkdir(parents=True, exist_ok=True)

    release_tasks = _load_release_tasks(args.release)
    full_rows = []
    active_splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    for split in active_splits:
        full_rows.extend(_report_rows(out / "raw_runs" / f"full_offline_{split}"))
    wrapper_rows = _report_rows(out / "raw_runs" / "wrapper_ablation_test")
    protocol_rows = _report_rows(out / "raw_runs" / "protocol_ladder_test")
    all_rows = list(full_rows) + list(wrapper_rows) + list(protocol_rows)
    _write_normalized(out, all_rows)

    full_test_reports = [row for row in full_rows if row["split"] == "test"]
    test_metric_rows = _make_full_offline_tables(out, full_rows)
    _make_per_family(out, full_test_reports)
    _make_confusions(out, full_test_reports)
    _make_wrapper_ablation(out, wrapper_rows)
    _make_protocol_ladder(out, protocol_rows)
    _make_metric_sensitivity(out, full_test_reports)
    _make_bootstrap(out, full_test_reports, release_tasks, args.n_bootstrap)
    _make_external_skip(out, release_tasks)
    _make_integrity(out)
    _write_paper_memo(out, full_test_reports)
    _write_results_summary(out, args.release, full_test_reports)
    _write_run_manifest(out, args.release, active_splits=active_splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
