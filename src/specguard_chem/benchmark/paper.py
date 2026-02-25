from __future__ import annotations

"""Paper figure/table generation from benchmark sweep outputs."""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..utils import jsonio


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_aggregate(*, runs_dir: Path | None, aggregate_path: Path | None) -> tuple[Dict[str, Any], Path]:
    if aggregate_path is not None:
        aggregate = jsonio.read_json(aggregate_path)
        if not isinstance(aggregate, dict):
            raise ValueError("Aggregate JSON must be an object")
        base_dir = aggregate_path.parent
        return aggregate, base_dir
    if runs_dir is None:
        raise ValueError("Either runs_dir or aggregate_path must be provided")
    aggregate_file = runs_dir / "aggregate.json"
    if not aggregate_file.exists():
        raise FileNotFoundError(f"Missing aggregate.json at {aggregate_file}")
    aggregate = jsonio.read_json(aggregate_file)
    if not isinstance(aggregate, dict):
        raise ValueError("Aggregate JSON must be an object")
    return aggregate, runs_dir


def _load_reports(aggregate: Mapping[str, Any], sweep_dir: Path) -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    for row in aggregate.get("baselines", []):
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        report_rel = row.get("report_path")
        if not isinstance(name, str) or not isinstance(report_rel, str):
            continue
        report_path = sweep_dir / report_rel
        if not report_path.exists():
            continue
        payload = jsonio.read_json(report_path)
        if isinstance(payload, dict):
            reports[name] = payload
    return reports


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
            writer.writerow(dict(row))


def _write_md_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_order(aggregate: Mapping[str, Any]) -> List[str]:
    order = aggregate.get("baseline_order")
    if isinstance(order, list):
        return [str(item) for item in order]
    names: List[str] = []
    for row in aggregate.get("baselines", []):
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            names.append(str(row["name"]))
    return names


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _plot_pass_at_budget(
    *,
    reports: Mapping[str, Dict[str, Any]],
    baseline_order: Iterable[str],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name in baseline_order:
        summary = (reports.get(name) or {}).get("summary") or {}
        points = summary.get("pass_at_steps")
        if not isinstance(points, list) or not points:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            step = _safe_float(point.get("step_budget"))
            rate = _safe_float(point.get("pass_rate"))
            if step is None or rate is None:
                continue
            xs.append(step)
            ys.append(rate)
        if xs:
            ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Step Budget")
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Pass@Budget by Baseline")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, figures_dir, "pass_at_budget")


def _plot_risk_coverage(
    *,
    reports: Mapping[str, Dict[str, Any]],
    baseline_order: Iterable[str],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name in baseline_order:
        summary = (reports.get(name) or {}).get("summary") or {}
        curves = summary.get("risk_coverage_curve") or {}
        points = curves.get("expected_accept") if isinstance(curves, dict) else None
        if not isinstance(points, list) or not points:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            coverage = _safe_float(point.get("coverage"))
            risk = _safe_float(point.get("risk"))
            if coverage is None or risk is None:
                continue
            xs.append(coverage)
            ys.append(risk)
        if xs:
            ax.plot(xs, ys, label=name)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Risk-Coverage Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, figures_dir, "risk_coverage")


def _plot_reliability(
    *,
    reports: Mapping[str, Dict[str, Any]],
    baseline_order: Iterable[str],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
    for name in baseline_order:
        payload = reports.get(name) or {}
        records = payload.get("records")
        if not isinstance(records, list):
            continue
        probs: List[float] = []
        truths: List[float] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            prob = _safe_float(row.get("final_p_hard_pass"))
            if prob is None:
                continue
            probs.append(prob)
            truths.append(1.0 if bool(row.get("hard_pass")) else 0.0)
        if not probs:
            continue
        bins = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(probs, bins) - 1
        xs: List[float] = []
        ys: List[float] = []
        for bin_idx in range(10):
            members = [i for i, idx in enumerate(bin_ids) if idx == bin_idx]
            if not members:
                continue
            xs.append(float(np.mean([probs[i] for i in members])))
            ys.append(float(np.mean([truths[i] for i in members])))
        if xs:
            ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Predicted p_hard_pass")
    ax.set_ylabel("Observed Hard Pass Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Calibration Reliability Diagram")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, figures_dir, "calibration_reliability")


def _plot_edit_economy(
    *,
    aggregate: Mapping[str, Any],
    baseline_order: Iterable[str],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    rows_by_name = {
        str(row.get("name")): row
        for row in aggregate.get("baselines", [])
        if isinstance(row, dict)
    }
    for name in baseline_order:
        row = rows_by_name.get(name)
        if row is None:
            continue
        metrics = row.get("metrics") or {}
        x = _safe_float(metrics.get("avg_final_edit_cost_brics"))
        y = _safe_float(metrics.get("accept_rate"))
        if x is None or y is None:
            continue
        ax.scatter([x], [y], label=name, s=50)
    ax.set_xlabel("Avg Final BRICS Edit Cost")
    ax.set_ylabel("Accept Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Edit Economy vs Pass Rate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, figures_dir, "edit_economy_vs_pass_rate")


def _plot_interrupt_resume(
    *,
    aggregate: Mapping[str, Any],
    baseline_order: Iterable[str],
    figures_dir: Path,
) -> None:
    rows_by_name = {
        str(row.get("name")): row
        for row in aggregate.get("baselines", [])
        if isinstance(row, dict)
    }
    names: List[str] = []
    resume: List[float] = []
    extra_steps: List[float] = []
    for name in baseline_order:
        row = rows_by_name.get(name)
        if row is None:
            continue
        metrics = row.get("metrics") or {}
        resume_rate = _safe_float(metrics.get("resume_success_rate"))
        extra = _safe_float(metrics.get("avg_extra_steps_after_interrupt"))
        if resume_rate is None and extra is None:
            continue
        names.append(name)
        resume.append(resume_rate if resume_rate is not None else 0.0)
        extra_steps.append(extra if extra is not None else 0.0)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    xs = np.arange(len(names))
    ax1.bar(xs - 0.2, resume, width=0.4, label="resume_success_rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Resume Success Rate")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(
        xs + 0.2,
        extra_steps,
        width=0.4,
        color="#ffb347",
        label="avg_extra_steps_after_interrupt",
    )
    ax2.set_ylabel("Avg Extra Steps")
    ax1.set_title("Interrupt Resume Metrics")
    ax1.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, figures_dir, "interrupt_resume_metrics")


def _plot_invariance_failure(
    *,
    aggregate: Mapping[str, Any],
    baseline_order: Iterable[str],
    figures_dir: Path,
) -> None:
    rows_by_name = {
        str(row.get("name")): row
        for row in aggregate.get("baselines", [])
        if isinstance(row, dict)
    }
    names: List[str] = []
    values: List[float] = []
    for name in baseline_order:
        row = rows_by_name.get(name)
        if row is None:
            continue
        metrics = row.get("metrics") or {}
        failure = _safe_float(metrics.get("invariance_failure_rate"))
        if failure is None:
            continue
        names.append(name)
        values.append(failure)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xs = np.arange(len(names))
    ax.bar(xs, values, color="#7aa6c2")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Failure Rate")
    ax.set_title("Invariance Failure Rate by Baseline")
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, figures_dir, "invariance_failure_rate")


def _topline_rows(aggregate: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in aggregate.get("baselines", []):
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") or {}
        rows.append(
            {
                "baseline": row.get("name"),
                "model": row.get("model"),
                "protocol": row.get("protocol") or "mixed",
                "num_tasks": metrics.get("num_tasks"),
                "pass_at_1": metrics.get("pass_at_1"),
                "pass_at_3": metrics.get("pass_at_3"),
                "accept_rate": metrics.get("accept_rate"),
                "hard_violation_rate": metrics.get("hard_violation_rate"),
                "abstention_utility": metrics.get("abstention_utility"),
                "avg_steps_to_accept": metrics.get("avg_steps_to_accept"),
                "avg_final_edit_cost_brics": metrics.get("avg_final_edit_cost_brics"),
                "invariance_failure_rate": metrics.get("invariance_failure_rate"),
                "resume_success_rate": metrics.get("resume_success_rate"),
                "brier_score": metrics.get("brier_score"),
                "ece": metrics.get("ece"),
            }
        )
    return rows


def _family_rows(aggregate: Mapping[str, Any], baseline_order: Sequence[str]) -> List[Dict[str, Any]]:
    by_family = aggregate.get("by_spec_family")
    if not isinstance(by_family, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for family in sorted(by_family):
        payload = by_family.get(family)
        if not isinstance(payload, dict):
            continue
        for baseline in baseline_order:
            metrics = payload.get(baseline)
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "spec_family": family,
                    "baseline": baseline,
                    "num_tasks": metrics.get("num_tasks"),
                    "accept_rate": metrics.get("accept_rate"),
                    "hard_violation_rate": metrics.get("hard_violation_rate"),
                    "avg_spec_score": metrics.get("avg_spec_score"),
                }
            )
    return rows


def make_paper_artifacts(
    *,
    out_dir: Path,
    runs_dir: Path | None = None,
    aggregate_path: Path | None = None,
) -> Dict[str, Any]:
    aggregate, sweep_dir = _load_aggregate(runs_dir=runs_dir, aggregate_path=aggregate_path)
    reports = _load_reports(aggregate, sweep_dir)

    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    order = _baseline_order(aggregate)
    _plot_pass_at_budget(reports=reports, baseline_order=order, figures_dir=figures_dir)
    _plot_risk_coverage(reports=reports, baseline_order=order, figures_dir=figures_dir)
    _plot_reliability(reports=reports, baseline_order=order, figures_dir=figures_dir)
    _plot_edit_economy(aggregate=aggregate, baseline_order=order, figures_dir=figures_dir)
    _plot_interrupt_resume(aggregate=aggregate, baseline_order=order, figures_dir=figures_dir)
    _plot_invariance_failure(aggregate=aggregate, baseline_order=order, figures_dir=figures_dir)

    topline = _topline_rows(aggregate)
    by_family = _family_rows(aggregate, order)
    _write_csv(tables_dir / "topline_summary.csv", topline)
    _write_md_table(tables_dir / "topline_summary.md", topline)
    _write_csv(tables_dir / "spec_family_summary.csv", by_family)
    _write_md_table(tables_dir / "spec_family_summary.md", by_family)

    summary_lines = [
        "# Paper Metrics Summary",
        "",
        f"- benchmark_id: {aggregate.get('benchmark_id')}",
        f"- split: {aggregate.get('split')}",
        f"- baselines: {aggregate.get('n_baselines')}",
        f"- figures_dir: {figures_dir}",
        f"- tables_dir: {tables_dir}",
        "",
        "Generated figures:",
    ]
    for stem in (
        "pass_at_budget",
        "risk_coverage",
        "calibration_reliability",
        "edit_economy_vs_pass_rate",
        "interrupt_resume_metrics",
        "invariance_failure_rate",
    ):
        summary_lines.append(f"- {stem}.png / {stem}.pdf")
    (out_dir / "metrics_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "summary_path": str(out_dir / "metrics_summary.md"),
    }

