from __future__ import annotations

"""Paper figure/table generation from benchmark sweep outputs."""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..utils import jsonio

TRACKS: tuple[str, ...] = ("closed_book", "retrieval", "external")


def _clear_generated_outputs(figures_dir: Path, tables_dir: Path) -> None:
    for pattern in ("*.png", "*.pdf"):
        for path in figures_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    for pattern in ("*.csv", "*.md"):
        for path in tables_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_aggregate(
    *, runs_dir: Path | None, aggregate_path: Path | None
) -> tuple[Dict[str, Any], Path]:
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
    for row in aggregate.get("all_baselines", aggregate.get("baselines", [])):
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


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _baseline_order(aggregate: Mapping[str, Any]) -> List[str]:
    order = aggregate.get("baseline_order")
    if isinstance(order, list):
        return [str(item) for item in order]
    names: List[str] = []
    for row in aggregate.get("all_baselines", aggregate.get("baselines", [])):
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            names.append(str(row["name"]))
    return names


def _rows_by_track(
    aggregate: Mapping[str, Any], baseline_order: Sequence[str]
) -> Dict[str, List[Dict[str, Any]]]:
    by_track = aggregate.get("by_track")
    if isinstance(by_track, dict):
        rows: Dict[str, List[Dict[str, Any]]] = {}
        for track in TRACKS:
            payload = by_track.get(track)
            if not isinstance(payload, dict):
                rows[track] = []
                continue
            candidates = payload.get("baselines")
            if not isinstance(candidates, list):
                rows[track] = []
                continue
            filtered = [row for row in candidates if isinstance(row, dict)]
            by_name = {str(row.get("name")): row for row in filtered}
            ordered = [by_name[name] for name in baseline_order if name in by_name]
            rows[track] = ordered
        return rows

    rows = [
        row
        for row in aggregate.get("all_baselines", aggregate.get("baselines", []))
        if isinstance(row, dict)
    ]
    by_name = {str(row.get("name")): row for row in rows}
    ordered = [by_name[name] for name in baseline_order if name in by_name]
    buckets: Dict[str, List[Dict[str, Any]]] = {track: [] for track in TRACKS}
    for row in ordered:
        track = str(row.get("track") or "closed_book")
        if track not in buckets:
            buckets[track] = []
        buckets[track].append(row)
    return buckets


def _ci_band_by_step(row: Mapping[str, Any]) -> Dict[int, tuple[float, float]]:
    metrics = row.get("metrics") or {}
    if not isinstance(metrics, dict):
        return {}
    bootstrap = metrics.get("bootstrap_ci")
    if not isinstance(bootstrap, dict):
        return {}
    pass_ci = bootstrap.get("pass_at_steps")
    if not isinstance(pass_ci, list):
        return {}
    band: Dict[int, tuple[float, float]] = {}
    for point in pass_ci:
        if not isinstance(point, dict):
            continue
        step = point.get("step_budget")
        ci_low = _safe_float(point.get("ci_low"))
        ci_high = _safe_float(point.get("ci_high"))
        if isinstance(step, int) and ci_low is not None and ci_high is not None:
            band[step] = (ci_low, ci_high)
    return band


def _plot_pass_at_budget(
    *,
    reports: Mapping[str, Dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stem: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for row in rows:
        name = str(row.get("name"))
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
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=name)

        ci_band = _ci_band_by_step(row)
        lows: List[float] = []
        highs: List[float] = []
        aligned_xs: List[float] = []
        for step in xs:
            step_int = int(step)
            if step_int not in ci_band:
                continue
            low, high = ci_band[step_int]
            aligned_xs.append(step)
            lows.append(low)
            highs.append(high)
        if aligned_xs:
            ax.fill_between(aligned_xs, lows, highs, alpha=0.15)

    ax.set_xlabel("Step Budget")
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, figures_dir, stem)


def _plot_risk_coverage(
    *,
    reports: Mapping[str, Dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stem: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for row in rows:
        name = str(row.get("name"))
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
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, figures_dir, stem)


def _plot_reliability(
    *,
    reports: Mapping[str, Dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stem: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
    for row in rows:
        name = str(row.get("name"))
        payload = reports.get(name) or {}
        records = payload.get("records")
        if not isinstance(records, list):
            continue
        probs: List[float] = []
        truths: List[float] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            prob = _safe_float(record.get("final_p_hard_pass"))
            if prob is None:
                continue
            probs.append(prob)
            truths.append(1.0 if bool(record.get("hard_pass")) else 0.0)
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
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, figures_dir, stem)


def _plot_edit_economy(
    *,
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stem: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for row in rows:
        name = str(row.get("name"))
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        x = _safe_float(metrics.get("avg_final_edit_cost_brics"))
        y = _safe_float(metrics.get("accept_rate"))
        if x is None or y is None:
            continue
        ax.scatter([x], [y], label=name, s=55)
    ax.set_xlabel("Avg Final BRICS Edit Cost")
    ax.set_ylabel("Accept Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, figures_dir, stem)


def _plot_interrupt_resume(
    *,
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stem: str,
    title: str,
) -> None:
    names: List[str] = []
    resume: List[float] = []
    extra_steps: List[float] = []
    for row in rows:
        name = str(row.get("name"))
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        resume_rate = _safe_float(metrics.get("resume_success_rate"))
        extra = _safe_float(metrics.get("avg_extra_steps_after_interrupt"))
        if resume_rate is None and extra is None:
            continue
        names.append(name)
        resume.append(resume_rate if resume_rate is not None else 0.0)
        extra_steps.append(extra if extra is not None else 0.0)

    fig, ax1 = plt.subplots(figsize=(7.3, 4.4))
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
    ax1.set_title(title)
    ax1.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, figures_dir, stem)


def _plot_invariance_failure(
    *,
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    stem: str,
    title: str,
) -> None:
    names: List[str] = []
    values: List[float] = []
    for row in rows:
        name = str(row.get("name"))
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        failure = _safe_float(metrics.get("invariance_failure_rate"))
        if failure is None:
            continue
        names.append(name)
        values.append(failure)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    xs = np.arange(len(names))
    ax.bar(xs, values, color="#7aa6c2")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Failure Rate")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, figures_dir, stem)


def _topline_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = row.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    bootstrap = metrics.get("bootstrap_ci")
    if not isinstance(bootstrap, dict):
        bootstrap = {}

    def _ci_text(metric_name: str) -> str:
        item = bootstrap.get(metric_name)
        if not isinstance(item, dict):
            return ""
        mean = _safe_float(item.get("mean"))
        low = _safe_float(item.get("ci_low"))
        high = _safe_float(item.get("ci_high"))
        if mean is None or low is None or high is None:
            return ""
        return f"{mean:.3f} [{low:.3f}, {high:.3f}]"

    return {
        "baseline": row.get("name"),
        "model": row.get("model"),
        "track": row.get("track") or "closed_book",
        "protocol": row.get("protocol") or "mixed",
        "num_tasks": metrics.get("num_tasks"),
        "pass_at_1": metrics.get("pass_at_1"),
        "pass_at_1_ci95": _ci_text("pass_at_1"),
        "pass_at_3": metrics.get("pass_at_3"),
        "pass_at_3_ci95": _ci_text("pass_at_3"),
        "accept_rate": metrics.get("accept_rate"),
        "hard_violation_rate": metrics.get("hard_violation_rate"),
        "hard_violation_rate_ci95": _ci_text("hard_violation_rate"),
        "abstention_utility": metrics.get("abstention_utility"),
        "abstention_utility_ci95": _ci_text("abstention_utility"),
        "avg_steps_to_accept": metrics.get("avg_steps_to_accept"),
        "avg_verify_calls_used": metrics.get("avg_verify_calls_used"),
        "l3_avg_verify_calls_used": metrics.get("l3_avg_verify_calls_used"),
        "l3_avg_verify_calls_used_expected_accept": metrics.get(
            "l3_avg_verify_calls_used_expected_accept"
        ),
        "verify_usage_rate_on_L3": metrics.get("verify_usage_rate_on_L3"),
        "avg_final_edit_cost_brics": metrics.get("avg_final_edit_cost_brics"),
        "invariance_failure_rate": metrics.get("invariance_failure_rate"),
        "boundary_precision_failure_rate": metrics.get("boundary_precision_failure_rate"),
        "resume_success_rate": metrics.get("resume_success_rate"),
        "avg_extra_steps_after_interrupt": metrics.get("avg_extra_steps_after_interrupt"),
        "brier_score": metrics.get("brier_score"),
        "ece": metrics.get("ece"),
    }


def _family_rows(
    aggregate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    baseline_order: Sequence[str],
) -> List[Dict[str, Any]]:
    by_family = aggregate.get("by_spec_family")
    if not isinstance(by_family, dict):
        return []
    track_by_name = {str(row.get("name")): str(row.get("track") or "closed_book") for row in rows}
    rows_out: List[Dict[str, Any]] = []
    for family in sorted(by_family):
        payload = by_family.get(family)
        if not isinstance(payload, dict):
            continue
        for baseline in baseline_order:
            metrics = payload.get(baseline)
            if not isinstance(metrics, dict):
                continue
            track = track_by_name.get(baseline)
            rows_out.append(
                {
                    "spec_family": family,
                    "baseline": baseline,
                    "track": track,
                    "num_tasks": metrics.get("num_tasks"),
                    "accept_rate": metrics.get("accept_rate"),
                    "hard_violation_rate": metrics.get("hard_violation_rate"),
                    "avg_spec_score": metrics.get("avg_spec_score"),
                }
            )
    return rows_out


def _invariance_subfamily_rows(
    reports: Mapping[str, Dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name"))
        track = str(row.get("track") or "closed_book")
        summary = (reports.get(name) or {}).get("summary") or {}
        by_subfamily = summary.get("invariance_failure_rate_by_subfamily")
        if not isinstance(by_subfamily, dict):
            continue
        counts_by_subfamily = summary.get("invariance_counts_by_subfamily") or {}
        if not isinstance(counts_by_subfamily, dict):
            counts_by_subfamily = {}
        for subfamily in sorted(by_subfamily):
            value = by_subfamily.get(subfamily)
            counts = counts_by_subfamily.get(subfamily) or {}
            n_tasks = counts.get("n_tasks") if isinstance(counts, dict) else None
            n_failures = (
                counts.get("n_failures") if isinstance(counts, dict) else None
            )
            output.append(
                {
                    "baseline": name,
                    "track": track,
                    "invariance_subfamily": subfamily,
                    "failure_rate": value,
                    "n_tasks": n_tasks,
                    "n_failures": n_failures,
                }
            )
    return output


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
    _clear_generated_outputs(figures_dir, tables_dir)

    baseline_order = _baseline_order(aggregate)
    rows_by_track = _rows_by_track(aggregate, baseline_order)
    all_rows = [
        row
        for track in TRACKS
        for row in rows_by_track.get(track, [])
    ]

    for track, rows in rows_by_track.items():
        if not rows:
            continue
        label = track.replace("_", " ").title()
        _plot_pass_at_budget(
            reports=reports,
            rows=rows,
            figures_dir=figures_dir,
            stem=f"pass_at_budget_{track}",
            title=f"Pass@Budget ({label} Track)",
        )
        _plot_risk_coverage(
            reports=reports,
            rows=rows,
            figures_dir=figures_dir,
            stem=f"risk_coverage_{track}",
            title=f"Risk-Coverage ({label} Track)",
        )
        _plot_reliability(
            reports=reports,
            rows=rows,
            figures_dir=figures_dir,
            stem=f"calibration_reliability_{track}",
            title=f"Calibration Reliability ({label} Track)",
        )
        _plot_edit_economy(
            rows=rows,
            figures_dir=figures_dir,
            stem=f"edit_economy_vs_pass_rate_{track}",
            title=f"Edit Economy vs Pass Rate ({label} Track)",
        )
        _plot_interrupt_resume(
            rows=rows,
            figures_dir=figures_dir,
            stem=f"interrupt_resume_metrics_{track}",
            title=f"Interrupt Resume ({label} Track)",
        )
        _plot_invariance_failure(
            rows=rows,
            figures_dir=figures_dir,
            stem=f"invariance_failure_rate_{track}",
            title=f"Invariance Failure Rate ({label} Track)",
        )

    # Backward-compatible overall figures.
    if all_rows:
        _plot_pass_at_budget(
            reports=reports,
            rows=all_rows,
            figures_dir=figures_dir,
            stem="pass_at_budget",
            title="Pass@Budget by Baseline",
        )
        _plot_risk_coverage(
            reports=reports,
            rows=all_rows,
            figures_dir=figures_dir,
            stem="risk_coverage",
            title="Risk-Coverage Curve",
        )
        _plot_reliability(
            reports=reports,
            rows=all_rows,
            figures_dir=figures_dir,
            stem="calibration_reliability",
            title="Calibration Reliability Diagram",
        )
        _plot_edit_economy(
            rows=all_rows,
            figures_dir=figures_dir,
            stem="edit_economy_vs_pass_rate",
            title="Edit Economy vs Pass Rate",
        )
        _plot_interrupt_resume(
            rows=all_rows,
            figures_dir=figures_dir,
            stem="interrupt_resume_metrics",
            title="Interrupt Resume Metrics",
        )
        _plot_invariance_failure(
            rows=all_rows,
            figures_dir=figures_dir,
            stem="invariance_failure_rate",
            title="Invariance Failure Rate by Baseline",
        )

    topline_all = [_topline_row(row) for row in all_rows]
    _write_csv(tables_dir / "topline_summary.csv", topline_all)
    _write_md_table(tables_dir / "topline_summary.md", topline_all)
    _write_csv(tables_dir / "topline_summary_all.csv", topline_all)
    _write_md_table(tables_dir / "topline_summary_all.md", topline_all)

    for track, rows in rows_by_track.items():
        topline_track = [_topline_row(row) for row in rows]
        _write_csv(tables_dir / f"topline_summary_{track}.csv", topline_track)
        _write_md_table(tables_dir / f"topline_summary_{track}.md", topline_track)

    by_family = _family_rows(aggregate, all_rows, baseline_order)
    _write_csv(tables_dir / "spec_family_summary.csv", by_family)
    _write_md_table(tables_dir / "spec_family_summary.md", by_family)
    invariance_subfamily = _invariance_subfamily_rows(reports, all_rows)
    _write_csv(tables_dir / "invariance_subfamily_summary.csv", invariance_subfamily)
    _write_md_table(tables_dir / "invariance_subfamily_summary.md", invariance_subfamily)

    summary_lines = [
        "# Paper Metrics Summary",
        "",
        f"- benchmark_id: {aggregate.get('benchmark_id')}",
        f"- split: {aggregate.get('split')}",
        f"- baselines: {aggregate.get('n_baselines')}",
        f"- skipped_baselines: {aggregate.get('n_skipped_baselines', 0)}",
        f"- figures_dir: {figures_dir}",
        f"- tables_dir: {tables_dir}",
        "",
        "Track leaderboards:",
    ]
    for track in TRACKS:
        rows = rows_by_track.get(track, [])
        summary_lines.append(f"- {track}: {len(rows)} baseline(s)")

    summary_lines.extend(
        [
            "",
            "Generated figure stems:",
            "- pass_at_budget[_<track>]",
            "- risk_coverage[_<track>]",
            "- calibration_reliability[_<track>]",
            "- edit_economy_vs_pass_rate[_<track>]",
            "- interrupt_resume_metrics[_<track>]",
            "- invariance_failure_rate[_<track>]",
        ]
    )

    (out_dir / "metrics_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    return {
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "summary_path": str(out_dir / "metrics_summary.md"),
    }
