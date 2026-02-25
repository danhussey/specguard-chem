from __future__ import annotations

"""Baseline sweep orchestration over frozen benchmark splits."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from ..runner.runner import TaskRunner
from ..scoring import reports
from ..utils import jsonio
from .release import load_benchmark_release


@dataclass(frozen=True)
class BaselineEntry:
    name: str
    model: str
    protocol: Optional[str] = None


def load_baseline_matrix(path: Path) -> List[BaselineEntry]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("baselines")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("Baseline config must contain a list of baselines")

    matrix: List[BaselineEntry] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Baseline row #{index} must be an object")
        name = row.get("name")
        model = row.get("model")
        protocol = row.get("protocol")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Baseline row #{index} missing non-empty 'name'")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Baseline row #{index} missing non-empty 'model'")
        if protocol is not None and protocol not in {"L1", "L2", "L3"}:
            raise ValueError(
                f"Baseline row #{index} has invalid protocol '{protocol}' (expected L1/L2/L3)"
            )
        matrix.append(
            BaselineEntry(name=name.strip(), model=model.strip(), protocol=protocol)
        )
    return matrix


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, Any]:
    pass_at_budget = summary.get("pass_at_steps") if isinstance(summary.get("pass_at_steps"), list) else []
    pass_at_1 = None
    pass_at_3 = None
    for point in pass_at_budget:
        if not isinstance(point, dict):
            continue
        if point.get("step_budget") == 1:
            pass_at_1 = point.get("pass_rate")
        if point.get("step_budget") == 3:
            pass_at_3 = point.get("pass_rate")
    return {
        "num_tasks": summary.get("num_tasks"),
        "accept_rate": summary.get("accept_rate"),
        "hard_violation_rate": summary.get("hard_violation_rate"),
        "abstention_utility": summary.get("abstention_utility"),
        "pass_at_1": pass_at_1,
        "pass_at_3": pass_at_3,
        "avg_steps_to_accept": summary.get("avg_steps_to_accept"),
        "avg_verify_calls_to_accept": summary.get("avg_verify_calls_to_accept"),
        "avg_final_edit_cost_brics": summary.get("avg_final_edit_cost_brics"),
        "avg_trajectory_edit_cost_brics": summary.get("avg_trajectory_edit_cost_brics"),
        "invariance_failure_rate": summary.get("invariance_failure_rate"),
        "boundary_precision_failure_rate": summary.get("boundary_precision_failure_rate"),
        "resume_success_rate": summary.get("resume_success_rate"),
        "avg_extra_steps_after_interrupt": summary.get("avg_extra_steps_after_interrupt"),
        "brier_score": summary.get("brier_score"),
        "ece": summary.get("ece"),
    }


def run_benchmark_sweep(
    *,
    benchmark_dir: Path,
    split: str,
    baselines_path: Path,
    out_dir: Path,
    seed: int = 7,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be one of: train, dev, test")

    release = load_benchmark_release(benchmark_dir)
    tasks = release.load_split_tasks(split)
    if limit is not None:
        tasks = tasks[: max(limit, 0)]

    matrix = load_baseline_matrix(baselines_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows: List[Dict[str, Any]] = []
    by_family: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for entry in matrix:
        run_tasks = tasks
        if entry.protocol is not None:
            run_tasks = [task for task in tasks if task.protocol == entry.protocol]
        run_dir = out_dir / entry.name
        runner = TaskRunner(entry.model, seed=seed)
        runner.run_tasks(
            run_tasks,
            run_dir=run_dir,
            suite=f"{release.benchmark_id}_{split}",
            protocol=entry.protocol or "mixed",
            spec_loader=release.spec_loader,
        )
        records = reports.load_trace(run_dir)
        summary = reports.summarise(records)
        report_path = reports.write_report(run_dir, records=records, summary=summary)

        metrics = _summary_metrics(summary)
        row = {
            "name": entry.name,
            "model": entry.model,
            "protocol": entry.protocol,
            "run_dir": run_dir.name,
            "report_path": str(report_path.relative_to(out_dir)),
            "metrics": metrics,
            "spec_family_breakdown": summary.get("spec_family_breakdown", {}),
        }
        baseline_rows.append(row)

        family_breakdown = summary.get("spec_family_breakdown", {})
        if isinstance(family_breakdown, dict):
            for family, family_metrics in sorted(family_breakdown.items()):
                if not isinstance(family_metrics, dict):
                    continue
                by_family[family][entry.name] = {
                    "num_tasks": family_metrics.get("num_tasks"),
                    "accept_rate": family_metrics.get("accept_rate"),
                    "hard_violation_rate": family_metrics.get("hard_violation_rate"),
                    "avg_spec_score": family_metrics.get("avg_spec_score"),
                }

    aggregate = {
        "benchmark_id": release.benchmark_id,
        "split": split,
        "seed": seed,
        "limit": limit,
        "n_baselines": len(baseline_rows),
        "baselines": baseline_rows,
        "by_spec_family": {
            family: by_family[family] for family in sorted(by_family)
        },
        "baseline_order": [entry.name for entry in matrix],
    }
    jsonio.write_json(out_dir / "aggregate.json", aggregate)
    return aggregate

