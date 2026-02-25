from __future__ import annotations

"""Baseline sweep orchestration over frozen benchmark splits."""

from collections import defaultdict
from dataclasses import dataclass, field
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from ..runner.runner import TaskRunner
from ..scoring import reports
from ..scoring.metrics import decision_utility
from ..utils import jsonio
from .release import load_benchmark_release

TRACKS: tuple[str, ...] = ("closed_book", "retrieval", "external")


@dataclass(frozen=True)
class BaselineEntry:
    name: str
    model: str
    protocol: Optional[str] = None
    track: str = "closed_book"
    optional: bool = False
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)


def _infer_track(model: str) -> str:
    if model == "corpus_search":
        return "retrieval"
    if model in {"openai_chat", "openai_chat_verify_l3", "process"}:
        return "external"
    return "closed_book"


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

        raw_track = row.get("track")
        track = str(raw_track).strip() if isinstance(raw_track, str) else _infer_track(model)
        if track not in TRACKS:
            raise ValueError(
                f"Baseline row #{index} has invalid track '{track}' "
                f"(expected one of: {', '.join(TRACKS)})"
            )

        optional = bool(row.get("optional", False))
        adapter_kwargs = row.get("adapter_kwargs", row.get("model_kwargs", {}))
        if adapter_kwargs is None:
            adapter_kwargs = {}
        if not isinstance(adapter_kwargs, dict):
            raise ValueError(
                f"Baseline row #{index} adapter_kwargs/model_kwargs must be an object"
            )

        matrix.append(
            BaselineEntry(
                name=name.strip(),
                model=model.strip(),
                protocol=protocol,
                track=track,
                optional=optional,
                adapter_kwargs=dict(adapter_kwargs),
            )
        )
    return matrix


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, Any]:
    pass_at_budget = (
        summary.get("pass_at_steps")
        if isinstance(summary.get("pass_at_steps"), list)
        else []
    )
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
        "avg_verify_calls_used": summary.get("avg_verify_calls_used"),
        "l3_avg_verify_calls_used": summary.get("l3_avg_verify_calls_used"),
        "l3_avg_verify_calls_used_expected_accept": summary.get(
            "l3_avg_verify_calls_used_expected_accept"
        ),
        "verify_usage_rate_on_L3": summary.get("verify_usage_rate_on_L3"),
        "brier_score": summary.get("brier_score"),
        "ece": summary.get("ece"),
    }


def _resolve_expected_action(record: Mapping[str, Any]) -> str:
    expected_action = record.get("expected_action")
    if isinstance(expected_action, str):
        value = expected_action.strip().upper()
        if value in {"ACCEPT", "ABSTAIN", "REJECT"}:
            return value
    expected = str(record.get("expected", "PASS")).strip().upper()
    if expected == "PASS":
        return "ACCEPT"
    if expected == "ABSTAIN":
        return "ABSTAIN"
    return "REJECT"


def _resolve_final_decision(record: Mapping[str, Any]) -> str:
    final_decision = record.get("final_decision")
    if isinstance(final_decision, str):
        value = final_decision.strip().upper()
        if value in {"ACCEPT", "ABSTAIN", "REJECT"}:
            return value
    decision = str(record.get("decision", "")).strip().lower()
    if decision == "accept":
        return "ACCEPT"
    if decision == "reject":
        return "REJECT"
    return "ABSTAIN"


def _pass_at_k(records: List[Mapping[str, Any]], k: int) -> Optional[float]:
    accept_indices = [
        index
        for index, record in enumerate(records)
        if _resolve_expected_action(record) == "ACCEPT"
    ]
    if not accept_indices:
        return None
    accepted = 0
    for index in accept_indices:
        record = records[index]
        final_decision = _resolve_final_decision(record)
        steps = record.get("steps_used")
        if not isinstance(steps, int):
            rounds = record.get("rounds")
            steps = len(rounds) if isinstance(rounds, list) else 0
        if final_decision == "ACCEPT" and steps <= k:
            accepted += 1
    return accepted / len(accept_indices)


def _hard_violation_rate(records: List[Mapping[str, Any]]) -> Optional[float]:
    attempted = [record for record in records if _resolve_final_decision(record) != "ABSTAIN"]
    if not attempted:
        return None
    violations = sum(1 for record in attempted if not bool(record.get("hard_pass", False)))
    return violations / len(attempted)


def _abstention_utility(records: List[Mapping[str, Any]]) -> Optional[float]:
    expected_actions = [_resolve_expected_action(record) for record in records]
    final_decisions = [_resolve_final_decision(record) for record in records]
    if not expected_actions:
        return None
    return decision_utility(expected_actions, final_decisions, reports.DEFAULT_UTILITY_COSTS)


def _boundary_precision_failure_rate(records: List[Mapping[str, Any]]) -> Optional[float]:
    boundary = [
        record
        for record in records
        if str(record.get("task_family") or "") == "boundary_precision"
    ]
    if not boundary:
        return None
    failures = sum(1 for record in boundary if _resolve_final_decision(record) != "ACCEPT")
    return failures / len(boundary)


def _resume_task_indices(records: List[Mapping[str, Any]]) -> List[int]:
    indices: List[int] = []
    for index, record in enumerate(records):
        if not bool(record.get("resume_expected")):
            continue
        rounds = record.get("rounds")
        if not isinstance(rounds, list):
            continue
        if any(bool(round_item.get("interrupt")) for round_item in rounds if isinstance(round_item, dict)):
            indices.append(index)
    return indices


def _resume_success_rate(records: List[Mapping[str, Any]]) -> Optional[float]:
    indices = _resume_task_indices(records)
    if not indices:
        return None
    successes = sum(1 for index in indices if bool(records[index].get("resume_success")))
    return successes / len(indices)


def _avg_extra_steps_after_interrupt(records: List[Mapping[str, Any]]) -> Optional[float]:
    indices = _resume_task_indices(records)
    values: List[float] = []
    for index in indices:
        value = records[index].get("extra_steps_after_interrupt")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _bootstrap_ci(
    records: List[Mapping[str, Any]],
    *,
    metric_fn,
    metric_mean: Optional[float],
    n_bootstrap: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    if metric_mean is None:
        return None
    if not records or n_bootstrap <= 1:
        return {
            "mean": float(metric_mean),
            "ci_low": float(metric_mean),
            "ci_high": float(metric_mean),
            "n_bootstrap": 0,
        }
    rng = random.Random(seed)
    n = len(records)
    samples: List[float] = []
    for _ in range(n_bootstrap):
        draw = [records[rng.randrange(n)] for _ in range(n)]
        value = metric_fn(draw)
        if value is None:
            continue
        samples.append(float(value))
    if not samples:
        return {
            "mean": float(metric_mean),
            "ci_low": float(metric_mean),
            "ci_high": float(metric_mean),
            "n_bootstrap": 0,
        }
    samples.sort()
    low_index = int(0.025 * (len(samples) - 1))
    high_index = int(0.975 * (len(samples) - 1))
    return {
        "mean": float(metric_mean),
        "ci_low": float(samples[low_index]),
        "ci_high": float(samples[high_index]),
        "n_bootstrap": len(samples),
    }


def _metrics_with_bootstrap(
    *,
    records: List[Mapping[str, Any]],
    summary: Mapping[str, Any],
    seed: int,
    n_bootstrap: int,
) -> Dict[str, Any]:
    metrics = _summary_metrics(summary)
    ci: Dict[str, Any] = {}

    ci["pass_at_1"] = _bootstrap_ci(
        records,
        metric_fn=lambda rows: _pass_at_k(rows, 1),
        metric_mean=(None if metrics.get("pass_at_1") is None else float(metrics["pass_at_1"])),
        n_bootstrap=n_bootstrap,
        seed=seed + 11,
    )
    ci["pass_at_3"] = _bootstrap_ci(
        records,
        metric_fn=lambda rows: _pass_at_k(rows, 3),
        metric_mean=(None if metrics.get("pass_at_3") is None else float(metrics["pass_at_3"])),
        n_bootstrap=n_bootstrap,
        seed=seed + 13,
    )
    ci["hard_violation_rate"] = _bootstrap_ci(
        records,
        metric_fn=_hard_violation_rate,
        metric_mean=(
            None
            if metrics.get("hard_violation_rate") is None
            else float(metrics["hard_violation_rate"])
        ),
        n_bootstrap=n_bootstrap,
        seed=seed + 17,
    )
    ci["abstention_utility"] = _bootstrap_ci(
        records,
        metric_fn=_abstention_utility,
        metric_mean=(
            None
            if metrics.get("abstention_utility") is None
            else float(metrics["abstention_utility"])
        ),
        n_bootstrap=n_bootstrap,
        seed=seed + 19,
    )
    ci["boundary_precision_failure_rate"] = _bootstrap_ci(
        records,
        metric_fn=_boundary_precision_failure_rate,
        metric_mean=(
            None
            if metrics.get("boundary_precision_failure_rate") is None
            else float(metrics["boundary_precision_failure_rate"])
        ),
        n_bootstrap=n_bootstrap,
        seed=seed + 23,
    )
    ci["resume_success_rate"] = _bootstrap_ci(
        records,
        metric_fn=_resume_success_rate,
        metric_mean=(
            None
            if metrics.get("resume_success_rate") is None
            else float(metrics["resume_success_rate"])
        ),
        n_bootstrap=n_bootstrap,
        seed=seed + 29,
    )
    ci["avg_extra_steps_after_interrupt"] = _bootstrap_ci(
        records,
        metric_fn=_avg_extra_steps_after_interrupt,
        metric_mean=(
            None
            if metrics.get("avg_extra_steps_after_interrupt") is None
            else float(metrics["avg_extra_steps_after_interrupt"])
        ),
        n_bootstrap=n_bootstrap,
        seed=seed + 31,
    )

    pass_at_steps = summary.get("pass_at_steps")
    pass_curve_ci: List[Dict[str, Any]] = []
    if isinstance(pass_at_steps, list):
        for point in pass_at_steps:
            if not isinstance(point, dict):
                continue
            step_budget = point.get("step_budget")
            if not isinstance(step_budget, int):
                continue
            mean_value = point.get("pass_rate")
            if mean_value is None:
                continue
            ci_value = _bootstrap_ci(
                records,
                metric_fn=lambda rows, budget=step_budget: _pass_at_k(rows, budget),
                metric_mean=float(mean_value),
                n_bootstrap=n_bootstrap,
                seed=seed + 100 + step_budget,
            )
            if ci_value is None:
                continue
            pass_curve_ci.append({"step_budget": step_budget, **ci_value})
    ci["pass_at_steps"] = pass_curve_ci

    metrics["bootstrap_ci"] = ci
    return metrics


def run_benchmark_sweep(
    *,
    benchmark_dir: Path,
    split: str,
    baselines_path: Path,
    out_dir: Path,
    seed: int = 7,
    limit: Optional[int] = None,
    allow_external: bool = False,
    cache_dir: Optional[Path] = None,
    replay_cache: Optional[Path] = None,
    n_bootstrap: int = 400,
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
    skipped_rows: List[Dict[str, Any]] = []
    by_family: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for entry in matrix:
        run_tasks = tasks
        if entry.protocol is not None:
            run_tasks = [task for task in tasks if task.protocol == entry.protocol]
        run_dir = out_dir / entry.name
        baseline_cache_dir = cache_dir / entry.name if cache_dir is not None else None
        baseline_replay_cache = (
            replay_cache / entry.name if replay_cache is not None else None
        )
        try:
            runner = TaskRunner(
                entry.model,
                seed=seed,
                adapter_kwargs=entry.adapter_kwargs,
                allow_external=allow_external,
                cache_dir=baseline_cache_dir,
                replay_cache=baseline_replay_cache,
            )
            runner.run_tasks(
                run_tasks,
                run_dir=run_dir,
                suite=f"{release.benchmark_id}_{split}",
                protocol=entry.protocol or "mixed",
                spec_loader=release.spec_loader,
            )
        except Exception as exc:
            if entry.optional:
                skipped_rows.append(
                    {
                        "name": entry.name,
                        "model": entry.model,
                        "track": entry.track,
                        "protocol": entry.protocol,
                        "optional": True,
                        "reason": str(exc),
                    }
                )
                continue
            raise

        records = reports.load_trace(run_dir)
        summary = reports.summarise(records)
        report_path = reports.write_report(run_dir, records=records, summary=summary)

        metrics = _metrics_with_bootstrap(
            records=records,
            summary=summary,
            seed=seed,
            n_bootstrap=n_bootstrap,
        )
        row = {
            "name": entry.name,
            "model": entry.model,
            "track": entry.track,
            "protocol": entry.protocol,
            "optional": entry.optional,
            "adapter_kwargs": entry.adapter_kwargs,
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

    track_breakdown: Dict[str, Dict[str, Any]] = {}
    for track in TRACKS:
        track_rows = [row for row in baseline_rows if row.get("track") == track]
        track_by_family: Dict[str, Dict[str, Any]] = {}
        for family, family_payload in sorted(by_family.items()):
            if not isinstance(family_payload, dict):
                continue
            selected = {
                baseline: metrics
                for baseline, metrics in family_payload.items()
                if any(row.get("name") == baseline for row in track_rows)
            }
            if selected:
                track_by_family[family] = selected

        track_breakdown[track] = {
            "n_baselines": len(track_rows),
            "baseline_order": [
                entry.name
                for entry in matrix
                if entry.track == track and any(row.get("name") == entry.name for row in track_rows)
            ],
            "baselines": track_rows,
            "by_spec_family": track_by_family,
        }

    aggregate = {
        "benchmark_id": release.benchmark_id,
        "split": split,
        "seed": seed,
        "limit": limit,
        "n_baselines": len(baseline_rows),
        "n_skipped_baselines": len(skipped_rows),
        "baselines": baseline_rows,
        "all_baselines": baseline_rows,
        "skipped_baselines": skipped_rows,
        "by_track": track_breakdown,
        "by_spec_family": {family: by_family[family] for family in sorted(by_family)},
        "baseline_order": [entry.name for entry in matrix if any(row.get("name") == entry.name for row in baseline_rows)],
    }
    jsonio.write_json(out_dir / "aggregate.json", aggregate)
    return aggregate
