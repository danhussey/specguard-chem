from __future__ import annotations

"""Summarize live/replayed external-baseline runs for paper triage."""

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from specguard_chem.scoring import reports
from specguard_chem.utils import jsonio

ACTIONS = ("ACCEPT", "REJECT", "ABSTAIN")
PREDICTED = ("ACCEPT", "REJECT", "ABSTAIN", "INVALID")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


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


def _write_md(path: Path, rows: Sequence[Mapping[str, Any]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_div(numer: float, denom: float) -> float | None:
    return None if denom == 0 else numer / denom


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
    if value in PREDICTED:
        return value
    decision = str(record.get("decision") or "").lower()
    if decision == "accept":
        return "ACCEPT"
    if decision == "reject":
        return "REJECT"
    if decision == "abstain":
        return "ABSTAIN"
    return "INVALID"


def _record_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    n = len(records)
    expected = [_expected(record) for record in records]
    predicted = [_predicted(record) for record in records]
    recalls = {}
    for action in ACTIONS:
        denom = sum(1 for item in expected if item == action)
        recalls[action] = _safe_div(
            sum(1 for exp, pred in zip(expected, predicted) if exp == action and pred == action),
            denom,
        )
    unsafe_denom = sum(1 for value in expected if value in {"REJECT", "ABSTAIN"})
    attempted = [index for index, value in enumerate(predicted) if value != "ABSTAIN"]
    verify_calls = [float(record.get("verify_calls_used") or 0) for record in records]
    return {
        "n_tasks": n,
        "action_accuracy": _safe_div(
            sum(1 for exp, pred in zip(expected, predicted) if exp == pred),
            n,
        ),
        "molecule_acceptance_rate": _safe_div(sum(1 for value in predicted if value == "ACCEPT"), n),
        "task_inconsistent_accept_rate": _safe_div(
            sum(1 for exp, pred in zip(expected, predicted) if exp in {"REJECT", "ABSTAIN"} and pred == "ACCEPT"),
            unsafe_denom,
        ),
        "accept_recall": recalls["ACCEPT"],
        "reject_recall": recalls["REJECT"],
        "abstain_recall": recalls["ABSTAIN"],
        "invalid_rate": _safe_div(sum(1 for value in predicted if value == "INVALID"), n),
        "task_schema_error_rate": _safe_div(sum(1 for record in records if record.get("schema_error")), n),
        "hard_violation_rate": _safe_div(
            sum(1 for index in attempted if not bool(records[index].get("hard_pass"))),
            len(attempted),
        ),
        "mean_verify_calls": mean(verify_calls) if verify_calls else 0.0,
    }


def _cache_summary(cache_root: Path | None, name: str) -> dict[str, Any]:
    if cache_root is None:
        return {}
    cache_path = cache_root / name / "cache.jsonl"
    if not cache_path.exists():
        return {}
    rows = jsonio.read_jsonl(cache_path)
    metadata = {}
    interface_errors = Counter()
    empty_raw = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not metadata and isinstance(row.get("model_metadata"), dict):
            metadata = dict(row["model_metadata"])
        raw = row.get("raw_model_output")
        if not isinstance(raw, str) or not raw.strip():
            empty_raw += 1
        parsed = row.get("parsed_adapter_response")
        if isinstance(parsed, dict) and str(parsed.get("action")) == "interface_error":
            error_type = str(parsed.get("interface_error_type") or "interface_error")
            interface_errors[error_type] += 1
    total = len(rows)
    return {
        "provider": metadata.get("provider"),
        "model_id": metadata.get("model_id"),
        "interface_tier": metadata.get("interface_tier"),
        "schema_hash": metadata.get("schema_hash"),
        "provider_feature": metadata.get("provider_feature"),
        "cache_steps": total,
        "interface_error_steps": sum(interface_errors.values()),
        "interface_error_step_rate": _safe_div(sum(interface_errors.values()), total),
        "empty_raw_steps": empty_raw,
        "empty_raw_step_rate": _safe_div(empty_raw, total),
        "interface_error_types": dict(interface_errors),
    }


def collect_rows(aggregate_path: Path, *, cache_root: Path | None) -> list[dict[str, Any]]:
    aggregate = jsonio.read_json(aggregate_path)
    if not isinstance(aggregate, dict):
        raise ValueError(f"aggregate is not a JSON object: {aggregate_path}")
    rows: list[dict[str, Any]] = []
    for baseline in aggregate.get("baselines", []):
        if not isinstance(baseline, dict):
            continue
        name = str(baseline.get("name") or "")
        run_dir = aggregate_path.parent / str(baseline.get("run_dir") or name)
        records = reports.load_trace(run_dir)
        metrics = _record_metrics(records)
        cache = _cache_summary(cache_root, name)
        adapter_kwargs = baseline.get("adapter_kwargs") if isinstance(baseline.get("adapter_kwargs"), dict) else {}
        rows.append(
            {
                "system": name,
                "provider": cache.get("provider"),
                "model_id": cache.get("model_id") or adapter_kwargs.get("model"),
                "interface_tier": cache.get("interface_tier") or adapter_kwargs.get("interface_tier"),
                "provider_feature": cache.get("provider_feature"),
                "schema_hash": cache.get("schema_hash"),
                **metrics,
                "cache_steps": cache.get("cache_steps"),
                "interface_error_steps": cache.get("interface_error_steps"),
                "interface_error_step_rate": cache.get("interface_error_step_rate"),
                "empty_raw_steps": cache.get("empty_raw_steps"),
                "empty_raw_step_rate": cache.get("empty_raw_step_rate"),
                "interface_error_types": json.dumps(cache.get("interface_error_types") or {}, sort_keys=True),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args(argv)

    rows = collect_rows(args.aggregate, cache_root=args.cache_root)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "external_baseline_metrics.csv", rows)
    _write_md(args.out / "external_baseline_metrics.md", rows, title="External Baseline Metrics")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate": str(args.aggregate),
        "cache_root": str(args.cache_root) if args.cache_root else None,
        "n_rows": len(rows),
        "total_cache_steps": sum(int(row.get("cache_steps") or 0) for row in rows),
        "total_interface_error_steps": sum(int(row.get("interface_error_steps") or 0) for row in rows),
        "rows": rows,
    }
    jsonio.write_json(args.out / "external_baseline_summary.json", summary)
    lines = [
        "# External Baseline Summary",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- aggregate: {args.aggregate}",
        f"- cache_root: {args.cache_root}",
        f"- rows: {summary['n_rows']}",
        f"- cache_steps: {summary['total_cache_steps']}",
        f"- interface_error_steps: {summary['total_interface_error_steps']}",
        "",
    ]
    (args.out / "external_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
