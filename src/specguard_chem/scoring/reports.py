from __future__ import annotations

"""Helpers for loading and summarising run artefacts."""

from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from ..utils import jsonio
from .metrics import hard_violation_rate


def load_trace(run_path: Path) -> List[Dict[str, Any]]:
    trace_path = run_path / "trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found at {trace_path}")
    return jsonio.read_jsonl(trace_path)


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "num_tasks": 0,
            "avg_spec_score": 0.0,
            "hard_violation_rate": 0.0,
            "abstain_rate": 0.0,
        }
    spec_scores = [record.get("spec_score", 0.0) for record in records]
    hard_flags = [bool(record.get("hard_pass", False)) for record in records]
    abstains = [bool(record.get("abstained", False)) for record in records]
    return {
        "num_tasks": len(records),
        "avg_spec_score": mean(spec_scores),
        "hard_violation_rate": hard_violation_rate(hard_flags),
        "abstain_rate": sum(abstains) / len(records),
    }


def write_report(run_path: Path, destination: Path | None = None) -> Path:
    records = load_trace(run_path)
    summary = summarise(records)
    target = destination or (run_path / "report.json")
    jsonio.write_json(target, {"summary": summary, "records": records})
    return target
