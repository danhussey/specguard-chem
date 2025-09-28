from __future__ import annotations

"""Helpers for loading and summarising run artefacts."""

from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from ..utils import jsonio
from .metrics import abstention_utility, hard_violation_rate
from .calibration import brier_score, expected_calibration_error

DEFAULT_COSTS = {
    "false_accept": 10.0,
    "false_reject": 2.0,
    "abstain": 1.0,
}


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
            "avg_rounds": 0.0,
            "avg_edit_distance": 0.0,
            "accept_rate": 0.0,
            "brier_score": None,
            "ece": None,
            "abstention_utility": 0.0,
            "avg_confidence": None,
        }

    spec_scores = [record.get("spec_score", 0.0) for record in records]
    hard_flags = [bool(record.get("hard_pass", False)) for record in records]
    decisions = [record.get("decision", "accept") for record in records]
    abstains = [decision == "abstain" for decision in decisions]
    rounds_counts = [len(record.get("rounds", [])) for record in records]
    edit_distances = [
        record.get("edit_distance")
        for record in records
        if record.get("edit_distance") is not None
    ]
    confidences = [
        record.get("final_confidence")
        for record in records
        if record.get("final_confidence") is not None
    ]

    calibration_pairs = [
        (
            int(bool(record.get("hard_pass", False))),
            record.get("final_confidence"),
        )
        for record in records
        if record.get("final_confidence") is not None
    ]
    if calibration_pairs:
        truths, probs = zip(*calibration_pairs)
        brier = brier_score(truths, probs)
        ece = expected_calibration_error(truths, probs)
    else:
        brier = None
        ece = None

    utility = abstention_utility(hard_flags, decisions, DEFAULT_COSTS)

    return {
        "num_tasks": len(records),
        "avg_spec_score": mean(spec_scores),
        "hard_violation_rate": hard_violation_rate(hard_flags),
        "abstain_rate": sum(abstains) / len(records),
        "avg_rounds": mean(rounds_counts),
        "avg_edit_distance": (mean(edit_distances) if edit_distances else 0.0),
        "accept_rate": decisions.count("accept") / len(records),
        "avg_confidence": (mean(confidences) if confidences else None),
        "brier_score": brier,
        "ece": ece,
        "abstention_utility": utility,
    }


def write_report(run_path: Path, destination: Path | None = None) -> Path:
    records = load_trace(run_path)
    summary = summarise(records)
    target = destination or (run_path / "report.json")
    jsonio.write_json(target, {"summary": summary, "records": records})
    return target
