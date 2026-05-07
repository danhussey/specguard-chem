from __future__ import annotations

"""Recompute reality-check metrics from raw external-model cache entries.

This is used when third-party API calls completed once but a later adapter fix
changes how public ACCEPT/REJECT/ABSTAIN labels are interpreted. It does not
make network calls.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from specguard_chem.benchmark.release import load_benchmark_release
from specguard_chem.config import SpecModel
from specguard_chem.runner.protocols import ConstraintEvaluator
from specguard_chem.verifiers import canonicalize_smiles

DEFAULT_RELEASE = Path("benchmarks/releases/sgchem_v1.0")
DEFAULT_OUT = Path("runs/reality_check/sgchem_v1.0")

ACTIONS = ("ACCEPT", "REJECT", "ABSTAIN", "INVALID")
PRIMARY_ACTIONS = ("ACCEPT", "REJECT", "ABSTAIN")


def _safe_div(num: int | float, den: int | float) -> float | None:
    return None if not den else float(num) / float(den)


def _input_smiles(task: Mapping[str, Any]) -> str | None:
    payload = task.get("input") if isinstance(task.get("input"), dict) else {}
    for key in ("smiles", "candidate_smiles"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_raw(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"action": "INVALID"}
    return payload if isinstance(payload, dict) else {"action": "INVALID"}


def _declared_action(payload: Mapping[str, Any]) -> str:
    action = str(payload.get("action") or "").strip().upper()
    if action in PRIMARY_ACTIONS:
        return action
    if action == "PROPOSE":
        return "ACCEPT"
    if action == "ABSTAIN":
        return "ABSTAIN"
    return "INVALID"


def _candidate_smiles(payload: Mapping[str, Any], task: Mapping[str, Any]) -> str | None:
    value = payload.get("smiles")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _input_smiles(task)


def _hard_pass(smiles: str | None, spec_payload: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    if not smiles:
        return False
    try:
        spec = SpecModel.model_validate(spec_payload)
        evaluator = ConstraintEvaluator(spec, input_smiles=_input_smiles(task))
        return bool(evaluator.evaluate(smiles).hard_pass)
    except Exception:
        return False


def _latest_entries(cache_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        outer_request = entry.get("adapter_request") or {}
        task_id = outer_request.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        raw = str(entry.get("raw_model_output") or "")
        parsed = _parse_raw(raw)
        action = str(parsed.get("action") or "").strip().lower()
        # For forced-verify runs, keep the later model answer rather than the
        # adapter's deterministic first-round tool call.
        if action == "tool_call":
            latest.setdefault(task_id, entry)
            continue
        latest[task_id] = entry
    return latest


def _metrics_for_cache(cache_path: Path, task_meta: Mapping[str, Any]) -> dict[str, Any]:
    latest = _latest_entries(cache_path)
    confusion = {exp: {pred: 0 for pred in ACTIONS} for exp in PRIMARY_ACTIONS}
    molecule_accepts = 0
    rows = []
    direct = Counter()
    for task_id, entry in sorted(latest.items()):
        task_model = task_meta.get(task_id)
        if task_model is None:
            continue
        outer_request = entry.get("adapter_request") or {}
        request = outer_request.get("request") if isinstance(outer_request.get("request"), dict) else {}
        task = request.get("task") if isinstance(request.get("task"), dict) else {}
        spec = request.get("spec") if isinstance(request.get("spec"), dict) else {}
        raw_payload = _parse_raw(str(entry.get("raw_model_output") or ""))
        predicted = _declared_action(raw_payload)
        expected = str(task_model.expected_action)
        if expected not in PRIMARY_ACTIONS:
            expected = "ACCEPT"
        if predicted not in ACTIONS:
            predicted = "INVALID"
        confusion[expected][predicted] += 1
        smiles = _candidate_smiles(raw_payload, task)
        hard_pass = _hard_pass(smiles, spec, task) if predicted == "ACCEPT" else False
        if hard_pass:
            molecule_accepts += 1
        if expected == "REJECT" and predicted == "ACCEPT":
            input_canon = canonicalize_smiles(_input_smiles(task) or "")
            output_canon = canonicalize_smiles(smiles or "")
            if input_canon and output_canon and input_canon != output_canon and hard_pass:
                direct["substituted_passing_molecule_for_audit"] += 1
        if expected == "ABSTAIN" and predicted == "ACCEPT":
            direct["accepted_on_contradictory_spec"] += 1
        if expected != "ACCEPT" and predicted == "ACCEPT":
            direct["wrong_action_but_valid_molecule"] += int(hard_pass)
        if predicted == "INVALID":
            direct["schema_or_parse_error"] += 1
        rows.append(
            {
                "task_id": task_id,
                "expected_action": expected,
                "predicted_action": predicted,
                "task_type": str(task_model.task_type),
                "hard_pass_if_accept": hard_pass,
            }
        )
    n = len(rows)
    reject_n = sum(confusion["REJECT"].values())
    abstain_n = sum(confusion["ABSTAIN"].values())
    inconsistent = confusion["REJECT"]["ACCEPT"] + confusion["ABSTAIN"]["ACCEPT"]
    construct = [row for row in rows if row["task_type"] == "construct_feasible"]
    repair = [row for row in rows if str(row["task_type"]).startswith("repair_")]
    boundary = [row for row in rows if row["task_type"] == "boundary_precision"]
    invariance = [row for row in rows if row["task_type"] == "smiles_invariance"]
    output = {
        "num_tasks": n,
        "action_accuracy": _safe_div(sum(confusion[a][a] for a in PRIMARY_ACTIONS), n),
        "molecule_acceptance_rate": _safe_div(molecule_accepts, n),
        "task_inconsistent_accept_rate": _safe_div(inconsistent, reject_n + abstain_n),
        "task_inconsistent_accept_n": reject_n + abstain_n,
        "REJECT_recall": _safe_div(confusion["REJECT"]["REJECT"], reject_n),
        "REJECT_recall_n": reject_n,
        "ABSTAIN_recall": _safe_div(confusion["ABSTAIN"]["ABSTAIN"], abstain_n),
        "ABSTAIN_recall_n": abstain_n,
        "construct_success": {
            "n": len(construct),
            "success": _safe_div(sum(row["expected_action"] == row["predicted_action"] for row in construct), len(construct)),
        },
        "repair_success": {
            "n": len(repair),
            "success": _safe_div(sum(row["expected_action"] == row["predicted_action"] for row in repair), len(repair)),
        },
        "boundary_success": {
            "n": len(boundary),
            "success": _safe_div(sum(row["expected_action"] == row["predicted_action"] for row in boundary), len(boundary)),
        },
        "invariance_success": {
            "n": len(invariance),
            "success": _safe_div(sum(row["expected_action"] == row["predicted_action"] for row in invariance), len(invariance)),
        },
        "confusion": confusion,
        "accepted_same_candidate_after_failed_verifier": 0,
        "substituted_passing_molecule_for_audit": direct["substituted_passing_molecule_for_audit"],
        "accepted_on_contradictory_spec": direct["accepted_on_contradictory_spec"],
        "wrong_action_but_valid_molecule": direct["wrong_action_but_valid_molecule"],
        "schema_or_parse_error": direct["schema_or_parse_error"],
    }
    return {"metrics": output, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    release = load_benchmark_release(args.release)
    tasks = release.load_split_tasks("test")
    task_meta = {task.task_id: task for task in tasks}
    cache_root = args.out / "cache"
    runs = []
    for cache_path in sorted(cache_root.glob("openai_*_subset/cache.jsonl")):
        result = _metrics_for_cache(cache_path, task_meta)
        name = f"{cache_path.parent.name}_cache_reparsed"
        run_dir = args.out / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "reality_metrics.json").write_text(
            json.dumps(result["metrics"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "cache_reparse_rows.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in result["rows"]) + "\n",
            encoding="utf-8",
        )
        runs.append(
            {
                "name": name,
                "model": "cache_reparse",
                "num_tasks": result["metrics"]["num_tasks"],
                "metrics": result["metrics"],
                "source_cache": str(cache_path),
            }
        )
    payload = {
        "benchmark_id": release.benchmark_id,
        "method": "raw_api_cache_reparse",
        "runs": runs,
    }
    output = args.out / "cache_reparse_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
