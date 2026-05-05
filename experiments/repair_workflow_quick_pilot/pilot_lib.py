from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pandas as pd

from specguard_chem.config import SpecModel
from specguard_chem.dataset.corpus import _scaffold_hash
from specguard_chem.models.local_mutation import _neighbors, _reaction_objects
from specguard_chem.runner.protocols import ConstraintEvaluator
from specguard_chem.utils.edit_distance import levenshtein
from specguard_chem.verifiers import canonicalize_smiles, is_valid_smiles, morgan_tanimoto

ROOT = Path(__file__).resolve().parents[2]
PILOT = Path(__file__).resolve().parent
TASKS = PILOT / "tasks.jsonl"
CORPUS = ROOT / "benchmarks" / "releases" / "sgchem_v1.0" / "corpus" / "corpus.parquet"
TARGET_COUNTS = {
    "repair_near_miss": 12,
    "repair_multi_violation": 12,
    "scaffold_or_similarity_preserving_repair": 8,
    "boundary_repair": 4,
    "construct_under_rule_card": 4,
}
FORBIDDEN_TERMS = (
    "potency",
    "target binding",
    "target-binding",
    "dosage",
    "therapeutic efficacy",
    "clinical utility",
    "patient",
    "disease",
)


def load_tasks(path: Path = TASKS) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_corpus() -> list[str]:
    df = pd.read_parquet(CORPUS)
    return sorted(
        str(value)
        for value in df["canonical_smiles"].dropna().tolist()
        if canonicalize_smiles(str(value))
    )


def spec_from_task(task: dict[str, Any]) -> SpecModel:
    return SpecModel.model_validate(task["public_rule_card"]["spec"])


def is_repair_task(task: dict[str, Any]) -> bool:
    return task["family"] != "construct_under_rule_card"


def failure_ids(task: dict[str, Any], smiles: str) -> list[str]:
    evaluator = ConstraintEvaluator(spec_from_task(task), input_smiles=task.get("starting_smiles"))
    result = evaluator.evaluate(smiles)
    return [outcome.constraint.id for outcome in result.hard_outcomes if not outcome.passed]


def evaluate_candidate(task: dict[str, Any], smiles: str | None) -> dict[str, Any]:
    if not smiles:
        return {
            "valid": False,
            "hard_pass": False,
            "success": False,
            "similarity": None,
            "similarity_ok": False,
            "scaffold_ok": False,
            "failure_reason": "abstained",
        }
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        return {
            "valid": False,
            "hard_pass": False,
            "success": False,
            "similarity": None,
            "similarity_ok": False,
            "scaffold_ok": False,
            "failure_reason": "invalid_smiles",
        }
    spec = spec_from_task(task)
    evaluator = ConstraintEvaluator(spec, input_smiles=task.get("starting_smiles"))
    result = evaluator.evaluate(canonical)
    hard_pass = bool(result.hard_pass)
    similarity_min = task.get("similarity_min")
    start = task.get("starting_smiles")
    similarity = morgan_tanimoto(start, canonical) if start else None
    similarity_ok = True
    if isinstance(similarity_min, (int, float)):
        similarity_ok = similarity is not None and similarity >= float(similarity_min)
    scaffold_ok = True
    if task.get("preserve_scaffold"):
        scaffold_ok = _scaffold_hash(canonical) == task.get("starting_scaffold_hash")
    success = hard_pass and similarity_ok and scaffold_ok
    if not hard_pass:
        reason = "hard_constraint_failure"
    elif not similarity_ok:
        reason = "similarity_failure"
    elif not scaffold_ok:
        reason = "scaffold_failure"
    elif is_repair_task(task) and canonical == canonicalize_smiles(str(start)):
        reason = "no_change"
        success = False
    else:
        reason = "success"
    return {
        "valid": True,
        "canonical_smiles": canonical,
        "hard_pass": hard_pass,
        "success": success,
        "similarity": similarity,
        "similarity_ok": similarity_ok,
        "scaffold_ok": scaffold_ok,
        "failure_reason": reason,
    }


def candidate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    successes = [row for row in rows if row["success"]]
    valid = [row for row in rows if row["valid"]]
    repair_rows = [row for row in rows if row["family"] != "construct_under_rule_card"]
    no_op_rows = [
        row
        for row in rows
        if row.get("canonical_smiles")
        and row.get("starting_canonical_smiles")
        and row.get("canonical_smiles") == row.get("starting_canonical_smiles")
    ]
    unrelated_valid = [
        row
        for row in rows
        if row["valid"]
        and row.get("similarity") is not None
        and row.get("similarity") < 0.25
    ]
    success_sims = [float(row["similarity"]) for row in successes if row.get("similarity") is not None]
    edit_distances = [
        levenshtein(str(row["starting_canonical_smiles"]), str(row["canonical_smiles"]))
        for row in successes
        if row.get("starting_canonical_smiles") and row.get("canonical_smiles")
    ]
    return {
        "n": n,
        "success@1": _rate(sum(1 for row in rows if row["success"]), n),
        "hard_pass_rate": _rate(sum(1 for row in rows if row["hard_pass"]), n),
        "similarity_preservation_rate": _rate(
            sum(1 for row in repair_rows if row["similarity_ok"]),
            len(repair_rows),
        ),
        "scaffold_preservation_rate": _rate(
            sum(1 for row in rows if (not row.get("requires_scaffold")) or row["scaffold_ok"]),
            n,
        ),
        "valid_smiles_rate": _rate(len(valid), n),
        "no_op_rate": _rate(len(no_op_rows), max(1, len(repair_rows))),
        "unrelated_valid_molecule_rate": _rate(len(unrelated_valid), max(1, len(valid))),
        "median_similarity_on_successes": median(success_sims) if success_sims else None,
        "median_edit_distance_on_successes": median(edit_distances) if edit_distances else None,
    }


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def summarize_baseline(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = candidate_metrics(rows)
    failure_counts: dict[str, int] = {}
    for row in rows:
        failure_counts[row["failure_reason"]] = failure_counts.get(row["failure_reason"], 0) + 1
    return {"baseline": name, **metrics, "failure_counts": failure_counts}


def evaluate_candidate_list(task: dict[str, Any], candidates: list[str | None]) -> dict[str, Any]:
    first = evaluate_candidate(task, candidates[0] if candidates else None)
    first.update(
        {
            "task_id": task["task_id"],
            "family": task["family"],
            "starting_canonical_smiles": task.get("starting_canonical_smiles"),
            "requires_scaffold": bool(task.get("preserve_scaffold")),
        }
    )
    first["success@3"] = any(evaluate_candidate(task, candidate).get("success") for candidate in candidates[:3])
    return first


def nearest_passing_corpus(task: dict[str, Any], corpus: list[str], *, k: int = 3) -> list[str | None]:
    scored: list[tuple[float, str]] = []
    start = task.get("starting_smiles")
    for smiles in corpus:
        ev = evaluate_candidate(task, smiles)
        if not ev["hard_pass"] or not ev["similarity_ok"] or not ev["scaffold_ok"]:
            continue
        sim = morgan_tanimoto(start, smiles) if start else 1.0
        scored.append((float(sim or 0.0), smiles))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [smiles for _, smiles in scored[:k]] or [None]


def mutation_candidates(task: dict[str, Any], *, depth: int, k: int = 3) -> list[str | None]:
    start = task.get("starting_smiles") or task.get("hidden_witness_smiles")
    if not start:
        return [None]
    reactions = _reaction_objects()
    frontier = {canonicalize_smiles(str(start)) or str(start)}
    seen = set(frontier)
    candidates: set[str] = set()
    for _ in range(depth):
        next_frontier: set[str] = set()
        for smiles in sorted(frontier):
            for neighbor in _neighbors(smiles, reactions):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_frontier.add(neighbor)
                candidates.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    passing = []
    for candidate in candidates:
        ev = evaluate_candidate(task, candidate)
        if ev["success"]:
            sim = ev["similarity"] if ev["similarity"] is not None else 1.0
            passing.append((float(sim), candidate))
    passing.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, candidate in passing[:k]] or sorted(candidates)[:k] or [None]


def render_md_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "baseline",
        "n",
        "success@1",
        "hard_pass_rate",
        "similarity_preservation_rate",
        "scaffold_preservation_rate",
        "valid_smiles_rate",
        "no_op_rate",
        "unrelated_valid_molecule_rate",
        "median_similarity_on_successes",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def micro_range_matches(text: str, *, threshold: float = 1.0) -> list[str]:
    matches = []
    pattern = re.compile(r"between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", re.I)
    for match in pattern.finditer(text):
        lo = float(match.group(1))
        hi = float(match.group(2))
        if abs(hi - lo) < threshold:
            matches.append(match.group(0))
    return matches


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
