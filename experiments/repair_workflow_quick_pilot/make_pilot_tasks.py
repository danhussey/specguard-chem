from __future__ import annotations

"""Build a small disposable constrained-repair pilot task set.

This intentionally does not use the sgchem release compiler and does not write
under benchmarks/releases. It reuses local specs, corpus records, and verifier
logic to answer one question quickly: does analogue repair with public
similarity/scaffold constraints produce a useful signal?
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from specguard_chem.config import SpecModel, load_spec
from specguard_chem.dataset.corpus import _scaffold_hash
from specguard_chem.models.local_mutation import _neighbors, _reaction_objects
from specguard_chem.runner.protocols import ConstraintEvaluator, EvaluationResult
from specguard_chem.verifiers import canonicalize_smiles, morgan_tanimoto

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TASKS = OUT / "tasks.jsonl"
CORPUS = ROOT / "benchmarks" / "releases" / "sgchem_v1.0" / "corpus" / "corpus.parquet"

TARGETS = {
    "repair_near_miss": 12,
    "repair_multi_violation": 12,
    "scaffold_or_similarity_preserving_repair": 8,
    "boundary_repair": 4,
    "construct_under_rule_card": 4,
}

SPEC_ORDER = [
    "spec_v2_ro5_balanced_test",
    "spec_v2_alert_strict_dev",
    "spec_v2_fragment_ultra_dev",
    "spec_v2_polar_bias_train",
    "spec_v2_cns_like_test",
    "spec_v2_low_rotor_train",
    "spec_v2_sa_strict_test",
    "spec_v2_amide_friendly_dev",
    "spec_v2_fragment_tight_train",
]


def _add_similarity(spec: SpecModel, threshold: float) -> SpecModel:
    data = spec.model_dump(mode="json")
    data["constraints"] = [
        item
        for item in data["constraints"]
        if item["id"] != "pilot_similarity_to_start"
    ]
    data["constraints"].append(
        {
            "id": "pilot_similarity_to_start",
            "type": "hard",
            "check": "similarity_min_to_input",
            "params": {
                "min": threshold,
                "fp": "morgan",
                "radius": 2,
                "nBits": 2048,
            },
            "weight": 1.0,
        }
    )
    return SpecModel.model_validate(data)


def _add_secondary_property_repair(spec: SpecModel, start: str, witness: str) -> SpecModel | None:
    start_props = ConstraintEvaluator(spec).evaluate(start).properties
    witness_props = ConstraintEvaluator(spec).evaluate(witness).properties
    for prop, width in (("MW", 30.0), ("TPSA", 20.0), ("logP", 1.5)):
        start_value = start_props.get(prop)
        witness_value = witness_props.get(prop)
        if start_value is None or witness_value is None:
            continue
        lower = float(witness_value) - width / 2.0
        upper = float(witness_value) + width / 2.0
        if prop in {"MW", "TPSA"}:
            lower = max(0.0, lower)
        if lower <= float(start_value) <= upper:
            continue
        data = spec.model_dump(mode="json")
        data["constraints"] = [
            item
            for item in data["constraints"]
            if item["id"] != "pilot_secondary_property_repair"
        ]
        data["constraints"].append(
            {
                "id": "pilot_secondary_property_repair",
                "type": "hard",
                "check": "property_bounds",
                "params": {
                    "mode": "all",
                    "bounds": {prop: {"min": round(lower, 3), "max": round(upper, 3)}},
                },
                "weight": 1.0,
            }
        )
        candidate = SpecModel.model_validate(data)
        if ConstraintEvaluator(candidate, input_smiles=start).evaluate(witness).hard_pass:
            return candidate
    return None


def _hard_fail_ids(result: EvaluationResult) -> list[str]:
    return [outcome.constraint.id for outcome in result.hard_outcomes if not outcome.passed]


def _soft_constraint_ids(spec: SpecModel) -> list[str]:
    return [constraint.id for constraint in spec.constraints if constraint.type == "soft"]


def _constraint_lines(spec: SpecModel) -> list[str]:
    lines: list[str] = []
    for constraint in spec.constraints:
        if constraint.type != "hard":
            continue
        params = constraint.params
        if constraint.check == "property_bounds":
            bounds = params.get("bounds", {})
            pieces = [
                f"{name} between {float(raw['min']):.1f} and {float(raw['max']):.1f}"
                for name, raw in sorted(bounds.items())
            ]
            lines.append(f"{constraint.id}: {', '.join(pieces)}.")
        elif constraint.check == "alert_set_absent":
            lines.append(
                f"{constraint.id}: no hits from alert set {params.get('alert_set')}."
            )
        elif constraint.check == "substructure_present":
            lines.append(
                f"{constraint.id}: require substructure {params.get('smarts_id')}."
            )
        elif constraint.check == "substructure_absent":
            lines.append(
                f"{constraint.id}: forbid substructure {params.get('smarts_id')}."
            )
        elif constraint.check == "sa_proxy_max":
            lines.append(f"{constraint.id}: SA proxy at most {float(params.get('max')):.1f}.")
        elif constraint.check == "similarity_min_to_input":
            lines.append(
                f"{constraint.id}: Morgan Tanimoto similarity to the starting molecule at least {float(params.get('min')):.2f}."
            )
        else:
            lines.append(f"{constraint.id}: {constraint.check}.")
    return lines


def _render_prompt(
    *,
    family: str,
    spec: SpecModel,
    starting_smiles: str | None,
    similarity_min: float | None,
    preserve_scaffold: bool,
) -> str:
    if family == "construct_under_rule_card":
        task = "Construct a molecule satisfying the medicinal-chemistry rule card."
        actions = "- CONSTRUCT with a candidate SMILES if all hard constraints can be satisfied.\n- ABSTAIN only if the rule card appears internally contradictory."
        schema = '{"action":"CONSTRUCT|ABSTAIN","smiles":"...","rationale":"..."}'
    else:
        task = "Repair the starting molecule while preserving analogue similarity."
        actions = "- REPAIR with a candidate SMILES satisfying all hard constraints.\n- ABSTAIN only if no compliant analogue can be found under the stated preservation constraints."
        schema = '{"action":"REPAIR|ABSTAIN","repaired_smiles":"...","rationale":"..."}'
    lines = [
        f"Task: {task}",
        "",
        "You are given a machine-checkable medicinal-chemistry rule card.",
    ]
    if starting_smiles:
        lines += ["", "Starting SMILES:", starting_smiles]
    lines += ["", "Allowed actions:", actions, "", "Hard constraints:"]
    lines += [f"{idx}. {line}" for idx, line in enumerate(_constraint_lines(spec), start=1)]
    if preserve_scaffold:
        lines += [
            "",
            "Additional preservation rule:",
            "- Preserve the Bemis-Murcko scaffold of the starting molecule.",
        ]
    if similarity_min is not None:
        lines += [
            "",
            "Analogue preservation:",
            f"- Candidate Morgan Tanimoto similarity to the starting molecule must be at least {similarity_min:.2f}.",
        ]
    lines += [
        "",
        "Output schema:",
        schema,
        "",
        "Do not make claims about biological activity, clinical use, dosing, or therapeutic value.",
    ]
    return "\n".join(lines)


def _task(
    *,
    index: int,
    family: str,
    spec: SpecModel,
    base_spec_id: str,
    start: str | None,
    witness: str,
    fail_ids: list[str],
    similarity_min: float | None,
    preserve_scaffold: bool = False,
    boundary_property: str | None = None,
    witness_source: str = "local_mutation_neighbor_outside_release_corpus",
) -> dict[str, Any]:
    sim = morgan_tanimoto(start, witness) if start else None
    return {
        "task_id": f"pilot_{index:03d}_{family}",
        "family": family,
        "spec_id": base_spec_id,
        "starting_smiles": start,
        "starting_canonical_smiles": canonicalize_smiles(start) if start else None,
        "starting_scaffold_hash": _scaffold_hash(start) if start else None,
        "similarity_min": similarity_min,
        "preserve_scaffold": preserve_scaffold,
        "boundary_property": boundary_property,
        "failure_constraint_ids": fail_ids,
        "public_rule_card": {
            "spec": spec.model_dump(mode="json"),
            "hard_constraints": _constraint_lines(spec),
            "soft_constraint_ids": _soft_constraint_ids(spec),
            "similarity_constraint_public": similarity_min is not None,
            "scaffold_constraint_public": preserve_scaffold,
        },
        "rendered_prompt": _render_prompt(
            family=family,
            spec=spec,
            starting_smiles=start,
            similarity_min=similarity_min,
            preserve_scaffold=preserve_scaffold,
        ),
        "output": "repaired molecule or ABSTAIN"
        if family != "construct_under_rule_card"
        else "constructed molecule or ABSTAIN",
        "hidden_witness_smiles": witness,
        "hidden_witness_canonical_smiles": canonicalize_smiles(witness),
        "hidden_witness_similarity_to_start": sim,
        "hidden_witness_scaffold_hash": _scaffold_hash(witness),
        "witness_source": witness_source,
    }


def _boundary_property(result: EvaluationResult) -> str | None:
    best_prop = None
    best_abs_margin = float("inf")
    for name, margin in result.property_margins.items():
        if margin >= 0:
            continue
        abs_margin = abs(float(margin))
        if abs_margin < best_abs_margin:
            best_abs_margin = abs_margin
            best_prop = name
    return best_prop


def _collect_candidates(
    *,
    family: str,
    specs: dict[str, SpecModel],
    corpus: set[str],
    smiles: list[str],
) -> list[dict[str, Any]]:
    reactions = _reaction_objects()
    rows: list[dict[str, Any]] = []
    for spec_id in SPEC_ORDER:
        base = specs[spec_id]
        similarity_min = 0.55 if family == "scaffold_or_similarity_preserving_repair" else 0.45
        effective = _add_similarity(base, similarity_min)
        for start in smiles:
            base_eval = ConstraintEvaluator(base).evaluate(start)
            fail_ids = _hard_fail_ids(base_eval)
            if family == "repair_near_miss" and len(fail_ids) != 1:
                continue
            if family == "repair_multi_violation" and len(set(fail_ids)) < 1:
                continue
            if family == "scaffold_or_similarity_preserving_repair" and len(fail_ids) < 1:
                continue
            if family == "boundary_repair":
                if len(fail_ids) != 1 or _boundary_property(base_eval) is None:
                    continue
            evaluator = ConstraintEvaluator(effective, input_smiles=start)
            for witness in _neighbors(start, reactions):
                if witness == start:
                    continue
                if witness in corpus and family != "repair_multi_violation":
                    continue
                sim = morgan_tanimoto(start, witness) or 0.0
                if sim < similarity_min:
                    continue
                if family == "scaffold_or_similarity_preserving_repair" and _scaffold_hash(start) != _scaffold_hash(witness):
                    continue
                if not evaluator.evaluate(witness).hard_pass:
                    continue
                row_spec = effective
                row_fail_ids = fail_ids
                witness_source = (
                    "local_mutation_neighbor_in_release_corpus"
                    if witness in corpus
                    else "local_mutation_neighbor_outside_release_corpus"
                )
                if family == "repair_multi_violation" and len(set(fail_ids)) < 2:
                    augmented = _add_secondary_property_repair(effective, start, witness)
                    if augmented is None:
                        continue
                    augmented_eval = ConstraintEvaluator(augmented, input_smiles=start).evaluate(start)
                    row_fail_ids = [
                        item
                        for item in _hard_fail_ids(augmented_eval)
                        if item != "pilot_similarity_to_start"
                    ]
                    if len(set(row_fail_ids)) < 2:
                        continue
                    row_spec = augmented
                    witness_source += "_with_pilot_secondary_property_repair"
                rows.append(
                    {
                        "spec_id": spec_id,
                        "spec": row_spec,
                        "start": start,
                        "witness": witness,
                        "fail_ids": row_fail_ids,
                        "similarity_min": similarity_min,
                        "preserve_scaffold": family == "scaffold_or_similarity_preserving_repair",
                        "boundary_property": _boundary_property(base_eval),
                        "witness_source": witness_source,
                        "score": (sim, spec_id, start, witness),
                    }
                )
                break
    rows.sort(key=lambda item: (-float(item["score"][0]), item["spec_id"], item["start"], item["witness"]))
    return rows


def _construct_candidates(specs: dict[str, SpecModel], smiles: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec_id in SPEC_ORDER:
        spec = specs[spec_id]
        evaluator = ConstraintEvaluator(spec)
        for candidate in smiles:
            if evaluator.evaluate(candidate).hard_pass:
                rows.append(
                    {
                        "spec_id": spec_id,
                        "spec": spec,
                        "start": None,
                        "witness": candidate,
                        "fail_ids": [],
                        "similarity_min": None,
                        "preserve_scaffold": False,
                        "boundary_property": None,
                    }
                )
                break
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus_df = pd.read_parquet(CORPUS)
    corpus_smiles = sorted(
        str(value)
        for value in corpus_df["canonical_smiles"].dropna().tolist()
        if canonicalize_smiles(str(value))
    )
    corpus_set = set(corpus_smiles)
    specs = {spec_id: load_spec(spec_id) for spec_id in SPEC_ORDER}
    used_starts: set[tuple[str, str]] = set()
    tasks: list[dict[str, Any]] = []

    for family, target in TARGETS.items():
        if family == "construct_under_rule_card":
            candidates = _construct_candidates(specs, corpus_smiles)
        else:
            candidates = _collect_candidates(
                family=family,
                specs=specs,
                corpus=corpus_set,
                smiles=corpus_smiles,
            )
        selected = 0
        for row in candidates:
            key = (family, str(row["start"]), row["spec_id"])
            if key in used_starts:
                continue
            used_starts.add(key)
            tasks.append(
                _task(
                    index=len(tasks) + 1,
                    family=family,
                    spec=row["spec"],
                    base_spec_id=row["spec_id"],
                    start=row["start"],
                    witness=row["witness"],
                    fail_ids=row["fail_ids"],
                    similarity_min=row["similarity_min"],
                    preserve_scaffold=bool(row["preserve_scaffold"]),
                    boundary_property=row["boundary_property"],
                    witness_source=row.get("witness_source", "release_corpus"),
                )
            )
            selected += 1
            if selected >= target:
                break
        if selected < target:
            raise RuntimeError(f"Only found {selected}/{target} tasks for {family}")

    TASKS.write_text(
        "\n".join(json.dumps(task, sort_keys=True) for task in tasks) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tasks": len(tasks), "counts": TARGETS, "path": str(TASKS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
