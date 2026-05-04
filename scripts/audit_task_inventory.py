from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specguard_chem.verifiers import canonicalize_smiles


RELEASE_SPLITS = ("train", "dev", "test")
INVENTORY_COLUMNS = (
    "task_id",
    "split",
    "family",
    "protocol",
    "spec_id",
    "expected_action",
    "input_smiles",
    "has_witness",
    "has_proof",
    "num_hard_constraints",
    "num_soft_preferences",
    "budget",
    "corpus_source",
    "invariance_group_id",
    "boundary_case_type",
)


@dataclass(frozen=True)
class LoadedTask:
    split: str
    payload: dict[str, Any]
    hard_constraints: int
    soft_preferences: int
    corpus_source: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_specs(release_dir: Path) -> dict[str, dict[str, Any]]:
    specs_dir = release_dir / "specs"
    catalog_path = specs_dir / "spec_catalog.json"
    if catalog_path.exists():
        catalog = _read_json(catalog_path)
        spec_ids = [str(row["id"]) for row in catalog.get("specs", [])]
    else:
        spec_ids = sorted(path.stem for path in specs_dir.glob("*.json") if path.stem != "spec_catalog")
    return {spec_id: _read_json(specs_dir / f"{spec_id}.json") for spec_id in spec_ids}


def _effective_constraints(
    spec: dict[str, Any], task_constraints: dict[str, Any] | None
) -> list[dict[str, Any]]:
    constraints = [dict(item) for item in spec.get("constraints", [])]
    if not task_constraints:
        return constraints

    overrides = task_constraints.get("overrides") or {}
    additions = task_constraints.get("additions") or []

    merged: list[dict[str, Any]] = []
    for constraint in constraints:
        updated = dict(constraint)
        override = overrides.get(str(updated.get("id", "")))
        if isinstance(override, dict):
            for key in ("check", "type", "severity", "weight"):
                if override.get(key) is not None:
                    updated[key] = override[key]
            if isinstance(override.get("params"), dict) and override["params"]:
                params = dict(updated.get("params") or {})
                params.update(override["params"])
                updated["params"] = params
        merged.append(updated)

    for addition in additions:
        if isinstance(addition, dict):
            merged.append(dict(addition))
    return merged


def _constraint_counts(spec: dict[str, Any], task_constraints: dict[str, Any] | None) -> tuple[int, int]:
    constraints = _effective_constraints(spec, task_constraints)
    hard = sum(1 for item in constraints if item.get("type") == "hard")
    soft = sum(1 for item in constraints if item.get("type") == "soft")
    return hard, soft


def _budget_string(task: dict[str, Any]) -> str:
    budgets = task.get("budgets") or {}
    keys = (
        "max_steps",
        "max_proposals",
        "max_verify_calls",
        "max_total_verifier_calls",
        "max_edit_cost",
    )
    return ";".join(f"{key}={budgets.get(key)}" for key in keys if key in budgets)


def _boundary_case_type(evidence: dict[str, Any]) -> str:
    prop = evidence.get("boundary_property")
    side = evidence.get("boundary_side")
    distance = evidence.get("boundary_distance")
    if prop and side and distance is not None:
        return f"{prop}:{side}:distance={distance}"
    if prop and side:
        return f"{prop}:{side}"
    return ""


def _has_proof(evidence: dict[str, Any]) -> bool:
    return bool(evidence.get("contradiction_proof") or evidence.get("budget_infeasible_note"))


def _inventory_row(task: LoadedTask) -> dict[str, Any]:
    payload = task.payload
    evidence = payload.get("evidence") or {}
    input_payload = payload.get("input") or {}
    return {
        "task_id": payload.get("task_id", ""),
        "split": task.split,
        "family": payload.get("task_family") or "",
        "protocol": payload.get("protocol", ""),
        "spec_id": payload.get("spec_id", ""),
        "expected_action": payload.get("expected_action", ""),
        "input_smiles": input_payload.get("smiles") or "",
        "has_witness": bool(evidence.get("feasible_witness_smiles")),
        "has_proof": _has_proof(evidence),
        "num_hard_constraints": task.hard_constraints,
        "num_soft_preferences": task.soft_preferences,
        "budget": _budget_string(payload),
        "corpus_source": task.corpus_source,
        "invariance_group_id": evidence.get("invariance_group_id") or "",
        "boundary_case_type": _boundary_case_type(evidence),
    }


def _normalize_prompt(prompt: str) -> str:
    text = prompt.lower().strip()
    text = re.sub(r"spec_v[0-9a-z_]+", "<spec>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered_headers = [str(header).replace("|", "\\|") for header in headers]
    rendered = ["| " + " | ".join(rendered_headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [str(cell).replace("|", "\\|") for cell in row]
        rendered.append("| " + " | ".join(cells) + " |")
    return "\n".join(rendered)


def _counter_table(counter: Counter[Any], key_name: str, value_name: str = "tasks") -> str:
    rows = [[key, value] for key, value in sorted(counter.items(), key=lambda item: str(item[0]))]
    return _markdown_table([key_name, value_name], rows)


def _duplicate_summary(values: list[tuple[str, str]], *, include_empty: bool = False) -> dict[str, int]:
    filtered = [value for value in values if include_empty or value[0]]
    counter = Counter(filtered)
    duplicate_groups = [item for item in counter.items() if item[1] > 1]
    return {
        "groups": len(duplicate_groups),
        "tasks_in_groups": sum(count for _value, count in duplicate_groups),
        "max_group_size": max((count for _value, count in duplicate_groups), default=0),
    }


def _top_duplicate_rows(values: list[tuple[str, ...]], *, limit: int = 10) -> list[list[Any]]:
    counter = Counter(values)
    rows: list[list[Any]] = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        if count <= 1:
            continue
        if not key or not key[0]:
            continue
        value = " / ".join(str(part) for part in key if part)
        rows.append([value, count])
        if len(rows) >= limit:
            break
    if not rows:
        return [["none", 0]]
    return rows


def _prompt_duplicate_rows(tasks: list[LoadedTask], *, limit: int = 10) -> list[list[Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[_normalize_prompt(str(task.payload.get("prompt", "")))].append(task.payload)

    rows: list[list[Any]] = []
    for prompt, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(group) <= 1:
            continue
        families = ",".join(sorted({str(row.get("task_family")) for row in group}))
        specs = len({str(row.get("spec_id")) for row in group})
        rows.append([len(group), families, specs, prompt[:120]])
        if len(rows) >= limit:
            break
    return rows or [[0, "none", 0, "none"]]


def _pick_examples(tasks: list[LoadedTask], family: str, limit: int = 10) -> list[LoadedTask]:
    candidates = [task for task in tasks if task.payload.get("task_family") == family]
    candidates.sort(
        key=lambda task: (
            str(task.payload.get("spec_id")),
            task.split,
            str(task.payload.get("protocol")),
            str(task.payload.get("task_id")),
        )
    )
    picked: list[LoadedTask] = []
    seen_buckets: set[tuple[str, str, str]] = set()
    for task in candidates:
        bucket = (
            task.split,
            str(task.payload.get("protocol")),
            str(task.payload.get("spec_id")),
        )
        if bucket in seen_buckets:
            continue
        picked.append(task)
        seen_buckets.add(bucket)
        if len(picked) >= limit:
            return picked
    for task in candidates:
        if task not in picked:
            picked.append(task)
        if len(picked) >= limit:
            break
    return picked


def _example_rows(tasks: list[LoadedTask], family: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for task in _pick_examples(tasks, family):
        payload = task.payload
        evidence = payload.get("evidence") or {}
        input_smiles = (payload.get("input") or {}).get("smiles") or ""
        witness = evidence.get("feasible_witness_smiles") or ""
        proof = "yes" if _has_proof(evidence) else "no"
        prompt = str(payload.get("prompt", ""))
        if len(prompt) > 90:
            prompt = prompt[:87] + "..."
        rows.append(
            [
                payload.get("task_id", ""),
                task.split,
                payload.get("protocol", ""),
                payload.get("spec_id", ""),
                payload.get("expected_action", ""),
                input_smiles,
                witness,
                proof,
                prompt,
            ]
        )
    return rows


def _load_tasks(release_dir: Path, specs: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> list[LoadedTask]:
    corpus_path = str((manifest.get("corpus") or {}).get("path") or "")
    tasks: list[LoadedTask] = []
    for split in RELEASE_SPLITS:
        for payload in _read_jsonl(release_dir / "tasks" / f"{split}.jsonl"):
            spec_id = str(payload.get("spec_id", ""))
            if spec_id not in specs:
                raise KeyError(f"Missing spec for task {payload.get('task_id')}: {spec_id}")
            hard, soft = _constraint_counts(specs[spec_id], payload.get("task_constraints"))
            evidence = payload.get("evidence") or {}
            input_smiles = (payload.get("input") or {}).get("smiles")
            witness = evidence.get("feasible_witness_smiles")
            if input_smiles or witness:
                corpus_source = corpus_path
            else:
                corpus_source = "spec_only/no_molecule"
            tasks.append(
                LoadedTask(
                    split=split,
                    payload=payload,
                    hard_constraints=hard,
                    soft_preferences=soft,
                    corpus_source=corpus_source,
                )
            )
    tasks.sort(key=lambda task: str(task.payload.get("task_id", "")))
    return tasks


def _write_inventory_csv(path: Path, tasks: list[LoadedTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        for task in tasks:
            writer.writerow(_inventory_row(task))


def _summary_markdown(
    release_dir: Path,
    tasks: list[LoadedTask],
    inventory_path: Path,
    manifest: dict[str, Any],
) -> str:
    rows = [_inventory_row(task) for task in tasks]

    by_family = Counter(row["family"] for row in rows)
    by_spec = Counter(row["spec_id"] for row in rows)
    by_expected_action = Counter(row["expected_action"] for row in rows)
    by_protocol = Counter(row["protocol"] for row in rows)
    by_split = Counter(row["split"] for row in rows)
    by_family_split = Counter((row["family"], row["split"]) for row in rows)
    by_boundary_type = Counter(
        row["boundary_case_type"].split(":distance=")[0]
        for row in rows
        if row["boundary_case_type"]
    )

    input_smiles = [str(row["input_smiles"]) for row in rows]
    canonical_inputs = [
        canonicalize_smiles(smiles) if smiles else ""
        for smiles in input_smiles
    ]
    raw_input_pairs = [(smiles, "") for smiles in input_smiles]
    canonical_input_pairs = [(smiles, "") for smiles in canonical_inputs]
    raw_input_spec_pairs = [
        (str(row["input_smiles"]), str(row["spec_id"])) for row in rows
    ]
    canonical_input_spec_pairs = [
        (canonical_inputs[index], str(row["spec_id"])) for index, row in enumerate(rows)
    ]
    exact_prompts = [(str(task.payload.get("prompt", "")), "") for task in tasks]
    normalized_prompts = [(_normalize_prompt(str(task.payload.get("prompt", ""))), "") for task in tasks]

    prompt_summary = _duplicate_summary(normalized_prompts)
    exact_prompt_summary = _duplicate_summary(exact_prompts)
    raw_input_summary = _duplicate_summary(raw_input_pairs)
    canonical_input_summary = _duplicate_summary(canonical_input_pairs)
    raw_input_spec_summary = _duplicate_summary(raw_input_spec_pairs)
    canonical_input_spec_summary = _duplicate_summary(canonical_input_spec_pairs)

    has_witness = sum(1 for row in rows if row["has_witness"])
    has_proof = sum(1 for row in rows if row["has_proof"])
    contradiction_proofs = sum(
        1 for task in tasks if (task.payload.get("evidence") or {}).get("contradiction_proof")
    )
    budget_proofs = sum(
        1 for task in tasks if (task.payload.get("evidence") or {}).get("budget_infeasible_note")
    )
    invariance_tasks = sum(1 for row in rows if row["invariance_group_id"])
    invariance_groups = len({row["invariance_group_id"] for row in rows if row["invariance_group_id"]})
    no_input = sum(1 for value in input_smiles if not value)

    family_split_rows = [
        [family, by_family_split.get((family, "train"), 0), by_family_split.get((family, "dev"), 0), by_family_split.get((family, "test"), 0), by_family[family]]
        for family in sorted(by_family)
    ]

    duplicate_overview_rows = [
        ["duplicate raw input molecules", raw_input_summary["groups"], raw_input_summary["tasks_in_groups"], raw_input_summary["max_group_size"]],
        ["duplicate canonical input molecules", canonical_input_summary["groups"], canonical_input_summary["tasks_in_groups"], canonical_input_summary["max_group_size"]],
        ["duplicate raw input/spec pairs", raw_input_spec_summary["groups"], raw_input_spec_summary["tasks_in_groups"], raw_input_spec_summary["max_group_size"]],
        ["duplicate canonical input/spec pairs", canonical_input_spec_summary["groups"], canonical_input_spec_summary["tasks_in_groups"], canonical_input_spec_summary["max_group_size"]],
        ["exact duplicate prompts", exact_prompt_summary["groups"], exact_prompt_summary["tasks_in_groups"], exact_prompt_summary["max_group_size"]],
        ["near-duplicate normalized prompts", prompt_summary["groups"], prompt_summary["tasks_in_groups"], prompt_summary["max_group_size"]],
    ]

    manifest_counts = manifest.get("counts") or {}
    lines: list[str] = [
        "# Current Task Inventory Summary",
        "",
        f"- release_dir: `{release_dir.as_posix()}`",
        f"- inventory_csv: `{inventory_path.as_posix()}`",
        f"- benchmark_id: `{manifest.get('benchmark_id', release_dir.name)}`",
        f"- total_tasks: {len(tasks)}",
        f"- manifest_total_tasks: {manifest_counts.get('total_tasks', 'n/a')}",
        f"- corpus_path: `{(manifest.get('corpus') or {}).get('path', '')}`",
        f"- no_input_tasks: {no_input}",
        f"- tasks_with_input_smiles: {len(tasks) - no_input}",
        f"- tasks_with_witness: {has_witness}",
        f"- tasks_with_abstention_proof: {has_proof}",
        f"- contradiction_proofs: {contradiction_proofs}",
        f"- budget_infeasible_notes: {budget_proofs}",
        f"- invariance_tasks: {invariance_tasks}",
        f"- invariance_groups: {invariance_groups}",
        "",
        "Note: `corpus_source` is inferred from the frozen release corpus path when a task has an input molecule or feasible witness. The task rows do not carry per-molecule corpus provenance.",
        "",
        "## Tasks Per Family",
        "",
        _counter_table(by_family, "family"),
        "",
        "## Tasks Per Family And Split",
        "",
        _markdown_table(["family", "train", "dev", "test", "total"], family_split_rows),
        "",
        "## Tasks Per Spec",
        "",
        _counter_table(by_spec, "spec_id"),
        "",
        "## Tasks Per Expected Action",
        "",
        _counter_table(by_expected_action, "expected_action"),
        "",
        "## Tasks Per Protocol",
        "",
        _counter_table(by_protocol, "protocol"),
        "",
        "## Tasks Per Split",
        "",
        _counter_table(by_split, "split"),
        "",
        "## Duplicate And Leakage Proxies",
        "",
        _markdown_table(["check", "duplicate_groups", "tasks_in_duplicate_groups", "max_group_size"], duplicate_overview_rows),
        "",
        "## Top Duplicate Raw Input Molecules",
        "",
        _markdown_table(["input_smiles", "tasks"], _top_duplicate_rows(raw_input_pairs)),
        "",
        "## Top Duplicate Canonical Input/Spec Pairs",
        "",
        _markdown_table(["canonical_input/spec_id", "tasks"], _top_duplicate_rows(canonical_input_spec_pairs)),
        "",
        "## Top Near-Duplicate Prompt Templates",
        "",
        _markdown_table(["tasks", "families", "num_specs", "normalized_prompt"], _prompt_duplicate_rows(tasks)),
        "",
        "## Boundary Case Types",
        "",
        _counter_table(by_boundary_type, "boundary_case_type") if by_boundary_type else "No boundary case metadata found.",
        "",
        "## Example Tasks By Family",
    ]

    for family in sorted(by_family):
        lines.extend(
            [
                "",
                f"### {family}",
                "",
                _markdown_table(
                    [
                        "task_id",
                        "split",
                        "protocol",
                        "spec_id",
                        "expected_action",
                        "input_smiles",
                        "witness",
                        "proof",
                        "prompt",
                    ],
                    _example_rows(tasks, family),
                ),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory a frozen SpecGuard-Chem release.")
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=Path("benchmarks/releases/sgchem_v0.3"),
        help="Frozen release directory containing tasks/, specs/, and MANIFEST.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("audits"),
        help="Directory for generated audit artifacts.",
    )
    args = parser.parse_args()

    release_dir = args.release_dir
    manifest_path = release_dir / "MANIFEST.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    specs = _load_specs(release_dir)
    tasks = _load_tasks(release_dir, specs, manifest)

    benchmark_id = str(manifest.get("benchmark_id") or release_dir.name)
    inventory_path = args.out_dir / f"{benchmark_id}_task_inventory.csv"
    summary_path = args.out_dir / f"{benchmark_id}_task_inventory_summary.md"

    _write_inventory_csv(inventory_path, tasks)
    summary = _summary_markdown(release_dir, tasks, inventory_path, manifest)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Wrote {inventory_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
