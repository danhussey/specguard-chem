from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pilot_lib import (
    FORBIDDEN_TERMS,
    TARGET_COUNTS,
    evaluate_candidate,
    failure_ids,
    is_repair_task,
    load_tasks,
    micro_range_matches,
)
from specguard_chem.verifiers import canonicalize_smiles, is_valid_smiles

OUT = Path(__file__).resolve().parent


def main() -> int:
    errors: list[str] = []
    tasks = load_tasks()
    counts = Counter(task["family"] for task in tasks)
    if len(tasks) != 40:
        errors.append(f"expected 40 tasks, found {len(tasks)}")
    for family, expected in TARGET_COUNTS.items():
        if counts.get(family, 0) != expected:
            errors.append(f"expected {expected} {family}, found {counts.get(family, 0)}")

    for task in tasks:
        task_id = task.get("task_id", "<missing>")
        prompt = str(task.get("rendered_prompt") or "")
        witness = str(task.get("hidden_witness_smiles") or "")
        if witness and witness in prompt:
            errors.append(f"{task_id}: hidden witness appears in public prompt")
        if "instance_soft_window" in prompt:
            errors.append(f"{task_id}: instance_soft_window appears in public prompt")
        micro = micro_range_matches(prompt)
        if micro:
            errors.append(f"{task_id}: micro public range(s): {micro}")
        for term in FORBIDDEN_TERMS:
            if term in prompt.lower():
                errors.append(f"{task_id}: forbidden public term {term!r}")
        if not task.get("public_rule_card", {}).get("hard_constraints"):
            errors.append(f"{task_id}: missing public hard constraints")

        if is_repair_task(task):
            start = task.get("starting_smiles")
            if not isinstance(start, str) or not is_valid_smiles(start):
                errors.append(f"{task_id}: starting_smiles is invalid")
                continue
            start_failures = [
                fid
                for fid in failure_ids(task, start)
                if fid != "pilot_similarity_to_start"
            ]
            if not start_failures:
                errors.append(f"{task_id}: starting molecule does not fail a hard constraint")
            if task["family"] == "repair_near_miss" and len(set(start_failures)) != 1:
                errors.append(f"{task_id}: near miss should fail exactly one distinct hard constraint")
            if task["family"] == "repair_multi_violation" and len(set(start_failures)) < 2:
                errors.append(f"{task_id}: multi violation should fail at least two distinct hard constraints")
            if not task.get("public_rule_card", {}).get("similarity_constraint_public"):
                errors.append(f"{task_id}: repair task missing public similarity constraint")
            if task.get("preserve_scaffold") and not task.get("public_rule_card", {}).get("scaffold_constraint_public"):
                errors.append(f"{task_id}: scaffold task missing public scaffold constraint")

        if witness:
            if canonicalize_smiles(witness) != task.get("hidden_witness_canonical_smiles"):
                errors.append(f"{task_id}: witness canonical mismatch")
            witness_eval = evaluate_candidate(task, witness)
            if not witness_eval["success"]:
                errors.append(f"{task_id}: hidden witness does not pass pilot checks: {witness_eval['failure_reason']}")

    payload = {
        "valid": not errors,
        "num_errors": len(errors),
        "errors": errors,
        "num_tasks": len(tasks),
        "counts": dict(counts),
    }
    (OUT / "validation_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
