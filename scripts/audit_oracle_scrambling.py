from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from specguard_chem.benchmark.effective_spec import build_effective_spec
from specguard_chem.benchmark.sweep import load_baseline_matrix
from specguard_chem.config import SpecModel, TaskModel
from specguard_chem.dataset.validate_v1 import load_release_tasks_by_split
from specguard_chem.models import build_adapter
from specguard_chem.runner.public_view import (
    build_public_adapter_request,
    render_adapter_request_prompt,
)
from specguard_chem.runner.runner import TaskRunner
from specguard_chem.utils import jsonio


def _spec_payload(release: Path, task: TaskModel) -> dict[str, Any]:
    raw_spec = jsonio.read_json(release / "specs" / f"{task.spec_id}.json")
    spec = SpecModel.model_validate(raw_spec)
    return build_effective_spec(spec, task.task_constraints).model_dump(mode="json")


def _public_prompt(release: Path, task_payload: Mapping[str, Any]) -> str:
    task = TaskModel.model_validate(task_payload)
    spec_payload = _spec_payload(release, task)
    request = build_public_adapter_request(
        task=task,
        spec=spec_payload,
        round_index=1,
        tools=TaskRunner._tool_spec(task.protocol),
        failure_feedback=None,
        interrupt=None,
    )
    return render_adapter_request_prompt(request)


def scramble_hidden_fields(task: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    scrambled = deepcopy(dict(task))
    actions = ("ACCEPT", "REJECT", "ABSTAIN")
    outcomes = {"ACCEPT": "PASS", "REJECT": "FAIL", "ABSTAIN": "ABSTAIN"}
    action = actions[index % len(actions)]
    scrambled["expected_action"] = action
    scrambled["expected"] = outcomes[action]
    scrambled["oracle_type"] = "feasible_witness"
    scrambled["evidence"] = {
        "oracle_type": "feasible_witness",
        "feasible_witness_smiles": f"SCRAMBLED_WITNESS_{index}",
        "proof": f"SCRAMBLED_PROOF_{index}",
        "unsat_certificate": {
            "kind": "scrambled",
            "constraints": [f"scrambled_constraint_{index}"],
            "reason": f"scrambled reason {index}",
        },
    }
    scrambled["split"] = "scrambled"
    scrambled["task_id"] = f"scrambled_task_{index:05d}"
    scrambled["bundle_id"] = f"scrambled_bundle_{index:05d}"
    return scrambled


def audit_oracle_scrambling(
    release: Path,
    *,
    baselines: Path = Path("baselines/paper_baselines.yaml"),
) -> dict[str, Any]:
    tasks_by_split = load_release_tasks_by_split(release)
    tasks = [task for rows in tasks_by_split.values() for task in rows]
    prompt_mismatches: list[str] = []
    original_prompts: list[str] = []
    scrambled_prompts: list[str] = []
    for index, task in enumerate(tasks):
        original = _public_prompt(release, task)
        scrambled = _public_prompt(release, scramble_hidden_fields(task, index=index))
        original_prompts.append(original)
        scrambled_prompts.append(scrambled)
        if original != scrambled:
            prompt_mismatches.append(str(task.get("task_id", index)))

    output_mismatches: list[dict[str, Any]] = []
    matrix = load_baseline_matrix(baselines) if baselines.exists() else []
    for entry in matrix:
        if entry.track == "oracle_upper_bound":
            continue
        if entry.optional:
            continue
        adapter_a = build_adapter(entry.model, seed=7, **entry.adapter_kwargs)
        adapter_b = build_adapter(entry.model, seed=7, **entry.adapter_kwargs)
        for index, task in enumerate(tasks[: min(len(tasks), 50)]):
            raw_task = TaskModel.model_validate(task)
            scrambled_task = TaskModel.model_validate(scramble_hidden_fields(task, index=index))
            spec_payload = _spec_payload(release, raw_task)
            request_original = build_public_adapter_request(
                task=raw_task,
                spec=spec_payload,
                round_index=1,
                tools=TaskRunner._tool_spec(raw_task.protocol),
                failure_feedback=None,
                interrupt=None,
            )
            request_scrambled = build_public_adapter_request(
                task=scrambled_task,
                spec=spec_payload,
                round_index=1,
                tools=TaskRunner._tool_spec(scrambled_task.protocol),
                failure_feedback=None,
                interrupt=None,
            )
            out_a = adapter_a.step(request_original)
            out_b = adapter_b.step(request_scrambled)
            if json.dumps(out_a, sort_keys=True) != json.dumps(out_b, sort_keys=True):
                output_mismatches.append(
                    {
                        "baseline": entry.name,
                        "task_id": task.get("task_id"),
                    }
                )
                break

    summary = {
        "valid": not prompt_mismatches and not output_mismatches,
        "tasks_checked": len(tasks),
        "public_views_identical_under_oracle_scrambling": not prompt_mismatches,
        "non_oracle_baseline_outputs_identical": not output_mismatches,
        "oracle_dependent_baselines": [
            entry.name for entry in matrix if entry.track == "oracle_upper_bound"
        ],
        "prompt_mismatches": prompt_mismatches[:50],
        "baseline_output_mismatches": output_mismatches[:50],
    }
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Oracle Scrambling Report",
        "",
        f"tasks_checked: {summary.get('tasks_checked', 0)}",
        "public_views_identical_under_oracle_scrambling: "
        f"{str(bool(summary.get('public_views_identical_under_oracle_scrambling'))).lower()}",
        "non_oracle_baseline_outputs_identical: "
        f"{str(bool(summary.get('non_oracle_baseline_outputs_identical'))).lower()}",
        f"valid: {str(bool(summary.get('valid'))).lower()}",
    ]
    oracle_baselines = summary.get("oracle_dependent_baselines")
    if isinstance(oracle_baselines, list):
        lines.append(f"oracle_dependent_baselines: {', '.join(map(str, oracle_baselines)) or 'none'}")
    mismatches = summary.get("prompt_mismatches") if isinstance(summary.get("prompt_mismatches"), list) else []
    output_mismatches = summary.get("baseline_output_mismatches") if isinstance(summary.get("baseline_output_mismatches"), list) else []
    if mismatches or output_mismatches:
        lines.extend(["", "Findings:"])
        for task_id in mismatches[:50]:
            lines.append(f"- public view changed for {task_id}")
        for row in output_mismatches[:50]:
            lines.append(f"- baseline output changed for {row.get('baseline')} on {row.get('task_id')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, default=Path("baselines/paper_baselines.yaml"))
    args = parser.parse_args()
    summary = audit_oracle_scrambling(args.release, baselines=args.baselines)
    out = args.release / "audits" / "oracle_scrambling_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
