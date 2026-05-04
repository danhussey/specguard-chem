from __future__ import annotations

"""Strict sgchem_v1 release validation."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

from ..audits.duplicates import duplicate_summary
from ..audits.leakage import leakage_summary
from ..audits.oracle_validation import validate_oracles
from ..audits.scope_safety import scan_agent_visible_scope
from ..config import SpecModel, TaskModel
from ..utils import jsonio
from .oracles import oracle_is_compatible
from .splits import RELEASE_SPLITS, split_proportions_within_tolerance

REQUIRED_V1_FIELDS: tuple[str, ...] = (
    "task_id",
    "suite",
    "bundle_id",
    "task_type",
    "task_family",
    "protocol",
    "prompt_template",
    "rendered_agent_input",
    "agent_visible_hash",
    "input",
    "spec_id",
    "spec_instance_hash",
    "scoring",
    "expected",
    "expected_action",
    "oracle_type",
    "evidence",
    "budgets",
    "generation",
)

TASK_TYPES_V1 = {
    "construct_feasible",
    "repair_near_miss",
    "repair_multi_violation",
    "audit_accept",
    "audit_reject",
    "abstain_contradiction",
    "boundary_precision",
    "smiles_invariance",
    "interrupt_resume",
    "tool_forced_l3",
}
EXPECTED_ACTIONS = {"ACCEPT", "REJECT", "ABSTAIN"}
FORBIDDEN_VISIBLE_LABELS = {
    "audit_accept",
    "audit_reject",
}
FORBIDDEN_VISIBLE_ORACLE_TERMS = {
    "expected_action",
    "oracle_type",
    "evidence",
    "feasible_witness",
    "proof",
    "unsat_certificate",
    "violation_certificate",
    "boundary_certificate",
    "equivalence_certificate",
    "hard_pass",
    "failing_constraints",
    "curation_status",
    "task_id",
    "bundle_id",
}


def load_release_tasks_by_split(release_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split: (
            jsonio.read_jsonl(release_dir / "tasks" / f"{split}.jsonl")
            if (release_dir / "tasks" / f"{split}.jsonl").exists()
            else []
        )
        for split in RELEASE_SPLITS
    }


def load_release_bundles_by_split(release_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split: (
            jsonio.read_jsonl(release_dir / "bundles" / f"{split}.jsonl")
            if (release_dir / "bundles" / f"{split}.jsonl").exists()
            else []
        )
        for split in RELEASE_SPLITS
    }


def _spec_loader_for_release(release_dir: Path):
    cache: dict[str, SpecModel] = {}

    def _load(spec_id: str) -> SpecModel:
        if spec_id not in cache:
            spec_path = release_dir / "specs" / f"{spec_id}.json"
            payload = jsonio.read_json(spec_path)
            cache[spec_id] = SpecModel.model_validate(payload)
        return cache[spec_id]

    return _load


def _schema_checks(tasks_by_split: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    errors: list[str] = []
    task_ids: list[str] = []
    for split, tasks in tasks_by_split.items():
        for index, task in enumerate(tasks, start=1):
            task_id = str(task.get("task_id") or f"{split}#{index}")
            task_ids.append(task_id)
            try:
                TaskModel.model_validate(task)
            except Exception as exc:
                errors.append(f"{task_id}: TaskModel validation failed: {exc}")
            missing = [field for field in REQUIRED_V1_FIELDS if field not in task]
            if missing:
                errors.append(f"{task_id}: missing required v1 fields: {', '.join(missing)}")
            task_type = str(task.get("task_type", ""))
            expected_action = str(task.get("expected_action", ""))
            oracle_type = str(task.get("oracle_type", ""))
            if task_type not in TASK_TYPES_V1:
                errors.append(f"{task_id}: unsupported task_type {task_type}")
            if expected_action not in EXPECTED_ACTIONS:
                errors.append(f"{task_id}: unsupported expected_action {expected_action}")
            if not oracle_is_compatible(task_type, oracle_type):
                errors.append(f"{task_id}: oracle_type {oracle_type} incompatible with {task_type}")
            if task.get("rendered_agent_input") is None or task.get("agent_visible_hash") is None:
                errors.append(f"{task_id}: legacy prompt-only task format is not allowed")
    duplicates = [task_id for task_id, count in Counter(task_ids).items() if count > 1]
    if duplicates:
        errors.append("duplicate task_id values: " + ", ".join(sorted(duplicates)[:20]))
    return {
        "passed": len(errors) == 0,
        "num_errors": len(errors),
        "errors": errors,
    }


def _bundle_reference_checks(
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    bundles_by_split: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    errors: list[str] = []
    bundle_to_split: dict[str, str] = {}
    bundle_task_ids: dict[str, set[str]] = {}
    bundle_spec_ids: dict[str, set[str]] = defaultdict(set)
    for split, bundles in bundles_by_split.items():
        for bundle in bundles:
            bundle_id = str(bundle.get("bundle_id", ""))
            if bundle_id in bundle_to_split and bundle_to_split[bundle_id] != split:
                errors.append(f"{bundle_id}: bundle appears in {bundle_to_split[bundle_id]} and {split}")
            bundle_to_split[bundle_id] = split
            task_ids = bundle.get("task_ids")
            bundle_task_ids[bundle_id] = set(str(value) for value in task_ids) if isinstance(task_ids, list) else set()
            if bundle.get("spec_id"):
                bundle_spec_ids[bundle_id].add(str(bundle.get("spec_id")))
            if len(bundle_task_ids[bundle_id]) < 3:
                errors.append(f"{bundle_id}: bundle has fewer than 3 tasks")
    task_by_id: dict[str, Mapping[str, Any]] = {}
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            task_by_id[task_id] = task
            bundle_id = str(task.get("bundle_id", ""))
            if bundle_id not in bundle_to_split:
                errors.append(f"{task_id}: references missing bundle {bundle_id}")
            elif bundle_to_split[bundle_id] != split:
                errors.append(f"{task_id}: bundle split {bundle_to_split[bundle_id]} differs from task split {split}")
            bundle_spec_ids[bundle_id].add(str(task.get("spec_id", "")))
    for bundle_id, task_ids in bundle_task_ids.items():
        for task_id in task_ids:
            if task_id not in task_by_id:
                errors.append(f"{bundle_id}: listed task_id {task_id} does not exist")
        if len(bundle_spec_ids[bundle_id]) != 1:
            errors.append(f"{bundle_id}: references multiple spec IDs")
    return {
        "passed": len(errors) == 0,
        "num_errors": len(errors),
        "errors": errors,
    }


def _protocol_checks(tasks_by_split: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    errors: list[str] = []
    for _split, tasks in tasks_by_split.items():
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            protocol = str(task.get("protocol", ""))
            budgets = task.get("budgets") if isinstance(task.get("budgets"), dict) else {}
            visible = task.get("agent_visible_payload") if isinstance(task.get("agent_visible_payload"), dict) else {}
            allowed_tools = visible.get("allowed_tools") if isinstance(visible.get("allowed_tools"), list) else []
            max_verify_calls = int(budgets.get("max_verify_calls", 0) or 0)
            if protocol == "L1" and max_verify_calls != 0:
                errors.append(f"{task_id}: L1 must not have verifier tool access")
            if protocol == "L1" and "verify" in allowed_tools:
                errors.append(f"{task_id}: L1 visible tools must not include verify")
            if protocol == "L2" and max_verify_calls != 0:
                errors.append(f"{task_id}: L2 must not have direct verifier calls")
            if protocol == "L3" and max_verify_calls <= 0:
                errors.append(f"{task_id}: L3 requires max_verify_calls > 0")
            if task.get("task_type") == "tool_forced_l3" and "verify" not in allowed_tools:
                errors.append(f"{task_id}: tool_forced_l3 requires verify tool")
            if task.get("task_type") == "interrupt_resume":
                if int(budgets.get("max_steps", 0) or 0) < 2:
                    errors.append(f"{task_id}: interrupt_resume requires max_steps >= 2")
                evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
                interrupt = evidence.get("interrupt") if isinstance(evidence.get("interrupt"), dict) else {}
                fields = interrupt.get("expected_state_fields")
                field_text = " ".join(str(value) for value in fields) if isinstance(fields, list) else ""
                for required in ("state", "report", "continue"):
                    if required not in field_text:
                        errors.append(f"{task_id}: interrupt evidence missing {required} behavior")
    return {
        "passed": len(errors) == 0,
        "num_errors": len(errors),
        "errors": errors,
    }


def _prompt_visibility_checks(tasks_by_split: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    errors: list[str] = []
    for _split, tasks in tasks_by_split.items():
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            rendered = str(task.get("rendered_agent_input") or task.get("prompt") or "")
            lower = rendered.lower()
            for label in FORBIDDEN_VISIBLE_LABELS:
                if label in lower:
                    errors.append(f"{task_id}: visible prompt contains internal label {label}")
            for term in FORBIDDEN_VISIBLE_ORACLE_TERMS:
                if term in lower:
                    errors.append(f"{task_id}: visible prompt contains hidden oracle term {term}")
            evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
            input_block = task.get("input") if isinstance(task.get("input"), dict) else {}
            visible_smiles = {
                value
                for value in (
                    input_block.get("smiles"),
                    input_block.get("candidate_smiles"),
                )
                if isinstance(value, str) and value
            }
            for key in ("feasible_witness_smiles", "feasible_witness_canonical_smiles"):
                value = evidence.get(key)
                if isinstance(value, str) and value and value not in visible_smiles and value in rendered:
                    errors.append(f"{task_id}: rendered_agent_input exposes hidden witness SMILES")
    return {
        "passed": len(errors) == 0,
        "num_errors": len(errors),
        "errors": errors,
    }


def _composition_checks(tasks_by_split: Mapping[str, list[Mapping[str, Any]]], oracle_summary: Mapping[str, Any]) -> dict[str, Any]:
    tasks = [task for rows in tasks_by_split.values() for task in rows]
    by_action = Counter(str(task.get("expected_action")) for task in tasks)
    by_type = Counter(str(task.get("task_type")) for task in tasks)
    by_protocol = Counter(str(task.get("protocol")) for task in tasks)
    errors: list[str] = []
    if by_action.get("REJECT", 0) == 0:
        errors.append("REJECT count == 0")
    if by_action.get("ABSTAIN", 0) == 0:
        errors.append("ABSTAIN count == 0")
    if by_protocol.get("L3", 0) == 0:
        errors.append("L3 count == 0")
    if by_type.get("audit_reject", 0) == 0:
        errors.append("audit_reject count == 0")
    oracle_counts = oracle_summary.get("counts") if isinstance(oracle_summary.get("counts"), dict) else {}
    boundary = oracle_counts.get("boundary_groups", {}) if isinstance(oracle_counts.get("boundary_groups"), dict) else {}
    invariance = oracle_counts.get("invariance_groups", {}) if isinstance(oracle_counts.get("invariance_groups"), dict) else {}
    if int(boundary.get("checked", 0) or 0) < 1:
        errors.append("boundary_group count < configured minimum")
    if int(invariance.get("checked", 0) or 0) < 1:
        errors.append("invariance_group count < configured minimum")
    return {
        "passed": len(errors) == 0,
        "num_errors": len(errors),
        "errors": errors,
        "tasks_per_expected_action": dict(sorted(by_action.items())),
        "tasks_per_task_type": dict(sorted(by_type.items())),
        "tasks_per_protocol": dict(sorted(by_protocol.items())),
    }


def validate_release_v1(release_dir: Path, *, strict: bool = True) -> dict[str, Any]:
    tasks_by_split = load_release_tasks_by_split(release_dir)
    bundles_by_split = load_release_bundles_by_split(release_dir)
    spec_loader = _spec_loader_for_release(release_dir)

    schema = _schema_checks(tasks_by_split)
    bundle_refs = _bundle_reference_checks(tasks_by_split, bundles_by_split)
    oracles = validate_oracles(tasks_by_split, spec_loader=spec_loader)
    leakage = leakage_summary(tasks_by_split, bundles_by_split)
    duplicates = duplicate_summary(tasks_by_split)
    protocol = _protocol_checks(tasks_by_split)
    prompt_visibility = _prompt_visibility_checks(tasks_by_split)
    safety = scan_agent_visible_scope(tasks_by_split)
    composition = _composition_checks(tasks_by_split, oracles)

    split_errors: list[str] = []
    if leakage.get("bundle_cross_split", 0):
        split_errors.append("bundle crosses splits")
    if leakage.get("invariance_cross_split", 0):
        split_errors.append("invariance group crosses splits")
    if leakage.get("boundary_cross_split", 0):
        split_errors.append("boundary group crosses splits")
    if leakage.get("interrupt_cross_split", 0):
        split_errors.append("interrupt group crosses splits")
    if leakage.get("agent_visible_cross_split", 0):
        split_errors.append("agent-visible duplicate crosses splits")
    if leakage.get("canonical_input_spec_cross_split", 0):
        split_errors.append("canonical input/spec duplicate crosses splits")
    bundle_counts = leakage.get("split_bundle_counts") if isinstance(leakage.get("split_bundle_counts"), dict) else {}
    if strict and not split_proportions_within_tolerance({key: int(value) for key, value in bundle_counts.items()}):
        split_errors.append("split proportions outside configured tolerance")
    split_checks = {
        "passed": len(split_errors) == 0,
        "num_errors": len(split_errors),
        "errors": split_errors,
        **leakage,
    }

    safety_errors: list[str] = []
    if safety.get("agent_visible_forbidden_matches", 0):
        safety_errors.append("agent-visible forbidden term matches found")
    safety_checks = {
        "passed": len(safety_errors) == 0,
        "num_errors": len(safety_errors),
        "errors": safety_errors,
        **safety,
    }

    checks = {
        "schema": schema,
        "bundles": bundle_refs,
        "oracles": {
            "passed": bool(oracles.get("valid", False)),
            "num_errors": int(oracles.get("num_errors", 0)),
            "errors": oracles.get("errors", []),
            "counts": oracles.get("counts", {}),
        },
        "splits": split_checks,
        "protocols": protocol,
        "prompt_visibility": prompt_visibility,
        "safety_scope": safety_checks,
        "composition": composition,
        "duplicates": {
            "passed": duplicates.get("agent_visible_cross_split", 0) == 0
            and duplicates.get("unmarked_within_split_duplicates", 0) == 0,
            "num_errors": int(duplicates.get("agent_visible_cross_split", 0) or 0)
            + int(duplicates.get("unmarked_within_split_duplicates", 0) or 0),
            "errors": (
                (["agent-visible duplicates across splits"] if duplicates.get("agent_visible_cross_split", 0) else [])
                + (["unmarked agent-visible duplicates within split"] if duplicates.get("unmarked_within_split_duplicates", 0) else [])
            ),
            **duplicates,
        },
    }
    all_errors: list[str] = []
    for check_name, check in checks.items():
        errors = check.get("errors") if isinstance(check, dict) else []
        if isinstance(errors, list):
            all_errors.extend(f"{check_name}: {error}" for error in errors)

    result = {
        "valid": len(all_errors) == 0,
        "num_errors": len(all_errors),
        "num_warnings": 0,
        "errors": all_errors,
        "checks": checks,
        "num_tasks": sum(len(rows) for rows in tasks_by_split.values()),
        "num_bundles": sum(len(rows) for rows in bundles_by_split.values()),
    }
    return json.loads(json.dumps(result, sort_keys=True, ensure_ascii=True))


def validate_croissant_metadata(path: Path, *, anonymous: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    try:
        payload = jsonio.read_json(path)
    except Exception as exc:
        return {
            "valid": False,
            "num_errors": 1,
            "errors": [f"croissant JSON parse failed: {exc}"],
        }
    for field in ("name", "version", "license"):
        if not payload.get(field):
            errors.append(f"missing {field}")
    distribution = payload.get("distribution")
    distribution_text = json.dumps(distribution, sort_keys=True) if distribution is not None else ""
    for required in ("tasks/train.jsonl", "tasks/dev.jsonl", "tasks/test.jsonl", "specs/spec_catalog.json"):
        if required not in distribution_text:
            errors.append(f"distribution does not reference {required}")
    rai = payload.get("responsibleAI") or payload.get("responsible_ai")
    if not isinstance(rai, dict):
        errors.append("missing Responsible AI fields")
    else:
        for field in ("intendedUse", "outOfScopeUse", "dataGenerationProcess", "safetyLimitations"):
            if not rai.get(field):
                errors.append(f"Responsible AI missing {field}")
    if not payload.get("externalValidationStatus"):
        errors.append("missing externalValidationStatus")
    if anonymous:
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "".join(("Da", "niel")),
            "".join(("Hus", "sey")),
            "".join(("/Us", "ers/")),
            "".join(("github.com/", "dan", "hus", "sey")),
        ):
            if forbidden in rendered:
                errors.append(f"anonymous metadata contains {forbidden}")
    return {
        "valid": len(errors) == 0,
        "num_errors": len(errors),
        "errors": errors,
    }
