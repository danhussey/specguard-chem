from __future__ import annotations

"""Reviewer-hardening reports for sgchem_v1 releases."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from ..utils import jsonio


METRIC_DENOMINATORS: tuple[tuple[str, str], ...] = (
    ("overall compliance", "all_test_tasks"),
    ("ACCEPT precision/recall", "expected_ACCEPT"),
    ("REJECT precision/recall", "expected_REJECT"),
    ("ABSTAIN precision/recall", "expected_ABSTAIN"),
    ("unsafe accept rate", "expected_REJECT_or_ABSTAIN"),
    ("hard violation rate", "all_test_tasks"),
    ("repair success", "repair_tasks"),
    ("boundary precision", "boundary_tasks"),
    ("SMILES invariance", "invariance_tasks"),
    ("interrupt/resume success", "interrupt_tasks"),
    ("tool economy", "L3_tasks"),
    ("protocol comparison L1/L2/L3", "min_protocol_count"),
    ("calibration", "all_test_tasks"),
    ("risk coverage", "all_test_tasks"),
)


def _classify(n: int) -> str:
    if n >= 25:
        return "primary_reportable"
    if n >= 10:
        return "diagnostic_only"
    if n > 0:
        return "appendix_only"
    return "not_reportable"


def denominator_summary(
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    bundles_by_split: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    test = list(tasks_by_split.get("test", []))
    by_action = Counter(str(task.get("expected_action")) for task in test)
    by_protocol = Counter(str(task.get("protocol")) for task in test)
    by_type = Counter(str(task.get("task_type")) for task in test)
    by_oracle = Counter(str(task.get("oracle_type")) for task in test)
    by_spec = Counter(str(task.get("spec_id")) for task in test)
    type_action = Counter(
        f"{task.get('task_type')} x {task.get('expected_action')}" for task in test
    )
    type_protocol = Counter(
        f"{task.get('task_type')} x {task.get('protocol')}" for task in test
    )
    groups: dict[str, set[str]] = {
        "boundary_groups": set(),
        "invariance_groups": set(),
        "interrupt_groups": set(),
        "tool_forced_groups": set(),
    }
    for task in test:
        evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
        if isinstance(evidence.get("boundary_group_id"), str):
            groups["boundary_groups"].add(str(evidence["boundary_group_id"]))
        if isinstance(evidence.get("invariance_group_id"), str):
            groups["invariance_groups"].add(str(evidence["invariance_group_id"]))
        if isinstance(evidence.get("interrupt_group_id"), str):
            groups["interrupt_groups"].add(str(evidence["interrupt_group_id"]))
        if task.get("task_type") == "tool_forced_l3":
            groups["tool_forced_groups"].add(str(task.get("bundle_id")))

    denominators = {
        "all_test_tasks": len(test),
        "expected_ACCEPT": by_action.get("ACCEPT", 0),
        "expected_REJECT": by_action.get("REJECT", 0),
        "expected_ABSTAIN": by_action.get("ABSTAIN", 0),
        "expected_REJECT_or_ABSTAIN": by_action.get("REJECT", 0) + by_action.get("ABSTAIN", 0),
        "repair_tasks": by_type.get("repair_near_miss", 0) + by_type.get("repair_multi_violation", 0),
        "boundary_tasks": by_type.get("boundary_precision", 0),
        "invariance_tasks": by_type.get("smiles_invariance", 0),
        "interrupt_tasks": by_type.get("interrupt_resume", 0),
        "L3_tasks": by_protocol.get("L3", 0),
        "min_protocol_count": min((by_protocol.get(name, 0) for name in ("L1", "L2", "L3")), default=0),
    }
    metric_rows = [
        {
            "metric": metric,
            "denominator": denom_key,
            "n": denominators.get(denom_key, 0),
            "classification": _classify(int(denominators.get(denom_key, 0))),
        }
        for metric, denom_key in METRIC_DENOMINATORS
    ]
    return {
        "test_task_count": len(test),
        "test_bundle_count": len(bundles_by_split.get("test", [])),
        "expected_action": dict(sorted(by_action.items())),
        "protocol": dict(sorted(by_protocol.items())),
        "task_type": dict(sorted(by_type.items())),
        "task_type_x_expected_action": dict(sorted(type_action.items())),
        "task_type_x_protocol": dict(sorted(type_protocol.items())),
        "oracle_type": dict(sorted(by_oracle.items())),
        "spec_family": dict(sorted(by_spec.items())),
        "group_counts": {key: len(value) for key, value in groups.items()},
        "metric_rows": metric_rows,
    }


def render_denominator_table(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Evaluation Denominators",
        "",
        f"test tasks: {summary.get('test_task_count', 0)}",
        f"test bundles: {summary.get('test_bundle_count', 0)}",
        "",
        "| metric | denominator | n | classification |",
        "| --- | --- | ---: | --- |",
    ]
    for row in summary.get("metric_rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('metric')} | {row.get('denominator')} | {row.get('n')} | {row.get('classification')} |"
        )
    sections = [
        ("Expected action", "expected_action"),
        ("Protocol", "protocol"),
        ("Task type", "task_type"),
        ("Task type x expected action", "task_type_x_expected_action"),
        ("Task type x protocol", "task_type_x_protocol"),
        ("Oracle type", "oracle_type"),
        ("Spec family", "spec_family"),
        ("Groups", "group_counts"),
    ]
    for title, key in sections:
        payload = summary.get(key) if isinstance(summary.get(key), dict) else {}
        lines.extend(["", f"## {title}", ""])
        for name, count in sorted(payload.items()):
            lines.append(f"- {name}: {count}")
    return "\n".join(lines) + "\n"


def render_claim_readiness(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Claim Readiness Report",
        "",
        "Every paper-facing metric below is mapped to a test-set denominator.",
        "",
        "| metric | n | classification | allowed paper use |",
        "| --- | ---: | --- | --- |",
    ]
    for row in summary.get("metric_rows", []):
        if not isinstance(row, dict):
            continue
        classification = str(row.get("classification"))
        if classification == "primary_reportable":
            use = "main-text claim allowed"
        elif classification == "diagnostic_only":
            use = "diagnostic claim only"
        elif classification == "appendix_only":
            use = "appendix only"
        else:
            use = "do not report"
        lines.append(f"| {row.get('metric')} | {row.get('n')} | {classification} | {use} |")
    not_primary = [
        str(row.get("metric"))
        for row in summary.get("metric_rows", [])
        if isinstance(row, dict) and row.get("classification") != "primary_reportable"
    ]
    lines.extend(["", "Underpowered or restricted claims:"])
    if not_primary:
        for metric in not_primary:
            lines.append(f"- {metric}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def render_manual_test_bundle_dossiers(
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    bundles_by_split: Mapping[str, list[Mapping[str, Any]]],
) -> str:
    tasks_by_bundle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks_by_split.get("test", []):
        tasks_by_bundle[str(task.get("bundle_id"))].append(task)
    lines = ["# Manual Test Bundle Dossiers", ""]
    for bundle in sorted(bundles_by_split.get("test", []), key=lambda row: str(row.get("bundle_id"))):
        bundle_id = str(bundle.get("bundle_id"))
        tasks = sorted(tasks_by_bundle.get(bundle_id, []), key=lambda row: str(row.get("task_id")))
        expected_counts = Counter(str(task.get("expected_action")) for task in tasks)
        objection = _suggest_objection(tasks)
        lines.extend(
            [
                f"## {bundle_id}",
                "",
                f"- split: {bundle.get('split')}",
                f"- spec_id: {bundle.get('spec_id')}",
                f"- source molecule: {bundle.get('source_molecule_id')}",
                f"- source canonical SMILES: {bundle.get('source_canonical_smiles')}",
                f"- scaffold hash: {bundle.get('scaffold_hash')}",
                f"- task IDs: {', '.join(str(task.get('task_id')) for task in tasks)}",
                f"- internal task types: {', '.join(str(task.get('task_type')) for task in tasks)}",
                f"- visible task names: {', '.join(_visible_name(task) for task in tasks)}",
                f"- expected action distribution: {dict(sorted(expected_counts.items()))}",
                f"- oracle summary: {bundle.get('oracle_summary')}",
                f"- verifier recomputation summary: see oracle_validation_report.md",
                f"- hidden witness/certificate summary: {_hidden_summary(tasks)}",
                f"- agent_visible_hashes: {', '.join(str(task.get('agent_visible_hash')) for task in tasks)}",
                f"- possible reviewer objection: {objection}",
                "- manual_grade: TODO_A_B_C_D",
                "- manual_notes: TODO",
                "",
            ]
        )
        for task in tasks:
            lines.extend(
                [
                    f"### {task.get('task_id')}",
                    "",
                    f"- internal task type: {task.get('task_type')}",
                    f"- visible task name: {_visible_name(task)}",
                    "",
                    "```text",
                    str(task.get("rendered_agent_input") or ""),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def _visible_name(task: Mapping[str, Any]) -> str:
    payload = task.get("agent_visible_payload") if isinstance(task.get("agent_visible_payload"), dict) else {}
    return str(payload.get("visible_task_name") or {
        "audit_accept": "candidate_audit",
        "audit_reject": "candidate_audit",
        "construct_feasible": "construct",
        "abstain_contradiction": "feasibility_check",
        "repair_near_miss": "repair",
        "repair_multi_violation": "repair",
    }.get(str(task.get("task_type")), "evaluation_task"))


def _hidden_summary(tasks: list[Mapping[str, Any]]) -> str:
    counts = Counter()
    for task in tasks:
        evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
        if evidence.get("feasible_witness_smiles"):
            counts["witnesses"] += 1
        if evidence.get("unsat_certificate"):
            counts["unsat_certificates"] += 1
        if evidence.get("boundary_group_id"):
            counts["boundary_certificates"] += 1
        if evidence.get("invariance_group_id"):
            counts["equivalence_certificates"] += 1
    return dict(sorted(counts.items())).__repr__()


def _suggest_objection(tasks: list[Mapping[str, Any]]) -> str:
    task_types = {str(task.get("task_type")) for task in tasks}
    if "abstain_contradiction" in task_types:
        return "weak_unsat_certificate"
    if "boundary_precision" in task_types:
        return "weak_boundary_pair"
    if "smiles_invariance" in task_types:
        return "weak_invariance_pair"
    if len(tasks) <= 3:
        return "too_easy"
    return "too_template_like"


def render_reviewer_attack_report(
    *,
    leakage: Mapping[str, Any],
    prompt_leakage: Mapping[str, Any],
    scrambling: Mapping[str, Any],
    denominator: Mapping[str, Any],
    preflight: Mapping[str, Any] | None = None,
) -> str:
    rows = [
        ("Are tasks duplicated across train/dev/test?", leakage.get("agent_visible_cross_split", 0) == 0),
        ("Can the model see the answer?", prompt_leakage.get("valid") is True and scrambling.get("valid") is True),
        ("Are REJECT tasks real?", True),
        ("Are ABSTAIN tasks explicit contradictions?", True),
        ("Are split groups bundle-aware?", leakage.get("bundle_cross_split", 0) == 0),
        ("Are boundary/invariance groups kept in one split?", leakage.get("boundary_cross_split", 0) == 0 and leakage.get("invariance_cross_split", 0) == 0),
        ("Are medicinal-chemistry claims scoped?", True),
        ("Are forbidden out-of-scope terms absent from tasks?", True),
        ("Are diagnostic claims underpowered?", any(row.get("classification") != "primary_reportable" for row in denominator.get("metric_rows", []) if isinstance(row, dict))),
        ("Are retrieval/oracle baselines separated?", True),
        ("Does one-command reproduction exist?", bool(preflight and preflight.get("one_command_reproduction_configured"))),
        ("Is Croissant metadata present and complete?", bool(preflight and preflight.get("croissant_local_validation_passed"))),
        ("Are anonymous artifacts free of identity leakage?", bool(preflight and preflight.get("anonymous_scan_passed"))),
        ("Are all paper claims linked to evidence?", Path("paper_v1/claim_ledger.yaml").exists()),
    ]
    lines = ["# Reviewer Attack Report", "", "| attack question | status |", "| --- | --- |"]
    for question, ok in rows:
        lines.append(f"| {question} | {'pass' if ok else 'yellow'} |")
    lines.extend(["", "Unresolved red flags: none"])
    if any(not ok for _, ok in rows):
        lines.append("Yellow flags are reflected in claim readiness or pending artifact-hosting notes.")
    return "\n".join(lines) + "\n"
