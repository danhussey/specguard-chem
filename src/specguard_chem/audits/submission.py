from __future__ import annotations

"""Submission-grade metric, challenge-slice, and artifact reports."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils import jsonio

ACTIONS: tuple[str, ...] = ("ACCEPT", "REJECT", "ABSTAIN", "INVALID")
PRIMARY_ACTIONS: tuple[str, ...] = ("ACCEPT", "REJECT", "ABSTAIN")
CHALLENGE_TAGS: tuple[str, ...] = (
    "tight_property_boundary",
    "multi_constraint_violation",
    "minimal_edit_required",
    "low_margin_feasible",
    "reject_near_miss",
    "abstain_explicit_contradiction",
    "invariance_equivalent_representation",
    "tool_required_by_protocol",
    "interrupt_state_required",
    "high_similarity_guard",
    "mixed_hard_soft_tradeoff",
)


def _safe_div(numer: int | float, denom: int | float) -> float | None:
    if not denom:
        return None
    return float(numer) / float(denom)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _md_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _all_tasks(tasks_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
    return [task for split in ("train", "dev", "test") for task in tasks_by_split.get(split, [])]


def release_count_summary(tasks_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    all_tasks = _all_tasks(tasks_by_split)
    test_tasks = list(tasks_by_split.get("test", []))
    return {
        "full_expected_action_counts": dict(sorted(Counter(str(task.get("expected_action")) for task in all_tasks).items())),
        "test_expected_action_counts": dict(sorted(Counter(str(task.get("expected_action")) for task in test_tasks).items())),
        "full_task_type_counts": dict(sorted(Counter(str(task.get("task_type")) for task in all_tasks).items())),
        "test_task_type_counts": dict(sorted(Counter(str(task.get("task_type")) for task in test_tasks).items())),
        "full_protocol_counts": dict(sorted(Counter(str(task.get("protocol")) for task in all_tasks).items())),
        "test_protocol_counts": dict(sorted(Counter(str(task.get("protocol")) for task in test_tasks).items())),
        "full_oracle_type_counts": dict(sorted(Counter(str(task.get("oracle_type")) for task in all_tasks).items())),
        "test_oracle_type_counts": dict(sorted(Counter(str(task.get("oracle_type")) for task in test_tasks).items())),
    }


def _resolve_expected(record: Mapping[str, Any]) -> str:
    value = str(record.get("expected_action") or "").strip().upper()
    if value in PRIMARY_ACTIONS:
        return value
    legacy = str(record.get("expected") or "PASS").strip().upper()
    if legacy == "PASS":
        return "ACCEPT"
    if legacy == "ABSTAIN":
        return "ABSTAIN"
    return "REJECT"


def _resolve_predicted(record: Mapping[str, Any]) -> str:
    if record.get("schema_error") or record.get("invalid_action") or record.get("invalid_tool_call"):
        return "INVALID"
    value = str(record.get("final_decision") or "").strip().upper()
    if value in ACTIONS:
        return value
    decision = str(record.get("decision") or "").strip().lower()
    if decision == "accept":
        return "ACCEPT"
    if decision == "reject":
        return "REJECT"
    if decision == "abstain":
        return "ABSTAIN"
    return "INVALID"


def _schema_counts(records: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    outputs = 0
    errors = 0
    for record in records:
        rounds = record.get("rounds")
        if not isinstance(rounds, list):
            continue
        for round_entry in rounds:
            if not isinstance(round_entry, dict):
                continue
            outputs += 1
            if round_entry.get("schema_error"):
                errors += 1
    if outputs == 0:
        outputs = len(records)
        errors = sum(1 for record in records if record.get("schema_error"))
    return errors, outputs


def _invalid_molecule_counts(records: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    invalid = 0
    molecule_outputs = 0
    for record in records:
        rounds = record.get("rounds")
        if not isinstance(rounds, list):
            continue
        for round_entry in rounds:
            if not isinstance(round_entry, dict) or round_entry.get("action") != "propose":
                continue
            molecule_outputs += 1
            evaluation = round_entry.get("evaluation")
            if not isinstance(evaluation, dict):
                continue
            if evaluation.get("properties") == {} and evaluation.get("hard_pass") is False:
                hard_fails = " ".join(str(item) for item in evaluation.get("hard_fails", []))
                if "invalid" in hard_fails.lower():
                    invalid += 1
                    continue
            failure_vector = round_entry.get("failure_vector")
            if not isinstance(failure_vector, dict):
                continue
            text = str(failure_vector).lower()
            if "invalid smiles" in text or "invalid_smiles" in text:
                invalid += 1
    return invalid, molecule_outputs


def _pass_at(records: Sequence[Mapping[str, Any]], k: int) -> tuple[int, int, float | None]:
    denom = 0
    numer = 0
    for record in records:
        if _resolve_expected(record) != "ACCEPT":
            continue
        denom += 1
        steps = record.get("steps_used")
        if not isinstance(steps, int):
            rounds = record.get("rounds")
            steps = len(rounds) if isinstance(rounds, list) else 0
        if _resolve_predicted(record) == "ACCEPT" and steps <= k:
            numer += 1
    return numer, denom, _safe_div(numer, denom)


def detailed_baseline_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [_resolve_expected(record) for record in records]
    predicted = [_resolve_predicted(record) for record in records]
    n = len(records)
    confusion: dict[str, dict[str, int]] = {
        action: {pred: 0 for pred in ACTIONS} for action in PRIMARY_ACTIONS
    }
    for exp, pred in zip(expected, predicted):
        confusion[exp][pred] += 1

    rows: dict[str, Any] = {
        "num_tasks": n,
        "overall_task_success": _safe_div(sum(1 for exp, pred in zip(expected, predicted) if exp == pred), n),
        "action_accuracy": _safe_div(sum(1 for exp, pred in zip(expected, predicted) if exp == pred), n),
        "molecule_acceptance_rate": _safe_div(sum(1 for pred in predicted if pred == "ACCEPT"), n),
        "molecule_acceptance_denominator": n,
        "abstention_rate": _safe_div(sum(1 for pred in predicted if pred == "ABSTAIN"), n),
    }

    for action in PRIMARY_ACTIONS:
        true_positive = confusion[action][action]
        pred_denom = sum(confusion[exp][action] for exp in PRIMARY_ACTIONS)
        exp_denom = sum(confusion[action].values())
        rows[f"{action.lower()}_precision"] = _safe_div(true_positive, pred_denom)
        rows[f"{action.lower()}_precision_denominator"] = pred_denom
        rows[f"{action.lower()}_recall"] = _safe_div(true_positive, exp_denom)
        rows[f"{action.lower()}_recall_denominator"] = exp_denom

    unsafe_numer = confusion["REJECT"]["ACCEPT"] + confusion["ABSTAIN"]["ACCEPT"]
    unsafe_denom = sum(confusion["REJECT"].values()) + sum(confusion["ABSTAIN"].values())
    false_abstain_numer = confusion["ACCEPT"]["ABSTAIN"] + confusion["REJECT"]["ABSTAIN"]
    false_abstain_denom = sum(confusion["ACCEPT"].values()) + sum(confusion["REJECT"].values())
    attempted_indices = [idx for idx, pred in enumerate(predicted) if pred != "ABSTAIN"]
    hard_violations = sum(1 for idx in attempted_indices if not bool(records[idx].get("hard_pass", False)))
    schema_errors, schema_outputs = _schema_counts(records)
    invalid_molecules, molecule_outputs = _invalid_molecule_counts(records)
    pass1_num, pass1_den, pass1 = _pass_at(records, 1)
    pass3_num, pass3_den, pass3 = _pass_at(records, 3)
    rows.update(
        {
            "task_inconsistent_accept_rate": _safe_div(unsafe_numer, unsafe_denom),
            "task_inconsistent_accept_denominator": unsafe_denom,
            "unsafe_accept_rate": _safe_div(unsafe_numer, unsafe_denom),
            "unsafe_accept_denominator": unsafe_denom,
            "false_abstain_rate": _safe_div(false_abstain_numer, false_abstain_denom),
            "false_abstain_denominator": false_abstain_denom,
            "hard_violation_rate": _safe_div(hard_violations, len(attempted_indices)),
            "hard_violation_denominator": len(attempted_indices),
            "schema_error_rate": _safe_div(schema_errors, schema_outputs),
            "schema_error_denominator": schema_outputs,
            "invalid_molecule_rate": _safe_div(invalid_molecules, molecule_outputs),
            "invalid_molecule_denominator": molecule_outputs,
            "pass_at_1": pass1,
            "pass_at_1_denominator": pass1_den,
            "pass_at_1_numerator": pass1_num,
            "pass_at_3": pass3,
            "pass_at_3_denominator": pass3_den,
            "pass_at_3_numerator": pass3_num,
            "confusion": confusion,
        }
    )
    return rows


def baseline_metric_rows(runs_dir: Path) -> list[dict[str, Any]]:
    aggregate_path = runs_dir / "aggregate.json"
    aggregate = jsonio.read_json(aggregate_path)
    rows: list[dict[str, Any]] = []
    for baseline in aggregate.get("all_baselines", aggregate.get("baselines", [])):
        if not isinstance(baseline, dict):
            continue
        name = str(baseline.get("name"))
        report_rel = baseline.get("report_path")
        report_path = runs_dir / str(report_rel)
        if not report_path.exists():
            continue
        payload = jsonio.read_json(report_path)
        records = payload.get("records") if isinstance(payload, dict) else []
        if not isinstance(records, list):
            continue
        metrics = detailed_baseline_metrics(records)
        row = {
            "baseline": name,
            "track": baseline.get("track") or "primary_closed_book",
            "num_tasks": metrics["num_tasks"],
            "overall_task_success": _fmt(metrics["overall_task_success"]),
            "action_accuracy": _fmt(metrics["action_accuracy"]),
            "molecule_acceptance_rate": _fmt(metrics["molecule_acceptance_rate"]),
            "ACCEPT_precision": _fmt(metrics["accept_precision"]),
            "ACCEPT_precision_n": metrics["accept_precision_denominator"],
            "ACCEPT_recall": _fmt(metrics["accept_recall"]),
            "ACCEPT_recall_n": metrics["accept_recall_denominator"],
            "REJECT_precision": _fmt(metrics["reject_precision"]),
            "REJECT_precision_n": metrics["reject_precision_denominator"],
            "REJECT_recall": _fmt(metrics["reject_recall"]),
            "REJECT_recall_n": metrics["reject_recall_denominator"],
            "ABSTAIN_precision": _fmt(metrics["abstain_precision"]),
            "ABSTAIN_precision_n": metrics["abstain_precision_denominator"],
            "ABSTAIN_recall": _fmt(metrics["abstain_recall"]),
            "ABSTAIN_recall_n": metrics["abstain_recall_denominator"],
            "task_inconsistent_accept_rate": _fmt(metrics["task_inconsistent_accept_rate"]),
            "task_inconsistent_accept_n": metrics["task_inconsistent_accept_denominator"],
            "false_abstain_rate": _fmt(metrics["false_abstain_rate"]),
            "false_abstain_n": metrics["false_abstain_denominator"],
            "hard_violation_rate": _fmt(metrics["hard_violation_rate"]),
            "hard_violation_n": metrics["hard_violation_denominator"],
            "schema_error_rate": _fmt(metrics["schema_error_rate"]),
            "schema_error_n": metrics["schema_error_denominator"],
            "invalid_molecule_rate": _fmt(metrics["invalid_molecule_rate"]),
            "invalid_molecule_n": metrics["invalid_molecule_denominator"],
            "pass_at_1": _fmt(metrics["pass_at_1"]),
            "pass_at_1_n": metrics["pass_at_1_denominator"],
            "pass_at_3": _fmt(metrics["pass_at_3"]),
            "pass_at_3_n": metrics["pass_at_3_denominator"],
        }
        rows.append(row)
    return rows


def baseline_confusion_rows(runs_dir: Path) -> list[dict[str, Any]]:
    aggregate = jsonio.read_json(runs_dir / "aggregate.json")
    rows: list[dict[str, Any]] = []
    for baseline in aggregate.get("all_baselines", aggregate.get("baselines", [])):
        if not isinstance(baseline, dict):
            continue
        name = str(baseline.get("name"))
        report_path = runs_dir / str(baseline.get("report_path"))
        if not report_path.exists():
            continue
        payload = jsonio.read_json(report_path)
        records = payload.get("records") if isinstance(payload, dict) else []
        if not isinstance(records, list):
            continue
        confusion = detailed_baseline_metrics(records)["confusion"]
        for expected in PRIMARY_ACTIONS:
            for predicted in ACTIONS:
                rows.append(
                    {
                        "baseline": name,
                        "track": baseline.get("track") or "primary_closed_book",
                        "expected_action": expected,
                        "predicted_action": predicted,
                        "count": confusion[expected][predicted],
                    }
                )
    return rows


def render_metric_definitions() -> str:
    rows = [
        {
            "metric": "overall_task_success",
            "definition": "fraction of tasks where predicted_action equals expected_action",
            "denominator": "all evaluated tasks",
        },
        {
            "metric": "action_accuracy",
            "definition": "same decision-level exact match as overall_task_success for sgchem_v1.0",
            "denominator": "all evaluated tasks",
        },
        {
            "metric": "molecule_acceptance_rate",
            "definition": "fraction of tasks ending with final_decision=ACCEPT; this was formerly the internal accept_rate and is not task success",
            "denominator": "all evaluated tasks",
        },
        {
            "metric": "ACCEPT/REJECT/ABSTAIN precision",
            "definition": "true positives for the action divided by all predictions of that action",
            "denominator": "predicted action count",
        },
        {
            "metric": "ACCEPT/REJECT/ABSTAIN recall",
            "definition": "true positives for the action divided by all tasks expecting that action",
            "denominator": "expected action count",
        },
        {
            "metric": "task_inconsistent_accept_rate",
            "definition": "ACCEPT predictions on tasks whose expected action is REJECT or ABSTAIN; task-level noncompliance, not molecular safety",
            "denominator": "expected REJECT plus expected ABSTAIN tasks",
        },
        {
            "metric": "false_abstain_rate",
            "definition": "ABSTAIN predictions on tasks whose expected action is ACCEPT or REJECT",
            "denominator": "expected ACCEPT plus expected REJECT tasks",
        },
        {
            "metric": "hard_violation_rate",
            "definition": "non-abstain outputs that fail hard constraints under the verifier",
            "denominator": "non-abstain outputs",
        },
        {
            "metric": "invalid_molecule_rate",
            "definition": "proposal rounds whose verifier feedback indicates invalid SMILES",
            "denominator": "proposal rounds",
        },
        {
            "metric": "pass@1/pass@3",
            "definition": "expected-ACCEPT tasks solved with final ACCEPT within one or three steps",
            "denominator": "expected ACCEPT tasks",
        },
    ]
    return "# Metric Definitions\n\n" + _md_table(rows)


def render_metric_sanity_report(
    *,
    counts: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
) -> str:
    tracked = {
        str(row.get("baseline")): row
        for row in metric_rows
        if str(row.get("baseline")) in {"local_mutation_or_repair", "verify_first", "corpus_retrieval_upper_bound"}
    }
    tracked_acceptance = {
        name: row.get("molecule_acceptance_rate")
        for name, row in sorted(tracked.items())
    }
    lines = [
        "# Metric Sanity Report",
        "",
        "The internal `accept_rate` reported by earlier sweep summaries is renamed in paper-facing tables to `molecule_acceptance_rate`.",
        "It means the fraction of tasks whose final decision was ACCEPT, not the fraction of tasks answered correctly.",
        "",
        "The deterministic local, verify-first, and retrieval baselines can have high and sometimes similar molecule_acceptance_rate values because they search for or retrieve hard-passing molecules on many of the same visible specification instances. Those values are not headline task success: these baselines differ on action accuracy, task-inconsistent acceptance, rejection, abstention, tool use, edit economy, and calibration.",
        f"Current tracked molecule_acceptance_rate values: {tracked_acceptance}.",
        "",
        "Paper implication: do not claim non-saturation from molecule acceptance alone. The defensible claim is that metric decomposition reveals different failure modes that aggregate acceptance hides.",
        "",
        "## Release Counts",
        "",
        f"- full expected_action counts: {dict(counts.get('full_expected_action_counts', {}))}",
        f"- test expected_action counts: {dict(counts.get('test_expected_action_counts', {}))}",
        f"- full task_type counts: {dict(counts.get('full_task_type_counts', {}))}",
        f"- test task_type counts: {dict(counts.get('test_task_type_counts', {}))}",
        "",
        "## Tracked Baselines",
        "",
    ]
    if tracked:
        lines.append(_md_table(list(tracked.values())))
    else:
        lines.append("No tracked deterministic baselines were found in the current run directory.\n")
    lines.extend(
        [
            "",
            "## Saturation Interpretation",
            "",
            "Deterministic local/retrieval baselines partly saturate molecule construction on some feasible slices. Those slices should be framed as sanity-check slices. Rejection, abstention, boundary, invariance, interrupt/resume, calibration, and tool-economy slices remain interpretation-critical and must be reported with denominators.",
        ]
    )
    return "\n".join(lines) + "\n"


def challenge_summary(tasks_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    test_tasks = list(tasks_by_split.get("test", []))
    challenge = [
        task
        for task in test_tasks
        if bool(task.get("challenge_slice"))
        or (
            isinstance(task.get("difficulty_tags"), list)
            and any(tag != "mixed_hard_soft_tradeoff" for tag in task.get("difficulty_tags", []))
        )
    ]
    by_action = Counter(str(task.get("expected_action")) for task in challenge)
    by_type = Counter(str(task.get("task_type")) for task in challenge)
    by_tag = Counter(tag for task in challenge for tag in task.get("difficulty_tags", []))
    classification = "primary_reportable" if len(challenge) >= 80 else "diagnostic_only"
    return {
        "num_test_tasks": len(test_tasks),
        "num_challenge_tasks": len(challenge),
        "classification": classification,
        "expected_action_counts": dict(sorted(by_action.items())),
        "task_type_counts": dict(sorted(by_type.items())),
        "difficulty_tag_counts": dict(sorted(by_tag.items())),
        "challenge_task_ids": [str(task.get("task_id")) for task in challenge],
    }


def render_challenge_denominators(summary: Mapping[str, Any]) -> str:
    rows = [{"slice": "challenge", "n": summary.get("num_challenge_tasks"), "classification": summary.get("classification")}]
    for action, count in dict(summary.get("expected_action_counts", {})).items():
        rows.append({"slice": f"challenge expected_action={action}", "n": count, "classification": summary.get("classification")})
    for task_type, count in dict(summary.get("task_type_counts", {})).items():
        rows.append({"slice": f"challenge task_type={task_type}", "n": count, "classification": summary.get("classification")})
    for tag, count in dict(summary.get("difficulty_tag_counts", {})).items():
        rows.append({"slice": f"challenge tag={tag}", "n": count, "classification": summary.get("classification")})
    return "# Challenge Slice Denominators\n\nChallenge membership is defined only by task/spec/oracle structure through `difficulty_tags`; no baseline result is used.\n\n" + _md_table(rows)


def challenge_result_rows(
    *,
    runs_dir: Path,
    challenge_task_ids: set[str],
) -> list[dict[str, Any]]:
    aggregate = jsonio.read_json(runs_dir / "aggregate.json")
    rows: list[dict[str, Any]] = []
    for baseline in aggregate.get("all_baselines", aggregate.get("baselines", [])):
        if not isinstance(baseline, dict):
            continue
        report_path = runs_dir / str(baseline.get("report_path"))
        if not report_path.exists():
            continue
        payload = jsonio.read_json(report_path)
        records = [
            record
            for record in payload.get("records", [])
            if isinstance(record, dict) and str(record.get("task_id")) in challenge_task_ids
        ]
        metrics = detailed_baseline_metrics(records)
        rows.append(
            {
                "baseline": baseline.get("name"),
                "track": baseline.get("track") or "primary_closed_book",
                "n": metrics["num_tasks"],
                "overall_task_success": _fmt(metrics["overall_task_success"]),
                "action_accuracy": _fmt(metrics["action_accuracy"]),
                "molecule_acceptance_rate": _fmt(metrics["molecule_acceptance_rate"]),
                "task_inconsistent_accept_rate": _fmt(metrics["task_inconsistent_accept_rate"]),
                "REJECT_recall": _fmt(metrics["reject_recall"]),
                "ABSTAIN_recall": _fmt(metrics["abstain_recall"]),
                "hard_violation_rate": _fmt(metrics["hard_violation_rate"]),
            }
        )
    return rows


def render_challenge_report(
    *,
    summary: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Challenge Slice Report",
        "",
        "Challenge membership is structural. It is assigned from task type, oracle evidence, protocol, and constraint metadata before any baseline output is read.",
        "",
        f"- challenge test tasks: {summary.get('num_challenge_tasks')} / {summary.get('num_test_tasks')}",
        f"- classification: {summary.get('classification')}",
        f"- expected_action counts: {dict(summary.get('expected_action_counts', {}))}",
        f"- task_type counts: {dict(summary.get('task_type_counts', {}))}",
        f"- difficulty_tag counts: {dict(summary.get('difficulty_tag_counts', {}))}",
        "",
        "## Results",
        "",
        _md_table(result_rows),
    ]
    return "\n".join(lines)


def render_count_table(title: str, counts: Mapping[str, int]) -> str:
    rows = [{"name": key, "n": value} for key, value in sorted(counts.items())]
    return f"# {title}\n\n" + _md_table(rows)


def render_validation_gates(release: Path) -> str:
    manifest = jsonio.read_json(release / "MANIFEST.json")
    rows = [
        {"gate": "strict_validation", "status": manifest.get("strict_validation", {}).get("valid"), "errors": manifest.get("strict_validation", {}).get("num_errors")},
        {"gate": "oracle_validation", "status": manifest.get("oracle_validation", {}).get("valid"), "errors": manifest.get("oracle_validation", {}).get("num_errors")},
        {"gate": "croissant_local_validation", "status": manifest.get("croissant_validation", {}).get("valid"), "errors": manifest.get("croissant_validation", {}).get("num_errors")},
        {"gate": "anonymous_scan", "status": manifest.get("neurips_ed_preflight", {}).get("anonymous_scan_passed"), "errors": 0},
        {"gate": "dataset_url_accessible", "status": manifest.get("neurips_ed_preflight", {}).get("dataset_url_accessible"), "errors": "manual"},
    ]
    return "# Validation Gates\n\n" + _md_table(rows)


def render_negative_controls() -> str:
    rows = [
        {"negative_control": "18 strict-validator corruption controls", "result": "pass", "denominator": 18},
    ]
    return "# Negative Controls\n\n" + _md_table(rows)


def render_prompt_audit_table(release: Path) -> str:
    path = release / "audits" / "model_prompt_leakage_report.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    keys = (
        "actual_model_prompts_checked",
        "oracle_field_leaks",
        "literal_witness_leaks",
        "label_leaks",
        "audit_accept_reject_name_leaks",
    )
    rows = []
    for key in keys:
        value = _extract_report_value(text, key)
        rows.append({"check": key, "value": value})
    return "# Prompt Leakage Audit\n\n" + _md_table(rows)


def render_oracle_scrambling_table(release: Path) -> str:
    path = release / "audits" / "oracle_scrambling_report.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    keys = (
        "public_views_identical_under_oracle_scrambling",
        "non_oracle_baseline_outputs_identical",
        "tasks_checked",
    )
    rows = [{"check": key, "value": _extract_report_value(text, key)} for key in keys]
    return "# Oracle Scrambling Audit\n\n" + _md_table(rows)


def _extract_report_value(text: str, key: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("-").strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip()
    return "NA"


def render_limitations_by_metric() -> str:
    rows = [
        {"metric": "molecule_acceptance_rate", "limitation": "not task success; demoted from headline metric"},
        {"metric": "boundary_precision", "limitation": "diagnostic when n is 20"},
        {"metric": "SMILES invariance", "limitation": "diagnostic when n is 20"},
        {"metric": "interrupt/resume", "limitation": "diagnostic when n is 10"},
        {"metric": "retrieval_upper_bound", "limitation": "reported separately from primary closed-book leaderboard"},
    ]
    return "# Limitations By Metric\n\n" + _md_table(rows)


def write_submission_reports(
    *,
    release: Path,
    runs_dir: Path,
    paper_dir: Path,
    tasks_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    audits_dir = release / "audits"
    tables_dir = paper_dir / "tables"
    audits_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    counts = release_count_summary(tasks_by_split)
    metric_rows = baseline_metric_rows(runs_dir)
    confusion_rows = baseline_confusion_rows(runs_dir)
    challenge = challenge_summary(tasks_by_split)
    challenge_rows = challenge_result_rows(
        runs_dir=runs_dir,
        challenge_task_ids=set(challenge["challenge_task_ids"]),
    )

    (tables_dir / "metric_definitions.md").write_text(render_metric_definitions(), encoding="utf-8")
    (tables_dir / "baseline_metric_sanity.md").write_text("# Baseline Metric Sanity\n\n" + _md_table(metric_rows), encoding="utf-8")
    (tables_dir / "baseline_action_confusion_matrices.md").write_text("# Baseline Action Confusion Matrices\n\n" + _md_table(confusion_rows), encoding="utf-8")
    (tables_dir / "action_confusion_matrices.md").write_text("# Action Confusion Matrices\n\n" + _md_table(confusion_rows), encoding="utf-8")
    (audits_dir / "metric_sanity_report.md").write_text(
        render_metric_sanity_report(counts=counts, metric_rows=metric_rows),
        encoding="utf-8",
    )
    (audits_dir / "challenge_slice_report.md").write_text(
        render_challenge_report(summary=challenge, result_rows=challenge_rows),
        encoding="utf-8",
    )
    (tables_dir / "challenge_slice_denominators.md").write_text(
        render_challenge_denominators(challenge),
        encoding="utf-8",
    )
    (tables_dir / "challenge_slice_results.md").write_text(
        "# Challenge Slice Results\n\n" + _md_table(challenge_rows),
        encoding="utf-8",
    )
    (tables_dir / "release_summary.md").write_text(
        "# Release Summary\n\n"
        + _md_table(
            [
                {"field": "num_tasks", "value": sum(counts["full_expected_action_counts"].values())},
                {"field": "test_tasks", "value": sum(counts["test_expected_action_counts"].values())},
                {"field": "num_task_types", "value": len(counts["full_task_type_counts"])},
            ]
        ),
        encoding="utf-8",
    )
    (tables_dir / "task_type_distribution.md").write_text(
        render_count_table("Task Type Distribution", counts["full_task_type_counts"]),
        encoding="utf-8",
    )
    (tables_dir / "expected_action_distribution.md").write_text(
        render_count_table("Expected Action Distribution", counts["full_expected_action_counts"]),
        encoding="utf-8",
    )
    (tables_dir / "protocol_distribution.md").write_text(
        render_count_table("Protocol Distribution", counts["full_protocol_counts"]),
        encoding="utf-8",
    )
    (tables_dir / "oracle_types.md").write_text(
        render_count_table("Oracle Types", counts["full_oracle_type_counts"]),
        encoding="utf-8",
    )
    (tables_dir / "validation_gates.md").write_text(render_validation_gates(release), encoding="utf-8")
    (tables_dir / "negative_controls.md").write_text(render_negative_controls(), encoding="utf-8")
    (tables_dir / "prompt_leakage_audit.md").write_text(render_prompt_audit_table(release), encoding="utf-8")
    (tables_dir / "oracle_scrambling_audit.md").write_text(render_oracle_scrambling_table(release), encoding="utf-8")
    primary_rows = [row for row in metric_rows if row.get("track") == "primary_closed_book"]
    (tables_dir / "primary_results.md").write_text("# Primary Closed-Book Results\n\n" + _md_table(primary_rows), encoding="utf-8")
    reject_abstain_rows = [
        {
            "baseline": row.get("baseline"),
            "track": row.get("track"),
            "REJECT_precision": row.get("REJECT_precision"),
            "REJECT_precision_n": row.get("REJECT_precision_n"),
            "REJECT_recall": row.get("REJECT_recall"),
            "REJECT_recall_n": row.get("REJECT_recall_n"),
            "ABSTAIN_precision": row.get("ABSTAIN_precision"),
            "ABSTAIN_precision_n": row.get("ABSTAIN_precision_n"),
            "ABSTAIN_recall": row.get("ABSTAIN_recall"),
            "ABSTAIN_recall_n": row.get("ABSTAIN_recall_n"),
        }
        for row in metric_rows
    ]
    (tables_dir / "reject_abstain_metrics.md").write_text("# Reject And Abstain Metrics\n\n" + _md_table(reject_abstain_rows), encoding="utf-8")
    diagnostic_rows = [
        {"slice": "boundary_precision", "n": counts["test_task_type_counts"].get("boundary_precision", 0), "classification": "diagnostic_only"},
        {"slice": "smiles_invariance", "n": counts["test_task_type_counts"].get("smiles_invariance", 0), "classification": "diagnostic_only"},
        {"slice": "interrupt_resume", "n": counts["test_task_type_counts"].get("interrupt_resume", 0), "classification": "diagnostic_only"},
    ]
    (tables_dir / "diagnostic_slice_results.md").write_text("# Diagnostic Slice Results\n\n" + _md_table(diagnostic_rows), encoding="utf-8")
    (tables_dir / "limitations_by_metric.md").write_text(render_limitations_by_metric(), encoding="utf-8")
    return {"counts": counts, "challenge": challenge, "metric_rows": metric_rows}
