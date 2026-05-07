from __future__ import annotations

"""Reality-check runs for sgchem_v1.0 without regenerating benchmark tasks."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from specguard_chem.benchmark.release import load_benchmark_release
from specguard_chem.runner.runner import TaskRunner
from specguard_chem.scoring import reports
from specguard_chem.utils import jsonio
from specguard_chem.verifiers import canonicalize_smiles


DEFAULT_RELEASE = Path("benchmarks/releases/sgchem_v1.0")
DEFAULT_OUT = Path("runs/reality_check/sgchem_v1.0")


def _expected_action(record: Mapping[str, Any]) -> str:
    value = str(record.get("expected_action") or "ACCEPT").upper()
    return value if value in {"ACCEPT", "REJECT", "ABSTAIN"} else "ACCEPT"


def _final_decision(record: Mapping[str, Any]) -> str:
    if record.get("schema_error") or record.get("invalid_action") or record.get("invalid_tool_call"):
        return "INVALID"
    value = str(record.get("final_decision") or "").upper()
    return value if value in {"ACCEPT", "REJECT", "ABSTAIN", "INVALID"} else "INVALID"


def _predicted_action(record: Mapping[str, Any]) -> str:
    rounds = record.get("rounds")
    if isinstance(rounds, list):
        for round_item in reversed(rounds):
            if not isinstance(round_item, Mapping):
                continue
            value = str(round_item.get("declared_public_action") or "").upper()
            if value in {"ACCEPT", "REJECT", "ABSTAIN"}:
                return value
    return _final_decision(record)


def _safe_div(num: int | float, den: int | float) -> float | None:
    return None if not den else float(num) / float(den)


def _family_success(records: list[dict[str, Any]], family_prefixes: tuple[str, ...]) -> dict[str, Any]:
    subset = [
        record
        for record in records
        if str(record.get("task_family") or "").startswith(family_prefixes)
    ]
    correct = sum(1 for record in subset if _expected_action(record) == _final_decision(record))
    return {"n": len(subset), "success": _safe_div(correct, len(subset))}


def direct_feedback_audit(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        expected = _expected_action(record)
        final_decision = _predicted_action(record)
        final_canonical = canonicalize_smiles(record.get("final_smiles") or "")
        failed_tool_candidates: set[str] = set()
        for round_item in record.get("rounds") or []:
            if not isinstance(round_item, dict):
                continue
            evaluation = round_item.get("evaluation")
            smiles = round_item.get("smiles")
            if not isinstance(evaluation, dict) or not smiles:
                continue
            if round_item.get("action") == "tool_call" and evaluation.get("hard_pass") is False:
                canonical = canonicalize_smiles(str(smiles))
                if canonical:
                    failed_tool_candidates.add(canonical)
            if evaluation.get("hard_pass") is False:
                hard_fails = evaluation.get("hard_fails")
                if not isinstance(hard_fails, list):
                    hard_fails = []
                if "invalid_smiles" in hard_fails:
                    counts["schema_or_parse_error"] += 1
        if final_decision == "ACCEPT" and final_canonical in failed_tool_candidates:
            counts["accepted_same_candidate_after_failed_verifier"] += 1
        if expected == "REJECT" and final_decision == "ACCEPT":
            input_canonical = None
            for round_item in record.get("rounds") or []:
                if isinstance(round_item, dict) and round_item.get("smiles"):
                    input_canonical = canonicalize_smiles(str(round_item.get("smiles")))
                    break
            if final_canonical and input_canonical and final_canonical != input_canonical:
                counts["substituted_passing_molecule_for_audit"] += 1
        if expected == "ABSTAIN" and final_decision == "ACCEPT":
            counts["accepted_on_contradictory_spec"] += 1
        if expected != "ACCEPT" and final_decision == "ACCEPT" and bool(record.get("hard_pass")):
            counts["wrong_action_but_valid_molecule"] += 1
        if record.get("schema_error") or record.get("invalid_action") or record.get("invalid_tool_call"):
            counts["schema_or_parse_error"] += 1
    for key in (
        "accepted_same_candidate_after_failed_verifier",
        "substituted_passing_molecule_for_audit",
        "accepted_on_contradictory_spec",
        "wrong_action_but_valid_molecule",
        "schema_or_parse_error",
    ):
        counts.setdefault(key, 0)
    return dict(counts)


def reality_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [_expected_action(record) for record in records]
    predicted = [_predicted_action(record) for record in records]
    final_decisions = [_final_decision(record) for record in records]
    confusion = {
        action: {pred: 0 for pred in ("ACCEPT", "REJECT", "ABSTAIN", "INVALID")}
        for action in ("ACCEPT", "REJECT", "ABSTAIN")
    }
    for exp, pred in zip(expected, predicted):
        confusion[exp][pred] += 1
    reject_n = sum(confusion["REJECT"].values())
    abstain_n = sum(confusion["ABSTAIN"].values())
    inconsistent_n = confusion["REJECT"]["ACCEPT"] + confusion["ABSTAIN"]["ACCEPT"]
    output = {
        "num_tasks": len(records),
        "action_accuracy": _safe_div(sum(int(exp == pred) for exp, pred in zip(expected, predicted)), len(records)),
        "molecule_acceptance_rate": _safe_div(sum(int(pred == "ACCEPT") for pred in final_decisions), len(records)),
        "task_inconsistent_accept_rate": _safe_div(inconsistent_n, reject_n + abstain_n),
        "task_inconsistent_accept_n": reject_n + abstain_n,
        "REJECT_recall": _safe_div(confusion["REJECT"]["REJECT"], reject_n),
        "REJECT_recall_n": reject_n,
        "ABSTAIN_recall": _safe_div(confusion["ABSTAIN"]["ABSTAIN"], abstain_n),
        "ABSTAIN_recall_n": abstain_n,
        "construct_success": _family_success(records, ("construct_feasible",)),
        "repair_success": _family_success(records, ("repair_", "interrupt_resume", "tool_forced_l3")),
        "boundary_success": _family_success(records, ("boundary_precision",)),
        "invariance_success": _family_success(records, ("smiles_invariance",)),
        "confusion": confusion,
    }
    output.update(direct_feedback_audit(records))
    return output


def select_reality_subset(tasks: list[Any], *, accept_count: int = 40) -> list[Any]:
    reject = [task for task in tasks if task.expected_action == "REJECT"]
    abstain = [task for task in tasks if task.expected_action == "ABSTAIN"]
    accept = [task for task in tasks if task.expected_action == "ACCEPT"]
    by_type: dict[str, list[Any]] = {}
    for task in accept:
        by_type.setdefault(str(task.task_type), []).append(task)
    selected_accept: list[Any] = []
    while len(selected_accept) < accept_count and any(by_type.values()):
        for task_type in sorted(by_type):
            if by_type[task_type] and len(selected_accept) < accept_count:
                selected_accept.append(by_type[task_type].pop(0))
    selected = sorted(reject + abstain + selected_accept, key=lambda task: task.task_id)
    return selected


def limit_reality_subset(tasks: list[Any], *, max_tasks: int) -> list[Any]:
    if max_tasks <= 0 or len(tasks) <= max_tasks:
        return list(tasks)
    by_action: dict[str, list[Any]] = {}
    for task in sorted(tasks, key=lambda item: (str(item.expected_action), str(item.task_type), item.task_id)):
        by_action.setdefault(str(task.expected_action), []).append(task)
    selected: list[Any] = []
    while len(selected) < max_tasks and any(by_action.values()):
        for action in ("REJECT", "ABSTAIN", "ACCEPT"):
            if by_action.get(action) and len(selected) < max_tasks:
                selected.append(by_action[action].pop(0))
    return sorted(selected, key=lambda task: task.task_id)


def run_one(
    *,
    name: str,
    model: str,
    tasks: list[Any],
    release,
    out_dir: Path,
    seed: int,
    adapter_kwargs: dict[str, Any] | None = None,
    allow_external: bool = False,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = out_dir / name
    runner = TaskRunner(
        model,
        seed=seed,
        adapter_kwargs=adapter_kwargs or {},
        allow_external=allow_external,
        cache_dir=(cache_dir / name if cache_dir else None),
    )
    runner.run_tasks(
        tasks,
        run_dir=run_dir,
        suite=f"{release.benchmark_id}_reality_check",
        protocol="mixed",
        spec_loader=release.spec_loader,
    )
    records = reports.load_trace(run_dir)
    summary = reports.summarise(records)
    reports.write_report(run_dir, records=records, summary=summary)
    metrics = reality_metrics(records)
    jsonio.write_json(run_dir / "reality_metrics.json", metrics)
    return {
        "name": name,
        "model": model,
        "num_tasks": len(records),
        "metrics": metrics,
        "report_path": str((run_dir / "report.json").relative_to(out_dir)),
    }


def _merge_runs(existing: Mapping[str, Any] | None, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if existing:
        for row in existing.get("runs", []):
            if isinstance(row, dict) and row.get("name"):
                merged[str(row["name"])] = dict(row)
    for row in runs:
        if row.get("name"):
            merged[str(row["name"])] = row
    return list(merged.values())


def _merge_external_status(
    existing: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> dict[str, Any]:
    if not existing:
        return dict(current)
    previous = existing.get("external_status")
    if not isinstance(previous, Mapping):
        return dict(current)
    merged = dict(previous)
    if current.get("actual_llm_runs_executed"):
        merged["actual_llm_runs_executed"] = True
        merged["num_external_runs_executed"] = int(
            merged.get("num_external_runs_executed") or 0
        ) + int(current.get("num_external_runs_executed") or 0)
        merged["reason"] = current.get("reason")
    elif not merged.get("actual_llm_runs_executed"):
        merged.update(current)
    return merged


def render_memo(payload: Mapping[str, Any]) -> str:
    rows = []
    for row in payload.get("runs", []):
        metrics = row.get("metrics", {})
        rows.append(
            "| {name} | {n} | {action:.3f} | {accept:.3f} | {inconsistent:.3f} | {reject:.3f} | {abstain:.3f} | {same} | {sub} | {contra} |".format(
                name=row.get("name"),
                n=row.get("num_tasks"),
                action=metrics.get("action_accuracy") or 0.0,
                accept=metrics.get("molecule_acceptance_rate") or 0.0,
                inconsistent=metrics.get("task_inconsistent_accept_rate") or 0.0,
                reject=metrics.get("REJECT_recall") or 0.0,
                abstain=metrics.get("ABSTAIN_recall") or 0.0,
                same=metrics.get("accepted_same_candidate_after_failed_verifier", 0),
                sub=metrics.get("substituted_passing_molecule_for_audit", 0),
                contra=metrics.get("accepted_on_contradictory_spec", 0),
            )
        )
    external_status = payload.get("external_status", {})
    external_runs = [
        row
        for row in payload.get("runs", [])
        if isinstance(row, Mapping) and str(row.get("name", "")).startswith("openai_")
    ]
    all_external_abstain = bool(external_runs) and all(
        (row.get("metrics", {}) or {}).get("molecule_acceptance_rate") == 0.0
        and (row.get("metrics", {}) or {}).get("ABSTAIN_recall") == 1.0
        for row in external_runs
    )
    wrapper = next((row for row in payload.get("runs", []) if row.get("name") == "well_engineered_wrapper_full"), None)
    wrapper_metrics = (wrapper or {}).get("metrics", {})
    recommendation = "reframe"
    rationale = (
        "The deterministic wrapper has high action accuracy and perfect REJECT/ABSTAIN recall "
        "when it uses the public verifier/spec interface. This suggests the artifact is best "
        "framed as an oracle-first compiler and harness for testing public/private isolation, "
        "task semantics, and wrapper-vs-agent behavior, not as standalone evidence that raw LLM "
        "agents need this benchmark."
    )
    if external_status.get("actual_llm_runs_executed"):
        rationale += (
            " Actual LLM runs were executed; in the current snapshot the observed "
            "failure mode is dominated by action-policy behavior rather than rich "
            "molecular search."
        )
    else:
        rationale += " Actual LLM runs were not executed in this offline pass, so agent-capability claims remain unsupported."
    if all_external_abstain:
        llm_failure_answer = (
            "The successful OpenAI runs failed in a simple but paper-relevant way: "
            "they collapsed to ABSTAIN across the stratified subset, producing zero "
            "molecule acceptance, zero REJECT recall, and perfect ABSTAIN recall. "
            "That is an action-policy failure rather than evidence of robust "
            "constructive specification compliance."
        )
        schema_answer = (
            "Mostly yes for the current OpenAI snapshot: the dominant failure is "
            "blanket abstention/action policy, not hidden-oracle leakage or verifier "
            "misuse."
        )
    elif external_runs:
        llm_failure_answer = (
            "Actual LLM runs were executed; inspect the action-level metrics above "
            "rather than molecule acceptance alone."
        )
        schema_answer = (
            "Partly. The current report separates action-policy errors, verifier "
            "feedback errors, and schema/parse errors so this can be judged directly."
        )
    else:
        llm_failure_answer = (
            "Not determined in this pass; no successful external LLM runs are recorded."
        )
        schema_answer = (
            "The deterministic wrapper avoids schema failures, so deterministic "
            "failures are semantic/search related rather than schema-only."
        )
    lines = [
        "# Reality Check Decision Memo",
        "",
        "## Question",
        "",
        "Does sgchem_v1.0 support a compelling NeurIPS E&D paper, or is it mostly an over-engineered deterministic rule puzzle?",
        "",
        "## Experiment Status",
        "",
        f"- subset policy: {payload.get('subset_policy')}",
        f"- subset size: {payload.get('subset_size')}",
        f"- full-test size run: {payload.get('full_test_size')}",
        f"- external model status: {external_status.get('reason', 'not recorded')}",
        "",
        "## Results",
        "",
        "| run | n | action_accuracy | molecule_acceptance_rate | task_inconsistent_accept_rate | REJECT_recall | ABSTAIN_recall | accepted_same_candidate_after_failed_verifier | substituted_passing_molecule_for_audit | accepted_on_contradictory_spec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Answers",
        "",
        f"- Do actual LLM agents fail in interesting ways? {llm_failure_answer}",
        f"- Are failures mostly trivial prompt/schema issues? {schema_answer}",
        "- Does a deterministic wrapper solve the task? It is the strongest current baseline; see table above.",
        "- Is the benchmark measuring LLM capability or wrapper engineering? Current evidence strongly measures wrapper engineering and harness correctness. Actual LLM runs, when available, should be treated as adapter/protocol sanity checks unless richer failure modes appear.",
        f"- Is the main result worth submitting? Recommendation: {recommendation}.",
        "- Should we reframe, revise, or pivot? Reframe as an artifact/compiler and evaluation harness; do not claim raw-agent nontriviality until live model data supports it.",
        "",
        "## Recommendation",
        "",
        f"{recommendation.upper()}: {rationale}",
    ]
    if wrapper_metrics:
        lines.extend(
            [
                "",
                "## Wrapper Baseline Interpretation",
                "",
                f"- action_accuracy: {wrapper_metrics.get('action_accuracy')}",
                f"- molecule_acceptance_rate: {wrapper_metrics.get('molecule_acceptance_rate')}",
                f"- task_inconsistent_accept_rate: {wrapper_metrics.get('task_inconsistent_accept_rate')}",
                f"- REJECT_recall: {wrapper_metrics.get('REJECT_recall')}",
                f"- ABSTAIN_recall: {wrapper_metrics.get('ABSTAIN_recall')}",
            ]
        )
    return "\n".join(lines) + "\n"


def render_wrapper_table(payload: Mapping[str, Any]) -> str:
    rows = [
        "| run | track | n | action_accuracy | molecule_acceptance_rate | task_inconsistent_accept_rate | REJECT_recall | ABSTAIN_recall | interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("runs", []):
        if not isinstance(row, Mapping):
            continue
        if row.get("name") != "well_engineered_wrapper_full":
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        rows.append(
            "| {name} | wrapper_guarded | {n} | {action:.3f} | {accept:.3f} | {inconsistent:.3f} | {reject:.3f} | {abstain:.3f} | Saturates sgchem_v1.0 by implementing the public specification/verifier contract; report as an evaluation-validity ceiling, not as a model leaderboard result. |".format(
                name=row.get("name"),
                n=row.get("num_tasks"),
                action=metrics.get("action_accuracy") or 0.0,
                accept=metrics.get("molecule_acceptance_rate") or 0.0,
                inconsistent=metrics.get("task_inconsistent_accept_rate") or 0.0,
                reject=metrics.get("REJECT_recall") or 0.0,
                abstain=metrics.get("ABSTAIN_recall") or 0.0,
            )
        )
    if len(rows) == 2:
        rows.append("| none | wrapper_guarded | 0 | NA | NA | NA | NA | NA | Run `scripts/run_reality_check_experiments.py` without `--skip-wrapper` to populate this table. |")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--accept-count", type=int, default=40)
    parser.add_argument(
        "--max-subset-tasks",
        type=int,
        default=0,
        help="Optional cap for smoke runs; keeps a deterministic action-stratified subset.",
    )
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--run-external", action="store_true")
    parser.add_argument("--frontier-openai-model", default="gpt-4o")
    parser.add_argument("--cheap-openai-model", default="gpt-4o-mini")
    parser.add_argument(
        "--external-filter",
        action="append",
        default=[],
        help="Run only external variants whose run name contains this substring. May be repeated.",
    )
    parser.add_argument(
        "--no-merge-existing",
        action="store_true",
        help="Do not preserve existing reality-check runs when writing the summary.",
    )
    parser.add_argument(
        "--skip-wrapper",
        action="store_true",
        help="Skip deterministic wrapper reruns and rely on merged existing wrapper outputs.",
    )
    args = parser.parse_args()

    release = load_benchmark_release(args.release)
    test_tasks = release.load_split_tasks("test")
    subset_tasks = select_reality_subset(test_tasks, accept_count=args.accept_count)
    subset_tasks = limit_reality_subset(subset_tasks, max_tasks=args.max_subset_tasks)
    args.out.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    cache_dir = args.out / "cache"
    if not args.skip_wrapper:
        runs.append(
            run_one(
                name="well_engineered_wrapper_subset",
                model="well_engineered_wrapper",
                tasks=subset_tasks,
                release=release,
                out_dir=args.out,
                seed=args.seed,
            )
        )
        runs.append(
            run_one(
                name="well_engineered_wrapper_full",
                model="well_engineered_wrapper",
                tasks=test_tasks,
                release=release,
                out_dir=args.out,
                seed=args.seed,
            )
        )

    external_status = {
        "actual_llm_runs_executed": False,
        "reason": "External LLM calls were not run. The repository is configured for offline operation, and sending unpublished benchmark prompts to a third-party API requires explicit risk acceptance.",
    }
    if args.run_external:
        if not args.allow_external:
            external_status["reason"] = "--run-external was supplied without --allow-external."
        else:
            external_runs = [
                ("openai_frontier_no_tool_subset", "openai_chat", {"model": args.frontier_openai_model, "temperature": 0.0, "policy": "no_tools", "timeout": 45.0}),
                ("openai_frontier_tool_available_subset", "openai_chat", {"model": args.frontier_openai_model, "temperature": 0.0, "timeout": 45.0}),
                ("openai_frontier_forced_verify_first_subset", "openai_chat_verify_l3", {"model": args.frontier_openai_model, "temperature": 0.0, "timeout": 45.0}),
                ("openai_cheap_no_tool_subset", "openai_chat", {"model": args.cheap_openai_model, "temperature": 0.0, "policy": "no_tools", "timeout": 45.0}),
                ("openai_cheap_tool_available_subset", "openai_chat", {"model": args.cheap_openai_model, "temperature": 0.0, "timeout": 45.0}),
                ("openai_cheap_forced_verify_first_subset", "openai_chat_verify_l3", {"model": args.cheap_openai_model, "temperature": 0.0, "timeout": 45.0}),
            ]
            filters = [str(value) for value in args.external_filter if str(value)]
            if filters:
                external_runs = [
                    item
                    for item in external_runs
                    if any(filter_value in item[0] for filter_value in filters)
                ]
            executed = 0
            for name, model, kwargs in external_runs:
                error_path = args.out / f"{name}_ERROR.txt"
                if error_path.exists():
                    error_path.unlink()
                try:
                    runs.append(
                        run_one(
                            name=name,
                            model=model,
                            tasks=subset_tasks,
                            release=release,
                            out_dir=args.out,
                            seed=args.seed,
                            adapter_kwargs=kwargs,
                            allow_external=True,
                            cache_dir=cache_dir,
                        )
                    )
                    executed += 1
                except Exception as exc:
                    error_path.write_text(str(exc) + "\n", encoding="utf-8")
            external_status = {
                "actual_llm_runs_executed": executed > 0,
                "num_external_runs_executed": executed,
                "reason": (
                    "External runs requested with --allow-external; see per-run "
                    "directories/cache for successful runs and *_ERROR.txt files for failures."
                ),
            }

    existing_payload = None
    existing_summary_path = args.out / "reality_check_summary.json"
    if existing_summary_path.exists() and not args.no_merge_existing:
        try:
            existing_payload = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_payload = None
    runs = _merge_runs(existing_payload, runs) if not args.no_merge_existing else runs
    external_status = (
        _merge_external_status(existing_payload, external_status)
        if not args.no_merge_existing
        else external_status
    )

    representative_subset_size = len(subset_tasks)
    if existing_payload and not args.no_merge_existing:
        old_size = existing_payload.get("subset_size")
        if isinstance(old_size, int) and old_size > representative_subset_size:
            representative_subset_size = old_size

    payload = {
        "benchmark_id": release.benchmark_id,
        "seed": args.seed,
        "subset_policy": "all REJECT test tasks + all ABSTAIN test tasks + stratified ACCEPT tasks",
        "subset_size": representative_subset_size,
        "current_run_subset_size": len(subset_tasks),
        "full_test_size": len(test_tasks),
        "subset_expected_action_counts": dict(Counter(str(task.expected_action) for task in subset_tasks)),
        "full_expected_action_counts": dict(Counter(str(task.expected_action) for task in test_tasks)),
        "runs": runs,
        "external_status": external_status,
    }
    jsonio.write_json(args.out / "reality_check_summary.json", payload)
    memo = render_memo(payload)
    (Path("paper_v1") / "reality_check_decision_memo.md").write_text(memo, encoding="utf-8")
    tables_dir = Path("paper_v1") / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "wrapper_saturation.md").write_text(
        render_wrapper_table(payload),
        encoding="utf-8",
    )
    (args.out / "reality_check_decision_memo.md").write_text(memo, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
