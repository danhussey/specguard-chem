from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from specguard_chem.audits.submission import baseline_metric_rows
from specguard_chem.utils import jsonio


RELEASE = Path("benchmarks/releases/sgchem_v1.0")
PAPER_DIRS = (Path("paper"), Path("paper_v1"))

BANNED_PHRASES = (
    "drug discovery benchmark",
    "predicts toxicity",
    "evaluates toxicity",
    "synthesis planning benchmark",
    "therapeutic efficacy",
    "clinical utility",
    "lead optimization benchmark",
    "proves tool use improves performance",
    "non-saturated benchmark",
    "biological activity benchmark",
)
NONCLAIM_CUES = (
    "does not evaluate",
    "does not make",
    "not evaluate",
    "not a",
    "non-claim",
    "non-claims",
    "out-of-scope",
    "out of scope",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _paper_texts() -> list[tuple[Path, str]]:
    texts: list[tuple[Path, str]] = []
    for root in PAPER_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".tex", ".yaml", ".yml"}:
                texts.append((path, path.read_text(encoding="utf-8")))
    return texts


def _parse_count_table(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in _read(path).splitlines():
        if not line.startswith("|") or "---" in line or "| name |" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) >= 2 and parts[1].isdigit():
            counts[parts[0]] = int(parts[1])
    return counts


def _parse_release_summary(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in _read(path).splitlines():
        if not line.startswith("|") or "---" in line or "| field |" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) >= 2 and parts[1].isdigit():
            counts[parts[0]] = int(parts[1])
    return counts


def _parse_markdown_rows(path: Path) -> list[dict[str, str]]:
    lines = [line for line in _read(path).splitlines() if line.startswith("|")]
    if len(lines) < 3:
        return []
    header = [part.strip() for part in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        if "---" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _line_is_nonclaim(line: str) -> bool:
    lower = line.lower()
    return any(cue in lower for cue in NONCLAIM_CUES)


def _check_banned_phrases(errors: list[str], warnings: list[str]) -> None:
    for path, text in _paper_texts():
        for line_no, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            for phrase in BANNED_PHRASES:
                if phrase not in lower:
                    continue
                if phrase in {"therapeutic efficacy", "clinical utility"} and _line_is_nonclaim(line):
                    continue
                errors.append(f"{path}:{line_no}: banned or overclaim phrase `{phrase}`")


def _check_manifest_numbers(manifest: dict[str, Any], errors: list[str]) -> None:
    paper_text = "\n".join(text for _, text in _paper_texts())
    required_snippets = [
        f"{manifest['num_bundles']} bundles",
        f"{manifest['num_tasks']} tasks",
        f"{manifest['splits']['test']['tasks']} test tasks",
    ]
    required_snippets.extend(
        f"{action}={count}"
        for action, count in sorted((manifest.get("tasks_per_expected_action") or {}).items())
    )
    required_snippets.extend(
        f"{task_type}={count}"
        for task_type, count in sorted((manifest.get("tasks_per_task_type") or {}).items())
    )
    for snippet in required_snippets:
        _require(snippet in paper_text, f"paper text missing required release number `{snippet}`", errors)


def _check_tables(manifest: dict[str, Any], errors: list[str]) -> None:
    tables = Path("paper_v1") / "tables"
    release_summary = _parse_release_summary(tables / "release_summary.md")
    _require(release_summary.get("num_tasks") == manifest.get("num_tasks"), "release_summary num_tasks mismatch", errors)
    _require(
        release_summary.get("test_tasks") == manifest.get("splits", {}).get("test", {}).get("tasks"),
        "release_summary test_tasks mismatch",
        errors,
    )
    expected = _parse_count_table(tables / "expected_action_distribution.md")
    _require(expected == manifest.get("tasks_per_expected_action"), "expected_action_distribution mismatch", errors)
    task_types = _parse_count_table(tables / "task_type_distribution.md")
    _require(task_types == manifest.get("tasks_per_task_type"), "task_type_distribution mismatch", errors)
    for required in (
        "metric_definitions.md",
        "baseline_metric_sanity.md",
        "primary_results.md",
        "task_inconsistent_accept_rate.md",
        "reject_abstain_metrics.md",
        "challenge_slice_denominators.md",
        "challenge_slice_results.md",
        "wrapper_saturation.md",
    ):
        _require((tables / required).exists(), f"missing paper table {required}", errors)
    inconsistent_text = _read(tables / "task_inconsistent_accept_rate.md")
    _require(
        "task_inconsistent_accept_n" in inconsistent_text,
        "task_inconsistent_accept_rate table missing denominator",
        errors,
    )
    _require("correct_reject_n" in inconsistent_text, "task_inconsistent_accept_rate table missing reject denominator", errors)
    _require("correct_abstain_n" in inconsistent_text, "task_inconsistent_accept_rate table missing abstain denominator", errors)


def _check_baseline_metrics(errors: list[str]) -> None:
    runs = Path("runs/paper_sweeps/sgchem_v1.0_test")
    if not (runs / "aggregate.json").exists():
        errors.append("missing current aggregate.json for baseline metric comparison")
        return
    rows = baseline_metric_rows(runs)
    current = {
        str(row["baseline"]): row
        for row in rows
        if row.get("baseline") in {"local_mutation_or_repair", "verify_first", "corpus_retrieval_upper_bound"}
    }
    table_rows = {
        str(row.get("baseline")): row
        for row in _parse_markdown_rows(Path("paper_v1") / "tables" / "baseline_metric_sanity.md")
    }
    fields = (
        "num_tasks",
        "overall_task_success",
        "action_accuracy",
        "molecule_acceptance_rate",
        "REJECT_recall",
        "ABSTAIN_recall",
        "task_inconsistent_accept_rate",
    )
    for name, row in current.items():
        table_row = table_rows.get(name)
        _require(table_row is not None, f"baseline_metric_sanity missing row for {name}", errors)
        if table_row is None:
            continue
        for field in fields:
            _require(str(table_row.get(field)) == str(row.get(field)), f"{name} {field} table value is stale", errors)


def _check_final_url(manifest: dict[str, Any], mode: str, errors: list[str]) -> None:
    url = str(manifest.get("dataset_url") or "")
    status = str(manifest.get("neurips_ed_preflight", {}).get("dataset_url_accessible") or "")
    if mode == "final":
        _require(url.startswith(("http://", "https://")) and not url.startswith("PENDING"), "final mode requires hosted dataset URL", errors)
        _require(status == "passed", "final mode requires dataset_url_accessible=passed", errors)


def _check_no_local_paths(errors: list[str]) -> None:
    local_path_marker = "".join(("/", "Users", "/"))
    for path, text in _paper_texts():
        if local_path_marker in text:
            errors.append(f"{path}: contains local absolute path")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rc", "final"), default="rc")
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    if args.results is not None:
        return _check_paper_v2(args.release, args.results)

    errors: list[str] = []
    warnings: list[str] = []
    manifest = jsonio.read_json(args.release / "MANIFEST.json")
    _check_manifest_numbers(manifest, errors)
    _check_tables(manifest, errors)
    _check_baseline_metrics(errors)
    _check_final_url(manifest, args.mode, errors)
    _check_no_local_paths(errors)
    _check_banned_phrases(errors, warnings)

    print(
        json.dumps(
            {
                "valid": not errors,
                "mode": args.mode,
                "num_errors": len(errors),
                "num_warnings": len(warnings),
                "errors": errors,
                "warnings": warnings,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float | None:
    if value in {None, "", "NA", "nan"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(a: Any, b: Any, *, tol: float = 5e-4) -> bool:
    left = _as_float(a)
    right = _as_float(b)
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= tol


def _load_release_tasks(release: Path) -> dict[str, list[dict[str, Any]]]:
    tasks: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        path = release / "tasks" / f"{split}.jsonl"
        tasks[split] = jsonio.read_jsonl(path) if path.exists() else []
    return tasks


def _expected_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("expected_action") or "")
        counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items()))


def _check_paper_v2(release: Path, results: Path) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    tasks = _load_release_tasks(release)
    test_tasks = tasks.get("test", [])
    require(len(test_tasks) == 266, f"test denominator is {len(test_tasks)}, expected 266")
    release_expected = _expected_counts(test_tasks)

    normalized_path = results / "summaries" / "normalized_task_results.jsonl"
    normalized = jsonio.read_jsonl(normalized_path) if normalized_path.exists() else []
    test_normalized = [
        row
        for row in normalized
        if row.get("split") == "test" and row.get("adapter") == "always_accept"
    ]
    require(len(test_normalized) == 266, "normalized test denominator for always_accept is not 266")
    normalized_expected = _expected_counts(test_normalized)
    require(
        normalized_expected == release_expected,
        f"expected-action counts mismatch: normalized={normalized_expected}, release={release_expected}",
    )

    table = _read_csv_rows(results / "tables" / "full_offline_baseline_matrix_test.csv")
    normalized_metrics = _read_csv_rows(results / "summaries" / "normalized_run_metrics.csv")
    metrics_by_key = {
        (row.get("split"), row.get("adapter"), row.get("protocol")): row
        for row in normalized_metrics
        if row.get("split") == "test"
    }
    for row in table:
        key = (row.get("split"), row.get("adapter"), row.get("protocol"))
        source = metrics_by_key.get(key)
        require(source is not None, f"missing normalized metrics for {key}")
        if not source:
            continue
        for metric in (
            "action_accuracy",
            "molecule_acceptance_rate",
            "task_inconsistent_accept_rate",
            "reject_recall",
            "abstain_recall",
            "schema_error_rate",
        ):
            require(
                _close(row.get(metric), source.get(metric)),
                f"{key} {metric} table value does not match normalized metrics",
            )

    wrapper_rows = [row for row in table if row.get("adapter") == "well_engineered_wrapper"]
    require(wrapper_rows, "missing well_engineered_wrapper row in full offline test table")
    for row in wrapper_rows:
        require(
            row.get("access_model") == "verifier/search wrapper",
            "wrapper row is mixed into closed-book leaderboard",
        )

    external_rows = _read_csv_rows(results / "tables" / "external_diagnostic_snapshot.csv")
    if external_rows:
        for row in external_rows:
            cache_mode = str(row.get("cache_mode") or "").lower()
            notes = str(row.get("notes") or "").lower()
            require(
                "skipped" in cache_mode or "diagnostic" in notes or row.get("adapter") in {"openai_chat", "openai_chat_verify_l3", "process"},
                "external diagnostics are not labelled diagnostic/skipped",
            )
    else:
        warnings.append("external diagnostic snapshot table is empty")

    validation_logs = {
        "validate_dataset_strict.log": "strict validation",
        "model_prompt_leakage_audit.log": "prompt leakage audit",
        "oracle_scrambling_audit.log": "oracle scrambling audit",
    }
    for name, label in validation_logs.items():
        require((results / "validation" / name).exists(), f"missing {label} log")

    hidden_log = (results / "validation" / "model_prompt_leakage_audit.log")
    if hidden_log.exists():
        text = hidden_log.read_text(encoding="utf-8", errors="replace").lower()
        require(
            "oracle_field_leaks: 0" in text or '"oracle_field_leaks": 0' in text,
            "model prompt leakage audit does not report zero oracle field leaks",
        )
        require(
            "literal_witness_leaks: 0" in text or '"literal_witness_leaks": 0' in text,
            "model prompt leakage audit does not report zero literal witness leaks",
        )

    figure_sources = {
        "per_family_action_accuracy_heatmap": "per_family_metrics_test.csv",
        "wrapper_ablation_budget_curve": "wrapper_ablation_test.csv",
        "metric_rank_shift": "metric_ranking_sensitivity.csv",
        "protocol_ladder_action_accuracy": "protocol_ladder_test.csv",
        "wrapper_ablation_action_accuracy": "wrapper_ablation_test.csv",
    }
    for stem, source in figure_sources.items():
        for suffix in (".pdf", ".png"):
            require((results / "figures" / f"{stem}{suffix}").exists(), f"missing figure {stem}{suffix}")
        require((results / "tables" / source).exists(), f"missing source CSV {source} for figure {stem}")

    print(
        json.dumps(
            {
                "valid": not errors,
                "mode": "paper_v2",
                "num_errors": len(errors),
                "num_warnings": len(warnings),
                "errors": errors,
                "warnings": warnings,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
