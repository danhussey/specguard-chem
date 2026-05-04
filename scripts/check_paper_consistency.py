from __future__ import annotations

import argparse
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
        "ABSTAIN=120",
        "ACCEPT=426",
        "REJECT=142",
        "abstain_contradiction=120",
        "audit_accept=120",
        "audit_reject=120",
        "boundary_precision=44",
        "construct_feasible=120",
        "interrupt_resume=27",
        "repair_multi_violation=30",
        "repair_near_miss=36",
        "smiles_invariance=44",
        "tool_forced_l3=27",
    ]
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
        "unsafe_accept_rate.md",
        "reject_abstain_metrics.md",
        "challenge_slice_denominators.md",
        "challenge_slice_results.md",
    ):
        _require((tables / required).exists(), f"missing paper table {required}", errors)
    unsafe_text = _read(tables / "unsafe_accept_rate.md")
    _require("unsafe_accept_n" in unsafe_text, "unsafe_accept_rate table missing unsafe_accept denominator", errors)
    _require("correct_reject_n" in unsafe_text, "unsafe_accept_rate table missing reject denominator", errors)
    _require("correct_abstain_n" in unsafe_text, "unsafe_accept_rate table missing abstain denominator", errors)


def _check_baseline_metrics(errors: list[str]) -> None:
    runs = Path("runs/paper_sweeps/sgchem_v1.0_test")
    if not (runs / "aggregate.json").exists():
        errors.append("missing current aggregate.json for baseline metric comparison")
        return
    rows = baseline_metric_rows(runs)
    expected = {
        str(row["baseline"]): row
        for row in rows
        if row.get("baseline") in {"local_mutation_or_repair", "verify_first", "corpus_retrieval_upper_bound"}
    }
    for name, row in expected.items():
        _require(row.get("molecule_acceptance_rate") == "0.852", f"{name} molecule_acceptance_rate is not 0.852", errors)
        _require(row.get("overall_task_success") == "0.664", f"{name} overall_task_success is not 0.664", errors)
        _require(row.get("REJECT_recall") == "0.000", f"{name} REJECT_recall is not 0.000", errors)
        _require(row.get("ABSTAIN_recall") == "0.000", f"{name} ABSTAIN_recall is not 0.000", errors)
        _require(row.get("unsafe_accept_rate") == "0.561", f"{name} unsafe_accept_rate is not 0.561", errors)


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
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
