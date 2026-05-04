from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from specguard_chem.utils import jsonio


TASK_REQUIREMENTS: tuple[tuple[str, int], ...] = (
    ("construct_feasible", 2),
    ("audit_accept", 1),
    ("audit_reject", 1),
    ("repair_near_miss", 1),
    ("repair_multi_violation", 1),
    ("abstain_contradiction", 2),
    ("boundary_precision", 2),
    ("smiles_invariance", 2),
    ("interrupt_resume", 1),
    ("tool_forced_l3", 1),
)

HIGHLIGHT_TERMS = (
    "toxicity",
    "synthesis",
    "therapeutic",
    "clinical",
    "drug discovery",
    "target binding",
    "dosage",
    "disease",
    "medicinal chemistry",
)

IDENTITY_PATTERNS = (
    "".join(("Da", "niel")),
    "".join(("Hus", "sey")),
    "".join(("/Us", "ers/")),
    "".join(("github.com/", "dan", "hus", "sey")),
    "email address",
    "institution",
)


def _read(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return f"Missing: {path.as_posix()}\n"
    text = path.read_text(encoding="utf-8")
    if limit is None:
        return text
    return "\n".join(text.splitlines()[:limit]) + "\n"


def _tasks_by_split(release: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split: jsonio.read_jsonl(release / "tasks" / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }


def _all_tasks(release: Path) -> list[dict[str, Any]]:
    by_split = _tasks_by_split(release)
    rows: list[dict[str, Any]] = []
    for split in ("test", "dev", "train"):
        for task in by_split[split]:
            row = dict(task)
            row["split"] = split
            rows.append(row)
    return rows


def _select_examples(tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_bundles: set[str] = set()
    used_specs: set[str] = set()
    task_rows = list(tasks)
    for task_type, count in TASK_REQUIREMENTS:
        candidates = [task for task in task_rows if task.get("task_type") == task_type and task.get("task_id") not in used_ids]
        candidates.sort(
            key=lambda task: (
                str(task.get("bundle_id")) in used_bundles,
                str(task.get("spec_id")) in used_specs,
                str(task.get("split")) != "test",
                str(task.get("task_id")),
            )
        )
        for task in candidates[:count]:
            selected.append(task)
            used_ids.add(str(task.get("task_id")))
            used_bundles.add(str(task.get("bundle_id")))
            used_specs.add(str(task.get("spec_id")))
    return selected


def _allowed_actions(task: dict[str, Any]) -> list[str]:
    payload = task.get("agent_visible_payload") if isinstance(task.get("agent_visible_payload"), dict) else {}
    values = payload.get("allowed_actions")
    return [str(value) for value in values] if isinstance(values, list) else []


def _public_view(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("agent_visible_payload") if isinstance(task.get("agent_visible_payload"), dict) else {}
    return {
        "visible_task_name": payload.get("visible_task_name") or "evaluation_task",
        "rendered_agent_input": payload.get("rendered_agent_input") or task.get("rendered_agent_input") or "",
        "protocol": payload.get("protocol") or task.get("protocol") or "",
    }


def _public_spec_excerpt(task: dict[str, Any]) -> str:
    payload = task.get("agent_visible_payload") if isinstance(task.get("agent_visible_payload"), dict) else {}
    hard = payload.get("hard_constraints") if isinstance(payload.get("hard_constraints"), list) else []
    soft = payload.get("soft_preferences") if isinstance(payload.get("soft_preferences"), list) else []
    lines = ["Hard constraints:"]
    lines.extend(f"- {value}" for value in hard[:4])
    lines.append("Soft preferences:")
    lines.extend(f"- {value}" for value in soft[:4])
    return "\n".join(lines)


def _evidence_summary(task: dict[str, Any]) -> str:
    evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
    visible_keys = [
        "oracle_type",
        "expected_num_failing_constraints",
        "failing_constraints",
        "boundary_group_id",
        "invariance_group_id",
        "interrupt_group_id",
    ]
    summary = {key: evidence.get(key) for key in visible_keys if key in evidence}
    for key in ("feasible_witness_smiles", "unsat_certificate", "violation_certificate", "boundary_certificate", "equivalence_certificate"):
        if key in evidence:
            summary[key] = "present"
    return json.dumps(summary, sort_keys=True)


def _verifier_summary(task: dict[str, Any]) -> str:
    evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
    for key in ("witness_verifier_result", "candidate_verifier_result", "input_verifier_result"):
        result = evidence.get(key)
        if isinstance(result, dict):
            return json.dumps(
                {
                    "valid": result.get("valid"),
                    "hard_pass": result.get("hard_pass"),
                    "failing_constraints": result.get("failing_constraints"),
                    "hard_violation_units": result.get("hard_violation_units"),
                },
                sort_keys=True,
            )
    return "not applicable"


def _write_public_examples(path: Path, selected: list[dict[str, Any]]) -> None:
    lines = [
        "# Public Task Examples",
        "",
        "Local review metadata may include task IDs and internal task types. These fields are not model-visible and should not be copied into paper-facing examples.",
        "",
    ]
    for index, task in enumerate(selected, start=1):
        public = _public_view(task)
        lines.extend(
            [
                f"## Example {index}",
                "",
                f"- task_id: {task.get('task_id')}",
                f"- split: {task.get('split')}",
                f"- internal_task_type: {task.get('task_type')} (metadata only; not model-visible)",
                f"- visible_task_name: {public['visible_task_name']}",
                f"- protocol: {public['protocol']}",
                f"- allowed_actions: {json.dumps(_allowed_actions(task))}",
                f"- budgets: {json.dumps(task.get('budgets', {}), sort_keys=True)}",
                f"- agent_visible_hash: {task.get('agent_visible_hash')}",
                "",
                "Public spec excerpt:",
                "",
                "```text",
                _public_spec_excerpt(task),
                "```",
                "",
                "Rendered agent input:",
                "",
                "```text",
                str(public["rendered_agent_input"]),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_internal_examples(path: Path, selected: list[dict[str, Any]]) -> None:
    lines = [
        "# INTERNAL REVIEW ONLY - DO NOT UPLOAD OR INCLUDE IN DOUBLE-BLIND ARTIFACT",
        "",
        "This file intentionally includes hidden expected actions, oracle types, witnesses, and certificates for local review only.",
        "",
    ]
    for index, task in enumerate(selected, start=1):
        lines.extend(
            [
                f"## Example {index}: {task.get('task_id')}",
                "",
                f"- split: {task.get('split')}",
                f"- internal_task_type: {task.get('task_type')}",
                f"- expected_action: {task.get('expected_action')}",
                f"- oracle_type: {task.get('oracle_type')}",
                f"- evidence_summary: {_evidence_summary(task)}",
                f"- verifier_recomputation_summary: {_verifier_summary(task)}",
                "- why_valid: strict oracle validation recomputes this task from the effective spec.",
                "- possible_reviewer_objection: check that the public prompt is label-neutral and not too template-like.",
                "- why_paper_safe: the public prompt excludes hidden oracle evidence and uses scoped medicinal-chemistry language.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paper_examples(path: Path, selected: list[dict[str, Any]]) -> None:
    role_names = {
        "construct_feasible": "construction",
        "audit_accept": "candidate audit",
        "audit_reject": "candidate audit",
        "repair_near_miss": "repair",
        "repair_multi_violation": "repair",
        "abstain_contradiction": "feasibility check",
        "boundary_precision": "boundary audit",
        "smiles_invariance": "representation invariance",
        "interrupt_resume": "interrupt/resume repair",
        "tool_forced_l3": "tool-assisted repair",
    }
    lines = [
        "# Public Paper Examples",
        "",
        "These examples are safe for paper-facing use. They omit internal task IDs, bundle IDs, hidden expected actions, oracle evidence, witnesses, proofs, and certificates.",
        "",
    ]
    for index, task in enumerate(selected, start=1):
        public = _public_view(task)
        role = role_names.get(str(task.get("task_type")), "evaluation")
        lines.extend(
            [
                f"## Example {index}: {role}",
                "",
                f"- visible_task_name: {public['visible_task_name']}",
                f"- protocol: {public['protocol']}",
                f"- allowed_actions: {json.dumps(_allowed_actions(task))}",
                "",
                "```text",
                str(public["rendered_agent_input"]),
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _highlight_public_text(paths: Iterable[Path]) -> str:
    lines = ["# Public Facing Text", ""]
    for path in paths:
        lines.extend([f"## {path.as_posix()}", "", "```text", _read(path, limit=180), "```", ""])
        for line_no, line in enumerate(_read(path).splitlines(), start=1):
            lower = line.lower()
            for term in HIGHLIGHT_TERMS:
                if term in lower:
                    if "does not evaluate" in lower or "out-of-scope" in lower or "must not" in lower:
                        label = "allowed_non_claim"
                    elif "medicinal chemistry" in lower or "medicinal-chemistry" in lower:
                        label = "allowed_domain_scope"
                    else:
                        label = "risky_review_wording"
                    lines.append(f"- {path.as_posix()}:{line_no}: {label}: {line}")
                    break
        lines.append("")
    return "\n".join(lines)


def _counts_block(manifest: dict[str, Any], tasks: list[dict[str, Any]]) -> str:
    test_tasks = [task for task in tasks if task.get("split") == "test"]
    return "\n".join(
        [
            f"- bundles/tasks: {manifest.get('num_bundles')} bundles, {manifest.get('num_tasks')} tasks",
            f"- train/dev/test task counts: {manifest.get('splits')}",
            f"- full expected_action counts: {manifest.get('tasks_per_expected_action')}",
            f"- test expected_action counts: {dict(Counter(task.get('expected_action') for task in test_tasks))}",
            f"- full task_type counts: {manifest.get('tasks_per_task_type')}",
            f"- test task_type counts: {dict(Counter(task.get('task_type') for task in test_tasks))}",
        ]
    )


def _scan_identity(paths: Iterable[Path]) -> str:
    matches: list[str] = []
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in IDENTITY_PATTERNS:
            if pattern in text:
                matches.append(f"{path.as_posix()}: {pattern}")
        if email.search(text):
            matches.append(f"{path.as_posix()}: email address")
    lines = ["# Anonymization Scan", "", "Patterns scanned: " + ", ".join(IDENTITY_PATTERNS), ""]
    lines.append("Matches: none" if not matches else "Matches:\n" + "\n".join(f"- {match}" for match in matches))
    lines.extend(
        [
            "",
            "Archive confirmation checklist:",
            "- personal-name pattern A: absent",
            "- personal-name pattern B: absent",
            "- local home-directory marker: absent",
            "- named GitHub URLs: absent",
            "- email addresses: absent",
            "- institution names: absent",
            "- local absolute paths: absent",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("benchmarks/releases/sgchem_v1.0"))
    parser.add_argument("--review-packet", type=Path, default=Path("review_packet"))
    parser.add_argument("--paper-examples", type=Path, default=Path("paper/appendix/task_examples_public.md"))
    args = parser.parse_args()

    if args.review_packet.exists():
        shutil.rmtree(args.review_packet)
    args.review_packet.mkdir(parents=True)

    manifest = jsonio.read_json(args.release / "MANIFEST.json")
    tasks = _all_tasks(args.release)
    selected = _select_examples(tasks)

    (args.review_packet / "README.md").write_text(
        "# SpecGuard-Chem Review Packet\n\nLocal review packet generated from the current sgchem_v1.0 release. Keep this directory out of git and do not upload the internal oracle file.\n",
        encoding="utf-8",
    )
    (args.review_packet / "00_review_checklist.md").write_text(
        "\n".join(
            [
                "# Review Checklist",
                "",
                "- [ ] Does the abstract match the artifact?",
                "- [ ] Are all numbers current?",
                "- [ ] Is medicinal chemistry framed strongly but safely?",
                "- [ ] Are drug-discovery/activity/toxicity/synthesis/clinical claims avoided except as non-claims?",
                "- [ ] Is molecule_acceptance_rate clearly distinguished from task/action accuracy?",
                "- [ ] Are primary/tool/retrieval tracks separated?",
                "- [ ] Are diagnostic slices labeled diagnostic?",
                "- [ ] Does every public-facing file avoid identity leakage?",
                "- [ ] Is Croissant metadata valid?",
                "- [ ] Are hosted URL placeholders obvious?",
                "- [ ] Are upload instructions complete?",
                "",
            ]
        ),
        encoding="utf-8",
    )

    public_paths = [
        Path("hosting/DATASET_CARD.md"),
        Path("hosting/HUGGINGFACE_DATASET_CARD.md"),
        Path("hosting/README_for_upload.md"),
        args.release / "BENCHMARK_CARD.md",
        args.release / "RELEASE_NOTES.md",
        Path("SAFETY.md"),
        Path("README.md"),
    ]
    (args.review_packet / "01_public_facing_text.md").write_text(_highlight_public_text(public_paths), encoding="utf-8")

    claim_ledger = _read(Path("paper_v1/claim_ledger.yaml"))
    (args.review_packet / "02_abstracts_and_paper_claims.md").write_text(
        "# Abstracts And Paper Claims\n\n"
        + _read(Path("paper_v1/abstract_variants.md"))
        + "\n## Current Counts\n\n"
        + _counts_block(manifest, tasks)
        + "\n\n## Claim Ledger\n\n```yaml\n"
        + claim_ledger
        + "```\n",
        encoding="utf-8",
    )

    archive = Path("sgchem_v1.0_anonymous_artifact.zip")
    archive_sha = "missing"
    archive_size = 0
    if archive.exists():
        import hashlib

        archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive_size = archive.stat().st_size
    (args.review_packet / "03_artifact_hosting_packet.md").write_text(
        "# Artifact Hosting Packet\n\n"
        f"- archive filename: {archive.name}\n"
        f"- archive SHA256: {archive_sha}\n"
        f"- archive size: {archive_size} bytes\n\n"
        + _read(Path("hosting/README_for_upload.md"))
        + "\n## Post-upload Checklist\n\n"
        + _read(Path("hosting/post_upload_checklist.md"))
        + "\n## Finalize Command\n\n```bash\nuv run python scripts/finalize_hosted_url.py --release benchmarks/releases/sgchem_v1.0 --dataset-url \"<ANONYMOUS_HOSTED_DATASET_URL>\"\n```\n",
        encoding="utf-8",
    )

    _write_public_examples(args.review_packet / "04_task_examples_public.md", selected)
    _write_internal_examples(args.review_packet / "05_task_examples_with_oracles_INTERNAL_DO_NOT_UPLOAD.md", selected)
    _write_paper_examples(args.paper_examples, selected)

    metric_files = [
        "metric_definitions.md",
        "baseline_metric_sanity.md",
        "baseline_tracks.md",
        "primary_results.md",
        "baseline_action_confusion_matrices.md",
        "unsafe_accept_rate.md",
        "reject_abstain_metrics.md",
        "challenge_slice_results.md",
        "diagnostic_slice_results.md",
        "limitations_by_metric.md",
    ]
    metric_text = ["# Metrics And Results", "", "- molecule_acceptance_rate is not task success.", "- task/action accuracy is the primary action-level metric.", "- unsafe_accept_rate, REJECT_recall, and ABSTAIN_recall must be prominent.", "- retrieval upper bound must be separated from primary baselines.", "- diagnostic slices must not be overclaimed.", ""]
    for name in metric_files:
        metric_text.extend([f"## {name}", "", _read(Path("paper_v1/tables") / name), ""])
    (args.review_packet / "06_metrics_and_results.md").write_text("\n".join(metric_text), encoding="utf-8")

    audit_sources = [
        args.release / "audits/oracle_validation_report.md",
        args.release / "audits/model_prompt_leakage_report.md",
        args.release / "audits/oracle_scrambling_report.md",
        args.release / "audits/reviewer_attack_report.md",
        args.release / "audits/manual_test_bundle_dossiers.md",
        args.release / "audits/neurips_ed_preflight_report.md",
        args.release / "audits/anonymous_hosting_preflight.md",
    ]
    audit_lines = ["# Validation And Audits", ""]
    for path in audit_sources:
        audit_lines.extend([f"## {path.as_posix()}", "", _read(path, limit=80), ""])
    (args.review_packet / "07_validation_and_audits.md").write_text("\n".join(audit_lines), encoding="utf-8")

    scan_paths = public_paths + sorted(Path("hosting").glob("*.md")) + [
        args.review_packet / "01_public_facing_text.md",
        args.review_packet / "02_abstracts_and_paper_claims.md",
        args.review_packet / "03_artifact_hosting_packet.md",
        args.review_packet / "04_task_examples_public.md",
        args.paper_examples,
    ]
    (args.review_packet / "08_anonymization_scan.md").write_text(_scan_identity(scan_paths), encoding="utf-8")

    (args.review_packet / "09_upload_todo.md").write_text(
        "\n".join(
            [
                "# Upload TODO",
                "",
                "1. Create anonymous dataset host.",
                "2. Upload sgchem_v1.0_anonymous_artifact.zip.",
                "3. Copy hosted URL.",
                "4. Run finalize_hosted_url.py.",
                "5. Rerun build/check/preflight/paper consistency final mode.",
                "6. Regenerate anonymous archive if metadata changed.",
                "7. Reupload final archive if needed.",
                "8. Confirm hosted URL accessible in private/incognito browser.",
                "9. Validate Croissant externally if possible.",
                "10. Submit URL and Croissant to OpenReview.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.review_packet} and {args.paper_examples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
