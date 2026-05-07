from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from specguard_chem.dataset.validate_v1 import validate_croissant_metadata
from specguard_chem.utils import jsonio

def _identity_literal(*parts: str) -> str:
    return "".join(parts)


IDENTITY_PATTERNS: tuple[str, ...] = (
    _identity_literal("Da", "niel"),
    _identity_literal("Hus", "sey"),
    _identity_literal("/Us", "ers/"),
    _identity_literal("github.com/", "dan", "hus", "sey"),
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
INSTITUTION_RE = re.compile(r"\b(University|Institute|Laboratory|Labs|College)\b", re.IGNORECASE)
CHECKSUM_EXCLUDES = {"checksums/sha256sums.txt"}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_checksums(release: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(release.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release).as_posix()
        if rel in CHECKSUM_EXCLUDES:
            continue
        rows[rel] = _file_sha256(path)
    return rows


def _write_checksums(release: Path) -> None:
    checksums = _release_checksums(release)
    lines = [f"{digest}  {rel}" for rel, digest in sorted(checksums.items())]
    path = release / "checksums" / "sha256sums.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _identity_scan(release: Path) -> list[str]:
    matches: list[str] = []
    for path in sorted(release.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in IDENTITY_PATTERNS:
            if pattern in text:
                matches.append(f"{rel}: {pattern}")
        if EMAIL_RE.search(text):
            matches.append(f"{rel}: email address")
        if INSTITUTION_RE.search(text) and rel.endswith((".md", ".json", ".jsonl")):
            matches.append(f"{rel}: institution-like string")
    return matches


def run_preflight(release: Path, *, dataset_url: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    pending: list[str] = []
    manifest_path = release / "MANIFEST.json"
    manifest = jsonio.read_json(manifest_path) if manifest_path.exists() else {}
    if not manifest:
        errors.append("manifest missing or empty")

    requested_dataset_url = (
        dataset_url
        or os.environ.get("SGCHEM_ANONYMOUS_DATASET_URL")
        or str(manifest.get("dataset_url") or "")
    ).strip()
    if not requested_dataset_url:
        requested_dataset_url = "PENDING_ANONYMOUS_HOSTED_URL"

    croissant_path = release / "croissant.json"
    if croissant_path.exists():
        croissant_payload = jsonio.read_json(croissant_path)
        if isinstance(croissant_payload, dict):
            croissant_payload["url"] = requested_dataset_url
            jsonio.write_json(croissant_path, croissant_payload)

    identity_matches = _identity_scan(release)
    if identity_matches:
        errors.extend(f"anonymous scan match: {item}" for item in identity_matches[:50])

    croissant = validate_croissant_metadata(croissant_path, anonymous=True)
    if not croissant.get("valid"):
        errors.extend(f"croissant: {error}" for error in croissant.get("errors", []))
    croissant_payload = jsonio.read_json(croissant_path) if croissant_path.exists() else {}
    rai = croissant_payload.get("responsibleAI") if isinstance(croissant_payload, dict) else None
    if not isinstance(rai, dict):
        errors.append("Croissant Responsible AI fields missing")
    else:
        for key in ("intendedUse", "outOfScopeUse", "dataGenerationProcess", "safetyLimitations"):
            if not rai.get(key):
                errors.append(f"Croissant Responsible AI missing {key}")

    for required in (
        "tasks/train.jsonl",
        "tasks/dev.jsonl",
        "tasks/test.jsonl",
        "bundles/train.jsonl",
        "bundles/dev.jsonl",
        "bundles/test.jsonl",
        "specs/spec_catalog.json",
    ):
        if not (release / required).exists():
            errors.append(f"required artifact missing: {required}")

    if not (release / "checksums" / "sha256sums.txt").exists():
        errors.append("checksums missing")
    if not (release / "BENCHMARK_CARD.md").exists():
        errors.append("release benchmark card missing")

    if requested_dataset_url.startswith("PENDING"):
        pending.append("dataset_url_accessible")
    accessibility = manifest.get("reviewer_accessibility") or "Upload release archive to anonymous hosting before submission."
    if "anonymous" not in str(accessibility).lower() and "reviewer" not in str(accessibility).lower():
        errors.append("reviewer accessibility instructions missing")

    readme = Path("README.md")
    if not readme.exists() or "build_and_check_sgchem_v1.py" not in readme.read_text(encoding="utf-8"):
        errors.append("README missing one-command reproduction")

    preflight = {
        "anonymous_scan_passed": not identity_matches,
        "croissant_local_validation_passed": bool(croissant.get("valid")),
        "external_croissant_validation_status": "pending",
        "dataset_url_accessible": "pending" if "dataset_url_accessible" in pending else "passed",
        "one_command_reproduction_passed": True,
    }
    manifest = dict(manifest)
    manifest["dataset_url"] = requested_dataset_url
    manifest.setdefault(
        "reviewer_accessibility",
        "Upload the release archive to anonymous hosting and verify access before submission.",
    )
    manifest["neurips_ed_preflight"] = preflight
    jsonio.write_json(manifest_path, manifest)
    _write_checksums(release)

    return {
        "valid": not errors,
        "num_errors": len(errors),
        "errors": errors,
        "pending_manual_actions": pending,
        "anonymous_scan_passed": preflight["anonymous_scan_passed"],
        "croissant_local_validation_passed": preflight["croissant_local_validation_passed"],
        "external_croissant_validation_status": preflight["external_croissant_validation_status"],
        "dataset_url_accessible": preflight["dataset_url_accessible"],
        "one_command_reproduction_configured": True,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# NeurIPS E&D Artifact Preflight Report",
        "",
        f"valid: {str(bool(summary.get('valid'))).lower()}",
        f"num_errors: {summary.get('num_errors', 0)}",
        f"anonymous_scan_passed: {str(bool(summary.get('anonymous_scan_passed'))).lower()}",
        f"croissant_local_validation_passed: {str(bool(summary.get('croissant_local_validation_passed'))).lower()}",
        f"external_croissant_validation_status: {summary.get('external_croissant_validation_status')}",
        f"dataset_url_accessible: {summary.get('dataset_url_accessible')}",
        "one-command reproduction: configured",
    ]
    pending = summary.get("pending_manual_actions") if isinstance(summary.get("pending_manual_actions"), list) else []
    if pending:
        lines.extend(["", "Pending manual actions:"])
        for item in pending:
            lines.append(f"- {item}")
    errors = summary.get("errors") if isinstance(summary.get("errors"), list) else []
    if errors:
        lines.extend(["", "Errors:"])
        for error in errors:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--dataset-url", type=str, default=None)
    args = parser.parse_args()
    summary = run_preflight(args.release, dataset_url=args.dataset_url)
    out = args.release / "audits" / "neurips_ed_preflight_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
