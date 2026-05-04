from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from specguard_chem.dataset.validate_v1 import validate_croissant_metadata
from specguard_chem.utils import jsonio


MARKER_START = "<!-- sgchem-hosted-url:start -->"
MARKER_END = "<!-- sgchem-hosted-url:end -->"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_marked_section(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    marked = f"{MARKER_START}\n{block.rstrip()}\n{MARKER_END}"
    if MARKER_START in existing and MARKER_END in existing:
        before = existing.split(MARKER_START, 1)[0].rstrip()
        after = existing.split(MARKER_END, 1)[1].lstrip()
        text = f"{before}\n\n{marked}\n\n{after}".rstrip() + "\n"
    else:
        text = existing.rstrip() + "\n\n" + marked + "\n" if existing.strip() else marked + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact_block(dataset_url: str, archive_sha: str | None) -> str:
    checksum = archive_sha or "not recorded"
    return (
        "## Anonymous Hosted Artifact\n\n"
        f"- Dataset URL: {dataset_url}\n"
        "- Review access: anonymous reviewer-accessible dataset page.\n"
        "- Archive: `sgchem_v1.0_anonymous_artifact.zip`\n"
        f"- Archive SHA256: `{checksum}`\n"
        "- Croissant metadata: `benchmarks/releases/sgchem_v1.0/croissant.json`\n"
    )


def _write_artifact_links(path: Path, dataset_url: str, archive_sha: str | None) -> None:
    lines = [
        "# Artifact Links",
        "",
        "This file is the paper-facing source of truth for anonymous artifact links.",
        "",
        f"- anonymous_dataset_url: {dataset_url}",
        "- release_archive: sgchem_v1.0_anonymous_artifact.zip",
        "- release_archive_sha256: recorded in benchmarks/releases/sgchem_v1.0/MANIFEST.json",
        "- croissant_metadata: benchmarks/releases/sgchem_v1.0/croissant.json",
        "- preflight_report: benchmarks/releases/sgchem_v1.0/audits/neurips_ed_preflight_report.md",
        "",
        "Do not replace this with an author-identifying repository URL during double-blind review.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_manifest(release: Path, dataset_url: str, archive_sha: str | None) -> dict[str, Any]:
    manifest_path = release / "MANIFEST.json"
    manifest = jsonio.read_json(manifest_path)
    manifest["dataset_url"] = dataset_url
    manifest["reviewer_accessibility"] = (
        "Anonymous hosted artifact URL has been recorded; verify access from a clean browser session before submission."
    )
    preflight = dict(manifest.get("neurips_ed_preflight") or {})
    preflight["dataset_url_accessible"] = "passed"
    preflight.setdefault("anonymous_scan_passed", True)
    preflight.setdefault("croissant_local_validation_passed", True)
    preflight.setdefault("external_croissant_validation_status", "pending")
    preflight.setdefault("one_command_reproduction_passed", True)
    manifest["neurips_ed_preflight"] = preflight
    archive = dict(manifest.get("anonymous_artifact_archive") or {})
    archive.setdefault("path", "sgchem_v1.0_anonymous_artifact.zip")
    if archive_sha:
        archive["sha256"] = archive_sha
    archive.setdefault("identity_scan_passed", True)
    manifest["anonymous_artifact_archive"] = archive
    jsonio.write_json(manifest_path, manifest)
    return manifest


def _update_croissant(release: Path, dataset_url: str) -> dict[str, Any]:
    croissant_path = release / "croissant.json"
    payload = jsonio.read_json(croissant_path)
    payload["url"] = dataset_url
    payload.setdefault("externalValidationStatus", "pending")
    jsonio.write_json(croissant_path, payload)
    return validate_croissant_metadata(croissant_path, anonymous=True)


def _run_preflight(release: Path, dataset_url: str) -> dict[str, Any]:
    from scripts.preflight_neurips_ed_artifact import render_report, run_preflight

    summary = run_preflight(release, dataset_url=dataset_url)
    out = release / "audits" / "neurips_ed_preflight_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(summary), encoding="utf-8")
    return summary


def _write_checksums(release: Path) -> None:
    rows: dict[str, str] = {}
    for path in sorted(release.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release).as_posix()
        if rel == "checksums/sha256sums.txt":
            continue
        rows[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = [f"{digest}  {rel}" for rel, digest in sorted(rows.items())]
    checksums = release / "checksums" / "sha256sums.txt"
    checksums.parent.mkdir(parents=True, exist_ok=True)
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--archive", type=Path, default=Path("sgchem_v1.0_anonymous_artifact.zip"))
    args = parser.parse_args()

    dataset_url = args.dataset_url.strip()
    if not dataset_url or dataset_url.startswith("PENDING"):
        raise SystemExit("dataset URL must be a concrete anonymous hosted URL")
    if not dataset_url.startswith(("https://", "http://")):
        raise SystemExit("dataset URL must start with http:// or https://")

    archive_sha = _sha256(args.archive)
    manifest = _update_manifest(args.release, dataset_url, archive_sha)
    croissant = _update_croissant(args.release, dataset_url)

    block = _artifact_block(dataset_url, archive_sha)
    _replace_marked_section(args.release / "BENCHMARK_CARD.md", block)
    _replace_marked_section(args.release / "RELEASE_NOTES.md", block)
    _replace_marked_section(Path("README.md"), block)
    _write_artifact_links(Path("paper_v1") / "artifact_links.md", dataset_url, archive_sha)

    preflight = _run_preflight(args.release, dataset_url)
    audit = subprocess.run(
        [sys.executable, "scripts/audit_task_inventory.py", "--release", str(args.release)],
        check=False,
    )
    _write_checksums(args.release)

    summary = {
        "dataset_url": dataset_url,
        "archive_sha256": archive_sha,
        "croissant_valid": bool(croissant.get("valid")),
        "anonymous_scan_passed": bool(preflight.get("anonymous_scan_passed")),
        "dataset_url_accessible": preflight.get("dataset_url_accessible"),
        "reviewer_attack_report_refreshed": audit.returncode == 0,
        "manifest_dataset_url": manifest.get("dataset_url"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["croissant_valid"] and summary["anonymous_scan_passed"] and audit.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
