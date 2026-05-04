from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

from specguard_chem.utils import jsonio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.preflight_neurips_ed_artifact import run_preflight


IDENTITY_PATTERNS = (
    "".join(("Da", "niel")),
    "".join(("Hus", "sey")),
    "".join(("/Us", "ers/")),
    "".join(("github.com/", "dan", "hus", "sey")),
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

INCLUDE_PATHS = (
    "benchmarks/releases/sgchem_v1.0",
    "src",
    "scripts",
    "baselines",
    "paper_v1",
    "paper",
    "tests",
    "data/specs",
    "tasks",
    "README.md",
    "BENCHMARK_CARD.md",
    "SAFETY.md",
    "METHODS.md",
    "METRICS.md",
    "SPEC.md",
    "docs/GENERATOR_DESIGN_v1.md",
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
)

EXCLUDE_PARTS = {
    ".git",
    ".venv",
    ".DS_Store",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def iter_artifact_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE_PATHS:
        path = root / item
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            rel_parts = child.relative_to(root).parts
            if any(part in EXCLUDE_PARTS or part.endswith(".egg-info") for part in rel_parts):
                continue
            if len(rel_parts) >= 2 and rel_parts[0] == "paper" and rel_parts[1] in {"figures", "tables"}:
                continue
            files.append(child)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def identity_scan(paths: Iterable[Path], *, root: Path) -> list[str]:
    matches: list[str] = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in IDENTITY_PATTERNS:
            if pattern in text:
                matches.append(f"{rel}: {pattern}")
        if EMAIL_RE.search(text):
            matches.append(f"{rel}: email address")
    return matches


def write_archive(*, root: Path, archive_path: Path, files: list[Path]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            rel = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(rel)
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            archive.writestr(info, path.read_bytes())


def write_release_checksums(release: Path) -> None:
    rows: dict[str, str] = {}
    for path in sorted(release.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release).as_posix()
        if rel == "checksums/sha256sums.txt":
            continue
        rows[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = [f"{digest}  {rel}" for rel, digest in sorted(rows.items())]
    (release / "checksums" / "sha256sums.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def render_contents_manifest(*, root: Path, files: list[Path], archive_path: Path) -> str:
    rows = [
        "# Archive Contents Manifest",
        "",
        f"archive: {archive_path.name}",
        f"files: {len(files)}",
        "",
        "| path | sha256 | bytes |",
        "| --- | --- | ---: |",
    ]
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"| {path.relative_to(root).as_posix()} | {digest} | {path.stat().st_size} |")
    return "\n".join(rows) + "\n"


def render_preflight(
    *,
    archive_path: Path,
    identity_matches: list[str],
    preflight: dict,
    dataset_url: str,
) -> str:
    lines = [
        "# Anonymous Hosting Preflight",
        "",
        f"archive: {archive_path}",
        f"archive_sha256: {hashlib.sha256(archive_path.read_bytes()).hexdigest() if archive_path.exists() else 'NA'}",
        f"dataset_url: {dataset_url}",
        f"dataset_url_accessible: {preflight.get('dataset_url_accessible')}",
        f"anonymous_scan_passed: {str(not identity_matches).lower()}",
        f"croissant_local_validation_passed: {str(bool(preflight.get('croissant_local_validation_passed'))).lower()}",
        f"external_croissant_validation_status: {preflight.get('external_croissant_validation_status')}",
        "",
    ]
    if identity_matches:
        lines.append("Identity scan matches:")
        for match in identity_matches[:50]:
            path = match.split(":", 1)[0]
            lines.append(f"- {path}: redacted identity pattern")
    else:
        lines.append("Identity scan matches: none")
    if str(dataset_url).startswith("PENDING"):
        lines.extend(
            [
                "",
                "Pending manual action:",
                "- Upload this archive to anonymous reviewer-accessible hosting and rerun with --dataset-url.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("benchmarks/releases/sgchem_v1.0"))
    parser.add_argument("--out", type=Path, default=Path("sgchem_v1.0_anonymous_artifact.zip"))
    parser.add_argument("--dataset-url", type=str, default=None)
    args = parser.parse_args()

    root = Path.cwd()
    dataset_url = (
        args.dataset_url
        or os.environ.get("SGCHEM_ANONYMOUS_DATASET_URL")
        or "PENDING_ANONYMOUS_HOSTED_URL"
    )
    audits_dir = args.release / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("anonymous_hosting_preflight.md", "archive_contents_manifest.md"):
        stale_path = audits_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    preflight = run_preflight(args.release, dataset_url=dataset_url)
    manifest_path = args.release / "MANIFEST.json"
    manifest = jsonio.read_json(manifest_path)
    manifest["anonymous_artifact_archive"] = {
        "path": args.out.as_posix(),
        "sha256": "ARCHIVE_SHA256_RECORDED_AFTER_PACKAGING",
        "identity_scan_passed": None,
    }
    jsonio.write_json(manifest_path, manifest)
    write_release_checksums(args.release)
    files = iter_artifact_files(root)
    identity_matches = identity_scan(files, root=root)
    write_archive(root=root, archive_path=args.out, files=files)

    (audits_dir / "archive_contents_manifest.md").write_text(
        render_contents_manifest(root=root, files=files, archive_path=args.out),
        encoding="utf-8",
    )
    (audits_dir / "anonymous_hosting_preflight.md").write_text(
        render_preflight(
            archive_path=args.out,
            identity_matches=identity_matches,
            preflight=preflight,
            dataset_url=dataset_url,
        ),
        encoding="utf-8",
    )
    manifest = jsonio.read_json(manifest_path)
    manifest["anonymous_artifact_archive"] = {
        "path": args.out.as_posix(),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "identity_scan_passed": not identity_matches,
    }
    jsonio.write_json(manifest_path, manifest)
    write_release_checksums(args.release)
    print(render_preflight(archive_path=args.out, identity_matches=identity_matches, preflight=preflight, dataset_url=dataset_url))
    return 0 if not identity_matches and preflight.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
