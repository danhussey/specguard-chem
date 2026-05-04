from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


COMMANDS = [
    ["uv", "run", "--extra", "dev", "pytest"],
    [
        "uv",
        "run",
        "specguard-chem",
        "validate-dataset",
        "benchmarks/releases/sgchem_v1.0",
        "--strict",
    ],
    [
        "uv",
        "run",
        "python",
        "scripts/audit_model_prompt_leakage.py",
        "--release",
        "benchmarks/releases/sgchem_v1.0",
    ],
    [
        "uv",
        "run",
        "python",
        "scripts/audit_oracle_scrambling.py",
        "--release",
        "benchmarks/releases/sgchem_v1.0",
    ],
    [
        "uv",
        "run",
        "specguard-chem",
        "run-benchmark",
        "--benchmark",
        "benchmarks/releases/sgchem_v1.0",
        "--split",
        "test",
        "--baselines",
        "baselines/paper_baselines.yaml",
        "--out",
        "/tmp/sgchem_repro_run",
        "--seed",
        "7",
    ],
    [
        "uv",
        "run",
        "specguard-chem",
        "paper-figures",
        "--runs",
        "/tmp/sgchem_repro_run",
        "--out",
        "/tmp/sgchem_repro_paper",
    ],
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("sgchem_v1.0_anonymous_artifact.zip"))
    parser.add_argument("--report", type=Path, default=Path("audits/clean_reviewer_reproduction_report.md"))
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    if not args.archive.exists():
        package = subprocess.run(
            ["uv", "run", "python", "scripts/package_anonymous_artifact.py", "--out", str(args.archive)],
            check=False,
        )
        if package.returncode != 0:
            _write_report(args.report, archive=args.archive, temp_dir=None, rows=[{"command": "package", "returncode": package.returncode}], valid=False)
            return package.returncode

    temp_root = Path(tempfile.mkdtemp(prefix="sgchem_clean_repro_"))
    workdir = temp_root / "artifact"
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        archive.extractall(workdir)

    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(temp_root / ".venv")
    env.setdefault("UV_LINK_MODE", "copy")

    rows: list[dict[str, object]] = []
    valid = True
    for command in COMMANDS:
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rows.append(
            {
                "command": " ".join(command),
                "returncode": completed.returncode,
                "last_output": "\n".join(completed.stdout.splitlines()[-12:]),
            }
        )
        if completed.returncode != 0:
            valid = False
            break

    _write_report(args.report, archive=args.archive, temp_dir=temp_root, rows=rows, valid=valid)
    if not args.keep_temp and valid:
        shutil.rmtree(temp_root, ignore_errors=True)
    return 0 if valid else 1


def _write_report(
    report: Path,
    *,
    archive: Path,
    temp_dir: Path | None,
    rows: list[dict[str, object]],
    valid: bool,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Clean Reviewer Reproduction Report",
        "",
        f"valid: {str(valid).lower()}",
        f"archive: {archive}",
        f"temp_dir: {temp_dir}",
        "",
        "| command | returncode |",
        "| --- | ---: |",
    ]
    for row in rows:
        lines.append(f"| `{row.get('command')}` | {row.get('returncode')} |")
    lines.extend(["", "Command output tails:", ""])
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Command {index}",
                "",
                "```text",
                str(row.get("last_output") or ""),
                "```",
                "",
            ]
        )
    report.write_text("\n".join(lines), encoding="utf-8")
    machine = report.with_suffix(".json")
    machine.write_text(json.dumps({"valid": valid, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
