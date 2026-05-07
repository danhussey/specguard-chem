from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from specguard_chem.audits.submission import write_submission_reports
from specguard_chem.dataset.validate_v1 import load_release_tasks_by_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--paper", type=Path, default=Path("paper_v1"))
    args = parser.parse_args()

    tasks_by_split = load_release_tasks_by_split(args.release)
    summary = write_submission_reports(
        release=args.release,
        runs_dir=args.runs,
        paper_dir=args.paper,
        tasks_by_split=tasks_by_split,
    )
    _write_checksums(args.release)
    challenge = summary.get("challenge", {})
    print(
        "metric sanity complete: "
        f"challenge_tasks={challenge.get('num_challenge_tasks')} "
        f"classification={challenge.get('classification')}"
    )
    return 0


def _write_checksums(release: Path) -> None:
    rows = {}
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


if __name__ == "__main__":
    raise SystemExit(main())
