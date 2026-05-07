from __future__ import annotations

import argparse
from pathlib import Path

from specguard_chem.benchmark.release import load_benchmark_release
from specguard_chem.runner.runner import TaskRunner
from specguard_chem.scoring import reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("benchmarks/releases/sgchem_v1.0"))
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", choices=("L1", "L2", "L3"), default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    release = load_benchmark_release(args.release)
    tasks = release.load_split_tasks(args.split)
    if args.protocol:
        tasks = [task for task in tasks if task.protocol == args.protocol]
    run_dir = args.out / args.name
    runner = TaskRunner(args.model, seed=args.seed)
    runner.run_tasks(
        tasks,
        run_dir=run_dir,
        suite=f"{release.benchmark_id}_{args.split}",
        protocol=args.protocol or "mixed",
        spec_loader=release.spec_loader,
    )
    records = reports.load_trace(run_dir)
    summary = reports.summarise(records)
    reports.write_report(run_dir, records=records, summary=summary)
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
