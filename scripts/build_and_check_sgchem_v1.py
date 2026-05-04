from __future__ import annotations

"""Build and validate sgchem_v1.0 end to end."""

import subprocess
import sys


COMMANDS = [
    ["uv", "run", "pytest"],
    [
        "uv",
        "run",
        "specguard-chem",
        "compile-benchmark",
        "--benchmark-id",
        "sgchem_v1.0",
        "--out",
        "benchmarks/releases/sgchem_v1.0",
        "--seed",
        "7",
        "--target-bundles",
        "120",
        "--min-tasks",
        "650",
        "--max-tasks",
        "900",
        "--min-test-task-type-count",
        "construct_feasible=25",
        "--min-test-task-type-count",
        "audit_accept=25",
        "--min-test-task-type-count",
        "audit_reject=25",
        "--min-test-task-type-count",
        "abstain_contradiction=25",
        "--min-test-task-type-count",
        "repair_near_miss=20",
        "--min-test-task-type-count",
        "repair_multi_violation=20",
        "--min-test-task-type-count",
        "boundary_precision=20",
        "--min-test-task-type-count",
        "smiles_invariance=20",
        "--min-test-task-type-count",
        "interrupt_resume=10",
        "--min-test-task-type-count",
        "tool_forced_l3=10",
        "--anonymous",
    ],
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
        "python",
        "scripts/preflight_neurips_ed_artifact.py",
        "--release",
        "benchmarks/releases/sgchem_v1.0",
    ],
    [
        "uv",
        "run",
        "specguard-chem",
        "validate-croissant",
        "benchmarks/releases/sgchem_v1.0/croissant.json",
        "--anonymous",
    ],
    [
        "uv",
        "run",
        "python",
        "scripts/audit_task_inventory.py",
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
        "runs/paper_sweeps/sgchem_v1.0_test",
        "--seed",
        "7",
    ],
    [
        "uv",
        "run",
        "specguard-chem",
        "paper-figures",
        "--runs",
        "runs/paper_sweeps/sgchem_v1.0_test",
        "--out",
        "paper_v1",
    ],
]


def main() -> int:
    for command in COMMANDS:
        print("+ " + " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
