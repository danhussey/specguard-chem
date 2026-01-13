from __future__ import annotations

"""CLI entrypoint for SpecGuard-Chem."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import list_available_suites
from .runner.runner import TaskRunner
from .scoring import reports

app = typer.Typer(help="Spec-driven evaluation harness for chemical guardrails.")
console = Console()


@app.command()
def run(
    suite: str = typer.Argument(..., help="Task suite to execute."),
    protocol: Optional[str] = typer.Option(
        None, "--protocol", "-p", help="Restrict to a specific protocol (L1/L2/L3)."
    ),
    model: str = typer.Option(
        "heuristic", "--model", "-m", help="Adapter identifier to use for execution."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of tasks processed."
    ),
    run_path: Optional[Path] = typer.Option(
        None, "--run-path", help="Directory to store run artefacts."
    ),
    seed: int = typer.Option(7, "--seed", help="Deterministic seed for adapters."),
) -> None:
    if suite not in list_available_suites():
        console.print(f"[red]Unknown suite[/red]: {suite}")
        raise typer.Exit(code=1)

    runner = TaskRunner(model, seed=seed)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"{suite}_{protocol or 'mixed'}"
    run_dir = run_path or Path("runs") / f"{timestamp}_{suffix}"
    results = runner.run_suite(suite, protocol=protocol, limit=limit, run_dir=run_dir)

    table = Table(title="SpecGuard Run Summary")
    table.add_column("task_id")
    table.add_column("protocol")
    table.add_column("hard_pass", justify="right")
    table.add_column("spec_score", justify="right")
    table.add_column("rounds", justify="right")
    table.add_column("abstained", justify="right")

    for record in results:
        table.add_row(
            record.task_id,
            record.protocol,
            str(int(record.hard_pass)),
            f"{record.spec_score:.2f}",
            str(len(record.rounds)),
            str(int(record.abstained)),
        )

    console.print(table)
    console.print(f"Artefacts written to [green]{run_dir}[/green]")


@app.command()
def report(
    run_path: Path = typer.Argument(..., help="Run directory containing trace.jsonl")
) -> None:
    if not run_path.exists():
        console.print(f"[red]Run path not found:[/red] {run_path}")
        raise typer.Exit(code=1)

    records = reports.load_trace(run_path)
    summary = reports.summarise(records)
    report_path = reports.write_report(run_path, records=records, summary=summary)

    metric_order = [
        "num_tasks",
        "avg_spec_score",
        "hard_violation_rate",
        "expected_pass_rate",
        "false_abstain_rate",
        "violation_rate",
        "correct_abstain_rate",
        "unsafe_completion_rate",
        "interrupt_compliance_rate",
        "avg_morgan_tanimoto",
        "median_morgan_tanimoto",
        "n_morgan_measured",
        "avg_edit_distance",
        "n_edit_measured",
        "abstain_rate",
        "accept_rate",
        "avg_rounds",
        "avg_confidence",
        "brier_score",
        "ece",
        "abstention_utility",
    ]

    spec_ids = {record.get("spec_id") for record in records if record.get("spec_id")}
    metadata = reports.build_metadata(spec_ids)
    if metadata.get("rdkit_version"):
        console.print(f"RDKit version: {metadata['rdkit_version']}")

    table = Table(title="Run Metrics")
    table.add_column("metric")
    table.add_column("value")

    def _format(value: object) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    for key in metric_order:
        if key in summary:
            table.add_row(key, _format(summary[key]))

    skip_keys = {
        "confusion",
        "n_expected_pass",
        "n_expected_abstain",
        "n_interrupt_tasks",
        "n_interrupt_fired",
        "n_interrupt_compliant",
    }
    for key, value in summary.items():
        if key not in metric_order and key not in skip_keys:
            table.add_row(key, _format(value))

    console.print(table)

    confusion = summary.get("confusion")
    if isinstance(confusion, dict):
        conf_table = Table(title="Confusion Counts")
        conf_table.add_column("label")
        conf_table.add_column("count", justify="right")
        for label in ("TA", "FA", "FV", "TB", "UA"):
            conf_table.add_row(label, str(confusion.get(label, 0)))
        conf_table.add_row("N_expected_PASS", str(summary.get("n_expected_pass", 0)))
        conf_table.add_row(
            "N_expected_ABSTAIN", str(summary.get("n_expected_abstain", 0))
        )
        console.print(conf_table)

    interrupt_counts = (
        summary.get("n_interrupt_tasks", 0),
        summary.get("n_interrupt_fired", 0),
        summary.get("n_interrupt_compliant", 0),
    )
    if any(count > 0 for count in interrupt_counts):
        interrupt_table = Table(title="Interrupt Counts")
        interrupt_table.add_column("label")
        interrupt_table.add_column("count", justify="right")
        interrupt_table.add_row(
            "N_interrupt_tasks", str(summary.get("n_interrupt_tasks", 0))
        )
        interrupt_table.add_row(
            "N_interrupt_fired", str(summary.get("n_interrupt_fired", 0))
        )
        interrupt_table.add_row(
            "N_interrupt_compliant", str(summary.get("n_interrupt_compliant", 0))
        )
        console.print(interrupt_table)

    console.print(f"Report written to [green]{report_path}[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()
