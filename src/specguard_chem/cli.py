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

    table = Table(title="Run Metrics")
    table.add_column("metric")
    table.add_column("value")
    for key, value in summary.items():
        table.add_row(key, f"{value:.3f}" if isinstance(value, float) else str(value))

    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
