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
from .scoring.leaderboard import export_leaderboard_submission

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

    metric_order = [
        "num_tasks",
        "avg_spec_score",
        "hard_violation_rate",
        "abstain_rate",
        "accept_rate",
        "avg_rounds",
        "avg_edit_distance",
        "avg_confidence",
        "brier_score",
        "ece",
        "abstention_utility",
    ]

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

    for key, value in summary.items():
        if key not in metric_order:
            table.add_row(key, _format(value))

    console.print(table)


@app.command()
def export_leaderboard(
    run_path: Path = typer.Argument(..., help="Run directory containing trace.jsonl"),
    model_name: str = typer.Option(..., "--model-name", help="Name of the model"),
    organization: str = typer.Option(..., "--organization", help="Organization name"),
    model_type: str = typer.Option("open-source", "--type", help="Model type (open-source, closed-source, academic)"),
    description: str = typer.Option(..., "--description", help="Model description"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output submission file path"),
) -> None:
    """Export evaluation results to leaderboard submission format."""
    if not run_path.exists():
        console.print(f"[red]Run path not found:[/red] {run_path}")
        raise typer.Exit(code=1)

    output_path = output or Path("submission.json")

    try:
        export_leaderboard_submission(
            run_path=run_path,
            model_name=model_name,
            organization=organization,
            model_type=model_type,
            description=description,
            output_path=output_path
        )
        console.print(f"✅ Leaderboard submission exported to [green]{output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Export failed:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
