from __future__ import annotations

"""Generate paper figures/tables from benchmark sweep outputs."""

from pathlib import Path
from typing import Optional

import typer

from specguard_chem.benchmark.paper import make_paper_artifacts

app = typer.Typer(help="Generate paper figures and tables from benchmark runs.")


@app.command()
def main(
    runs: Optional[Path] = typer.Option(
        None,
        "--runs",
        help="Sweep directory containing aggregate.json and baseline run folders.",
    ),
    aggregate: Optional[Path] = typer.Option(
        None,
        "--aggregate",
        help="Path to aggregate.json (alternative to --runs).",
    ),
    out: Path = typer.Option(
        Path("paper"),
        "--out",
        help="Output directory for figures/tables/metrics summary.",
    ),
) -> None:
    result = make_paper_artifacts(out_dir=out, runs_dir=runs, aggregate_path=aggregate)
    typer.echo(f"Figures: {result['figures_dir']}")
    typer.echo(f"Tables: {result['tables_dir']}")
    typer.echo(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    app()

