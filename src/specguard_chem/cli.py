from __future__ import annotations

"""CLI entrypoint for SpecGuard-Chem."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import list_available_specs, list_available_suites, load_spec
from .dataset import (
    build_corpus_records,
    compute_corpus_sha256,
    compute_taskset_sha256,
    generate_tasks_from_corpus,
    load_corpus_records,
    validate_dataset_file,
    write_corpus_records,
    write_tasks_jsonl,
)
from .runner.runner import TaskRunner
from .scoring import reports
from .utils import jsonio

app = typer.Typer(help="Spec-driven evaluation harness for chemical guardrails.")
console = Console()

BASELINE_MATRIX: List[Dict[str, str]] = [
    {
        "name": "heuristic_non_tool_l2",
        "model": "heuristic",
        "protocol": "L2",
    },
    {
        "name": "heuristic_tool_l3",
        "model": "open_source_example",
        "protocol": "L3",
    },
    {
        "name": "abstention_guard_l2",
        "model": "abstention_guard",
        "protocol": "L2",
    },
]

BASELINE_GROUP_FIELDS: tuple[str, ...] = (
    "name",
    "model",
    "protocol",
    "suite",
    "spec_split",
    "source",
)


def _metric_str(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _collect_baseline_summary_paths(inputs: List[Path]) -> List[Path]:
    collected: List[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            collected.extend(sorted(candidate.rglob("baseline_summary.json")))
        elif candidate.exists():
            collected.append(candidate)
    deduped: List[Path] = []
    seen: set[str] = set()
    for path in collected:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _parse_group_by_dimensions(group_by: str) -> List[str]:
    raw = [part.strip() for part in group_by.split(",") if part.strip()]
    if not raw:
        return ["name"]
    dims: List[str] = []
    seen: set[str] = set()
    for value in raw:
        if value in seen:
            continue
        seen.add(value)
        dims.append(value)
    unknown = [value for value in dims if value not in BASELINE_GROUP_FIELDS]
    if unknown:
        options = ", ".join(BASELINE_GROUP_FIELDS)
        unknown_label = ", ".join(unknown)
        raise ValueError(
            f"Unknown --group-by fields: {unknown_label}. Allowed fields: {options}"
        )
    return dims


@app.command("build-corpus")
def build_corpus(
    output: Path = typer.Option(
        Path("data/corpus.parquet"),
        "--output",
        "-o",
        help="Corpus artifact path (.parquet preferred, .jsonl supported).",
    ),
    seed: int = typer.Option(7, "--seed", help="Deterministic corpus seed."),
    max_molecules: int = typer.Option(
        2000, "--max-molecules", help="Maximum molecules to include in corpus."
    ),
    reaction_depth: int = typer.Option(
        2, "--reaction-depth", help="Number of reaction-expansion rounds."
    ),
) -> None:
    records = build_corpus_records(
        seed=seed, max_molecules=max_molecules, reaction_depth=reaction_depth
    )
    written = write_corpus_records(output, records)
    corpus_sha = compute_corpus_sha256(records)
    metadata = {
        "generated_at": datetime.utcnow().isoformat(),
        "seed": seed,
        "max_molecules": max_molecules,
        "reaction_depth": reaction_depth,
        "num_records": len(records),
        "corpus_sha256": corpus_sha,
        "output_path": str(written),
    }
    metadata_path = written.parent / f"{written.stem}.meta.json"
    jsonio.write_json(metadata_path, metadata)

    if written != output:
        console.print(
            f"[yellow]Parquet unavailable; wrote JSONL fallback:[/yellow] {written}"
        )
    console.print(f"Corpus records: [green]{len(records)}[/green]")
    console.print(f"Corpus SHA256: [green]{corpus_sha}[/green]")
    console.print(f"Metadata written to [green]{metadata_path}[/green]")


@app.command("generate-tasks")
def generate_tasks(
    corpus_path: Path = typer.Option(
        Path("data/corpus.parquet"),
        "--corpus",
        help="Input corpus path (.parquet or .jsonl).",
    ),
    output: Path = typer.Option(
        Path("tasks/suites/generated_v1.jsonl"),
        "--output",
        "-o",
        help="Output task-suite JSONL path.",
    ),
    target_tasks: int = typer.Option(
        1000, "--target-tasks", help="Number of tasks to generate."
    ),
    seed: int = typer.Option(7, "--seed", help="Deterministic taskgen seed."),
    suite_name: str = typer.Option(
        "generated_v1", "--suite-name", help="Suite name written into generated tasks."
    ),
    spec_ids: str = typer.Option(
        "",
        "--spec-ids",
        help="Comma-separated spec IDs to include (default: all).",
    ),
    spec_split: Optional[str] = typer.Option(
        None,
        "--spec-split",
        help="Optional spec split filter (train|dev|test).",
    ),
    near_miss_margin_band: float = typer.Option(
        5.0,
        "--near-miss-margin-band",
        help="Max absolute negative margin for near-miss task construction.",
    ),
    boundary_margin_band: float = typer.Option(
        1.0,
        "--boundary-margin-band",
        help="Max distance to nearest hard property bound for boundary-precision tasks.",
    ),
) -> None:
    if not corpus_path.exists() and not corpus_path.with_suffix(".jsonl").exists():
        console.print(f"[red]Corpus not found:[/red] {corpus_path}")
        raise typer.Exit(code=1)
    corpus_records = load_corpus_records(corpus_path)

    selected_ids = []
    requested = [part.strip() for part in spec_ids.split(",") if part.strip()]
    available_specs = list_available_specs()
    if requested:
        unknown = [spec_id for spec_id in requested if spec_id not in available_specs]
        if unknown:
            console.print(f"[red]Unknown spec IDs:[/red] {', '.join(unknown)}")
            raise typer.Exit(code=1)
        selected_ids = requested
    else:
        selected_ids = available_specs

    specs = [load_spec(spec_id) for spec_id in selected_ids]
    if spec_split is not None:
        specs = [spec for spec in specs if spec.spec_split == spec_split]
    if not specs:
        console.print("[red]No specs selected for generation.[/red]")
        raise typer.Exit(code=1)

    tasks = generate_tasks_from_corpus(
        corpus_records=corpus_records,
        specs=specs,
        target_tasks=target_tasks,
        seed=seed,
        suite_name=suite_name,
        near_miss_margin_band=near_miss_margin_band,
        boundary_margin_band=boundary_margin_band,
    )
    write_tasks_jsonl(output, tasks)
    taskset_sha = compute_taskset_sha256(tasks)
    metadata = {
        "generated_at": datetime.utcnow().isoformat(),
        "seed": seed,
        "target_tasks": target_tasks,
        "actual_tasks": len(tasks),
        "suite_name": suite_name,
        "spec_ids": [spec.id for spec in specs],
        "spec_split": spec_split,
        "taskset_sha256": taskset_sha,
        "corpus_sha256": compute_corpus_sha256(corpus_records),
        "near_miss_margin_band": near_miss_margin_band,
        "boundary_margin_band": boundary_margin_band,
    }
    metadata_path = output.with_suffix(".meta.json")
    jsonio.write_json(metadata_path, metadata)

    console.print(f"Generated tasks: [green]{len(tasks)}[/green]")
    console.print(f"Taskset SHA256: [green]{taskset_sha}[/green]")
    console.print(f"Tasks written to [green]{output}[/green]")
    console.print(f"Metadata written to [green]{metadata_path}[/green]")


@app.command("validate-dataset")
def validate_dataset(
    dataset_path: Path = typer.Argument(
        ..., help="Task suite JSONL path or frozen benchmark release directory."
    ),
    near_miss_margin_band: float = typer.Option(
        5.0,
        "--near-miss-margin-band",
        help="Expected near-miss absolute margin ceiling.",
    ),
    boundary_margin_band: float = typer.Option(
        1.0,
        "--boundary-margin-band",
        help="Expected boundary-precision distance ceiling.",
    ),
    repair_start_hard_fail_threshold: float = typer.Option(
        0.70,
        "--repair-start-hard-fail-threshold",
        help="Minimum required fraction of repair tasks that start hard-failing.",
    ),
    min_tool_forced_l3_test_share: float = typer.Option(
        0.10,
        "--min-tool-forced-l3-test-share",
        help="Minimum required fraction of tool_forced_l3 tasks in TEST split for release directories.",
    ),
) -> None:
    if not dataset_path.exists():
        console.print(f"[red]Dataset path not found:[/red] {dataset_path}")
        raise typer.Exit(code=1)

    if dataset_path.is_dir():
        from .benchmark.release import validate_release_directory

        result = validate_release_directory(
            dataset_path,
            near_miss_margin_band=near_miss_margin_band,
            boundary_margin_band=boundary_margin_band,
            repair_start_hard_fail_threshold=repair_start_hard_fail_threshold,
            min_tool_forced_l3_test_share=min_tool_forced_l3_test_share,
        )
    else:
        result = validate_dataset_file(
            dataset_path,
            near_miss_margin_band=near_miss_margin_band,
            boundary_margin_band=boundary_margin_band,
            repair_start_hard_fail_threshold=repair_start_hard_fail_threshold,
        )

    summary_table = Table(title="Dataset Validation")
    summary_table.add_column("metric")
    summary_table.add_column("value")
    summary_table.add_row("valid", str(result["valid"]))
    summary_table.add_row("num_tasks", str(result["num_tasks"]))
    summary_table.add_row("num_errors", str(result["num_errors"]))
    summary_table.add_row("family_counts", json.dumps(result["family_counts"]))
    if "split_counts" in result:
        summary_table.add_row("split_counts", json.dumps(result["split_counts"]))
    summary_table.add_row(
        "repair_start_hard_fail_rate",
        _metric_str(result.get("repair_start_hard_fail_rate")),
    )
    console.print(summary_table)

    errors = result.get("errors", [])
    if errors:
        error_table = Table(title="Validation Errors (first 20)")
        error_table.add_column("error")
        for message in errors[:20]:
            error_table.add_row(message)
        console.print(error_table)
        raise typer.Exit(code=1)


@app.command("freeze-benchmark")
def freeze_benchmark(
    benchmark_id: str = typer.Option(
        "sgchem_v0.3", "--benchmark-id", help="Benchmark release identifier."
    ),
    out: Path = typer.Option(
        Path("benchmarks/releases/sgchem_v0.3"),
        "--out",
        help="Output release directory.",
    ),
    target_tasks: int = typer.Option(
        200, "--target-tasks", help="Number of tasks to generate in release."
    ),
    seed: int = typer.Option(7, "--seed", help="Deterministic generation seed."),
    near_miss_margin_band: float = typer.Option(
        5.0,
        "--near-miss-margin-band",
        help="Near-miss margin band used during generation and validation.",
    ),
    boundary_margin_band: float = typer.Option(
        1.0,
        "--boundary-margin-band",
        help="Boundary precision margin band used during generation and validation.",
    ),
    repair_start_hard_fail_threshold: float = typer.Option(
        0.70,
        "--repair-start-hard-fail-threshold",
        help="Minimum required fraction of repair tasks that start hard-failing.",
    ),
    min_tool_forced_l3_test_share: float = typer.Option(
        0.10,
        "--min-tool-forced-l3-test-share",
        help="Minimum required fraction of tool_forced_l3 tasks in TEST split.",
    ),
) -> None:
    from .benchmark.release import freeze_benchmark_release

    try:
        manifest = freeze_benchmark_release(
            benchmark_id=benchmark_id,
            out_dir=out,
            target_tasks=target_tasks,
            seed=seed,
            near_miss_margin_band=near_miss_margin_band,
            boundary_margin_band=boundary_margin_band,
            repair_start_hard_fail_threshold=repair_start_hard_fail_threshold,
            min_tool_forced_l3_test_share=min_tool_forced_l3_test_share,
        )
    except Exception as exc:
        console.print(f"[red]freeze-benchmark failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
    console.print(f"Benchmark release: [green]{out}[/green]")
    console.print(
        "Split counts: "
        + json.dumps((counts.get("by_split") if isinstance(counts, dict) else {}))
    )
    console.print(f"Manifest written to [green]{out / 'MANIFEST.json'}[/green]")


@app.command("run-benchmark")
def run_benchmark(
    benchmark: Path = typer.Option(
        ...,
        "--benchmark",
        help="Path to frozen benchmark release directory (contains MANIFEST.json).",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="Release split to run (train|dev|test).",
    ),
    baselines: Path = typer.Option(
        Path("baselines/paper_baselines.yaml"),
        "--baselines",
        help="Baseline matrix YAML file.",
    ),
    out: Path = typer.Option(
        Path("runs/paper_sweeps/sgchem_v0.3_test"),
        "--out",
        help="Output directory for sweep artifacts.",
    ),
    seed: int = typer.Option(7, "--seed", help="Deterministic seed for baselines."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Optional cap on number of tasks per baseline."
    ),
    allow_external: bool = typer.Option(
        False,
        "--allow-external",
        help="Allow live external adapter calls (API/process baselines).",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Optional cache directory root; step request/response caches are written per baseline.",
    ),
    replay_cache: Optional[Path] = typer.Option(
        None,
        "--replay-cache",
        help="Replay from previously cached step outputs instead of live model/process calls.",
    ),
    n_bootstrap: int = typer.Option(
        400,
        "--n-bootstrap",
        help="Number of bootstrap resamples for CI estimation in aggregate.json.",
    ),
) -> None:
    if not benchmark.exists():
        console.print(f"[red]Benchmark directory not found:[/red] {benchmark}")
        raise typer.Exit(code=1)
    if not baselines.exists():
        console.print(f"[red]Baseline YAML not found:[/red] {baselines}")
        raise typer.Exit(code=1)

    from .benchmark.sweep import run_benchmark_sweep

    try:
        aggregate = run_benchmark_sweep(
            benchmark_dir=benchmark,
            split=split,
            baselines_path=baselines,
            out_dir=out,
            seed=seed,
            limit=limit,
            allow_external=allow_external,
            cache_dir=cache_dir,
            replay_cache=replay_cache,
            n_bootstrap=n_bootstrap,
        )
    except Exception as exc:
        console.print(f"[red]run-benchmark failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Benchmark Sweep")
    table.add_column("baseline")
    table.add_column("model")
    table.add_column("track")
    table.add_column("protocol")
    table.add_column("num_tasks", justify="right")
    table.add_column("accept_rate", justify="right")
    table.add_column("hard_violation_rate", justify="right")
    for row in aggregate.get("baselines", []):
        metrics = row.get("metrics", {})
        table.add_row(
            str(row.get("name")),
            str(row.get("model")),
            str(row.get("track") or "closed_book"),
            str(row.get("protocol") or "mixed"),
            str(metrics.get("num_tasks", 0)),
            _metric_str(metrics.get("accept_rate")),
            _metric_str(metrics.get("hard_violation_rate")),
        )
    console.print(table)
    skipped = aggregate.get("skipped_baselines", [])
    if skipped:
        skipped_table = Table(title="Skipped Optional Baselines")
        skipped_table.add_column("baseline")
        skipped_table.add_column("track")
        skipped_table.add_column("reason")
        for row in skipped:
            skipped_table.add_row(
                str(row.get("name")),
                str(row.get("track")),
                str(row.get("reason")),
            )
        console.print(skipped_table)
    console.print(f"Aggregate written to [green]{out / 'aggregate.json'}[/green]")


@app.command("paper-figures")
def paper_figures(
    runs: Optional[Path] = typer.Option(
        None,
        "--runs",
        help="Sweep directory containing aggregate.json and per-baseline runs.",
    ),
    aggregate: Optional[Path] = typer.Option(
        None,
        "--aggregate",
        help="Path to aggregate.json (alternative to --runs).",
    ),
    out: Path = typer.Option(
        Path("paper"),
        "--out",
        help="Output directory for paper figures/tables.",
    ),
) -> None:
    from .benchmark.paper import make_paper_artifacts

    try:
        result = make_paper_artifacts(out_dir=out, runs_dir=runs, aggregate_path=aggregate)
    except Exception as exc:
        console.print(f"[red]paper-figures failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Figures: [green]{result['figures_dir']}[/green]")
    console.print(f"Tables: [green]{result['tables_dir']}[/green]")
    console.print(f"Summary: [green]{result['summary_path']}[/green]")


@app.command()
def run(
    suite: str = typer.Argument(..., help="Task suite to execute."),
    protocol: Optional[str] = typer.Option(
        None, "--protocol", "-p", help="Restrict to a specific protocol (L1/L2/L3)."
    ),
    spec_split: Optional[str] = typer.Option(
        None,
        "--spec-split",
        help="Optional spec split filter (train|dev|test).",
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
    allow_external: bool = typer.Option(
        False,
        "--allow-external",
        help="Allow external adapter calls (API/process).",
    ),
) -> None:
    if suite not in list_available_suites():
        console.print(f"[red]Unknown suite[/red]: {suite}")
        raise typer.Exit(code=1)

    runner = TaskRunner(model, seed=seed, allow_external=allow_external)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"{suite}_{protocol or 'mixed'}_{spec_split or 'all'}"
    run_dir = run_path or Path("runs") / f"{timestamp}_{suffix}"
    results = runner.run_suite(
        suite,
        protocol=protocol,
        spec_split=spec_split,
        limit=limit,
        run_dir=run_dir,
    )

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


@app.command("run-baselines")
def run_baselines(
    suite: str = typer.Option(
        "basic_plain", "--suite", help="Task suite to execute for baselines."
    ),
    spec_split: Optional[str] = typer.Option(
        None,
        "--spec-split",
        help="Optional spec split filter (train|dev|test).",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Optional per-run task limit."
    ),
    out_dir: Path = typer.Option(
        Path("runs/baselines"),
        "--out-dir",
        help="Directory for baseline run artifacts.",
    ),
    seed: int = typer.Option(7, "--seed", help="Deterministic seed for all runs."),
) -> None:
    if suite not in list_available_suites():
        console.print(f"[red]Unknown suite[/red]: {suite}")
        raise typer.Exit(code=1)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = out_dir / f"{timestamp}_{suite}_{spec_split or 'all'}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    results_table = Table(title="Baseline Matrix")
    results_table.add_column("name")
    results_table.add_column("model")
    results_table.add_column("protocol")
    results_table.add_column("num_tasks", justify="right")
    results_table.add_column("accept_rate", justify="right")
    results_table.add_column("hard_violation_rate", justify="right")
    results_table.add_column("abstention_utility", justify="right")

    run_payloads: List[Dict[str, Any]] = []
    for baseline in BASELINE_MATRIX:
        name = baseline["name"]
        model = baseline["model"]
        protocol = baseline["protocol"]
        run_path = batch_dir / name

        runner = TaskRunner(model, seed=seed)
        runner.run_suite(
            suite,
            protocol=protocol,
            spec_split=spec_split,
            limit=limit,
            run_dir=run_path,
        )
        records = reports.load_trace(run_path)
        summary = reports.summarise(records)
        report_path = reports.write_report(run_path, records=records, summary=summary)

        payload = {
            "name": name,
            "model": model,
            "protocol": protocol,
            "run_path": str(run_path),
            "report_path": str(report_path),
            "num_tasks": int(summary.get("num_tasks", 0)),
            "accept_rate": summary.get("accept_rate"),
            "hard_violation_rate": summary.get("hard_violation_rate"),
            "abstention_utility": summary.get("abstention_utility"),
        }
        run_payloads.append(payload)

        results_table.add_row(
            name,
            model,
            protocol,
            str(payload["num_tasks"]),
            _metric_str(payload["accept_rate"]),
            _metric_str(payload["hard_violation_rate"]),
            _metric_str(payload["abstention_utility"]),
        )

    summary_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "suite": suite,
        "spec_split": spec_split,
        "limit": limit,
        "seed": seed,
        "runs": run_payloads,
    }
    summary_path = batch_dir / "baseline_summary.json"
    jsonio.write_json(summary_path, summary_payload)

    console.print(results_table)
    console.print(f"Baseline summary written to [green]{summary_path}[/green]")


@app.command("compare-baselines")
def compare_baselines(
    summary_paths: List[Path] = typer.Argument(
        ...,
        help=(
            "One or more baseline_summary.json paths or directories containing them."
        ),
    ),
    group_by: str = typer.Option(
        "name",
        "--group-by",
        help=(
            "Comma-separated grouping fields for aggregate rows "
            "(name,model,protocol,suite,spec_split,source)."
        ),
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path for the aggregate comparison JSON artifact.",
    ),
) -> None:
    # Support direct function calls from unit tests where Typer default
    # OptionInfo objects can appear instead of parsed values.
    if not isinstance(group_by, str):
        group_by = "name"
    if output is not None and not isinstance(output, Path):
        output = None

    try:
        group_fields = _parse_group_by_dimensions(group_by)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    paths = _collect_baseline_summary_paths(summary_paths)
    if not paths:
        console.print("[red]No baseline summary files found.[/red]")
        raise typer.Exit(code=1)

    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = jsonio.read_json(path)
        if not isinstance(payload, dict):
            continue
        suite = payload.get("suite")
        spec_split = payload.get("spec_split")
        generated_at = payload.get("generated_at")
        run_items = payload.get("runs") if isinstance(payload.get("runs"), list) else []
        for run in run_items:
            if not isinstance(run, dict):
                continue
            name = run.get("name")
            model = run.get("model")
            protocol = run.get("protocol")
            if not isinstance(name, str) or not name:
                continue
            rows.append(
                {
                    "source": str(path),
                    "suite": suite,
                    "spec_split": spec_split,
                    "generated_at": generated_at,
                    "name": name,
                    "model": model,
                    "protocol": protocol,
                    "num_tasks": int(run.get("num_tasks", 0)),
                    "accept_rate": _as_float(run.get("accept_rate")),
                    "hard_violation_rate": _as_float(run.get("hard_violation_rate")),
                    "abstention_utility": _as_float(run.get("abstention_utility")),
                }
            )

    if not rows:
        console.print("[red]No valid baseline rows found in summaries.[/red]")
        raise typer.Exit(code=1)

    aggregate_map: Dict[tuple[str, ...], Dict[str, Any]] = {}
    for row in rows:
        group_values = {
            field: str(row.get(field)) if row.get(field) is not None else None
            for field in group_fields
        }
        key = tuple(group_values[field] or "" for field in group_fields)
        bucket = aggregate_map.setdefault(
            key,
            {
                "group_values": group_values,
                "models": set(),
                "protocols": set(),
                "n_rows": 0,
                "accept_rate_values": [],
                "hard_violation_rate_values": [],
                "abstention_utility_values": [],
            },
        )
        if isinstance(row.get("model"), str):
            bucket["models"].add(row["model"])
        if isinstance(row.get("protocol"), str):
            bucket["protocols"].add(row["protocol"])
        bucket["n_rows"] += 1
        bucket["accept_rate_values"].append(row.get("accept_rate"))
        bucket["hard_violation_rate_values"].append(row.get("hard_violation_rate"))
        bucket["abstention_utility_values"].append(row.get("abstention_utility"))

    aggregate_rows: List[Dict[str, Any]] = []
    for key in sorted(aggregate_map):
        bucket = aggregate_map[key]
        row: Dict[str, Any] = dict(bucket["group_values"])
        row.update(
            {
                "models": sorted(bucket["models"]),
                "protocols": sorted(bucket["protocols"]),
                "n_rows": bucket["n_rows"],
                "mean_accept_rate": _mean_or_none(bucket["accept_rate_values"]),
                "mean_hard_violation_rate": _mean_or_none(
                    bucket["hard_violation_rate_values"]
                ),
                "mean_abstention_utility": _mean_or_none(
                    bucket["abstention_utility_values"]
                ),
            }
        )
        aggregate_rows.append(row)

    def _sort_key(item: Dict[str, Any]) -> tuple[float, float, tuple[str, ...]]:
        accept = item.get("mean_accept_rate")
        hard_violation = item.get("mean_hard_violation_rate")
        accept_sort = float(accept) if accept is not None else -1.0
        hard_sort = float(hard_violation) if hard_violation is not None else 1.0
        group_sort = tuple(str(item.get(field) or "") for field in group_fields)
        return (-accept_sort, hard_sort, group_sort)

    aggregate_rows.sort(key=_sort_key)

    table = Table(title="Baseline Comparison")
    for field in group_fields:
        table.add_column(field)
    include_models = "model" not in group_fields
    include_protocols = "protocol" not in group_fields
    if include_models:
        table.add_column("models")
    if include_protocols:
        table.add_column("protocols")
    table.add_column("n_rows", justify="right")
    table.add_column("mean_accept_rate", justify="right")
    table.add_column("mean_hard_violation_rate", justify="right")
    table.add_column("mean_abstention_utility", justify="right")
    for row in aggregate_rows:
        cells = [str(row.get(field) or "") for field in group_fields]
        if include_models:
            cells.append(",".join(row["models"]))
        if include_protocols:
            cells.append(",".join(row["protocols"]))
        cells.extend(
            [
                str(row["n_rows"]),
                _metric_str(row["mean_accept_rate"]),
                _metric_str(row["mean_hard_violation_rate"]),
                _metric_str(row["mean_abstention_utility"]),
            ]
        )
        table.add_row(*cells)
    console.print(table)

    if output is None:
        if len(paths) == 1:
            output = paths[0].with_name("baseline_compare.json")
        else:
            output = Path("runs/baseline_compare.json")
    comparison_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "sources": [str(path) for path in paths],
        "group_by": group_fields,
        "n_rows": len(rows),
        "rows": rows,
        "aggregate": aggregate_rows,
    }
    jsonio.write_json(output, comparison_payload)
    console.print(f"Baseline comparison written to [green]{output}[/green]")


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
        "pass_at_steps",
        "avg_steps_to_accept",
        "avg_verify_calls_to_accept",
        "accept_rate_by_protocol",
        "hard_violation_rate_by_protocol",
        "expected_pass_rate",
        "false_abstain_rate",
        "violation_rate",
        "correct_abstain_rate",
        "unsafe_completion_rate",
        "reject_on_abstain_expected_rate",
        "interrupt_compliance_rate",
        "n_resume_tasks",
        "n_resume_fired",
        "resume_token_ok_rate",
        "resume_success_rate",
        "avg_extra_steps_after_interrupt",
        "avg_morgan_tanimoto",
        "median_morgan_tanimoto",
        "n_morgan_measured",
        "avg_edit_distance",
        "n_edit_measured",
        "avg_final_edit_cost_brics",
        "n_final_edit_cost_brics_measured",
        "avg_trajectory_edit_distance",
        "n_trajectory_edit_distance_measured",
        "avg_trajectory_edit_cost_brics",
        "n_trajectory_edit_cost_brics_measured",
        "abstain_rate",
        "accept_rate",
        "avg_rounds",
        "avg_steps_used",
        "avg_proposals_used",
        "avg_verify_calls_used",
        "avg_p_hard_pass",
        "brier_score",
        "ece",
        "abstention_utility",
        "n_agent_outputs",
        "schema_error_rate",
        "invalid_action_rate",
        "invalid_tool_call_rate",
        "spec_family_breakdown",
        "spec_split_breakdown",
        "soft_compliance_rate_given_hard_pass",
        "weighted_soft_score_given_hard_pass",
        "n_invariance_groups",
        "n_invariance_groups_evaluable",
        "n_invariance_groups_incomplete",
        "invariance_failure_rate",
        "n_boundary_precision_tasks",
        "boundary_precision_failure_rate",
        "boundary_precision_pass_rate",
    ]

    spec_ids = {record.get("spec_id") for record in records if record.get("spec_id")}
    metadata = reports.build_metadata(spec_ids, records=records)
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
        "risk_coverage_curve",
        "cost_coverage_curve",
        "utility_sensitivity",
    }
    for key, value in summary.items():
        if key not in metric_order and key not in skip_keys:
            table.add_row(key, _format(value))

    console.print(table)

    confusion = summary.get("confusion")
    if isinstance(confusion, dict):
        conf_table = Table(title="Decision Confusion Counts")
        conf_table.add_column("label")
        conf_table.add_column("ACCEPT", justify="right")
        conf_table.add_column("REJECT", justify="right")
        conf_table.add_column("ABSTAIN", justify="right")
        for expected_action in ("ACCEPT", "ABSTAIN", "REJECT"):
            row = confusion.get(expected_action) if isinstance(confusion, dict) else None
            if not isinstance(row, dict):
                continue
            conf_table.add_row(
                f"expected_{expected_action}",
                str(row.get("ACCEPT", 0)),
                str(row.get("REJECT", 0)),
                str(row.get("ABSTAIN", 0)),
            )
        console.print(conf_table)

    legacy_confusion = summary.get("legacy_confusion")
    if isinstance(legacy_confusion, dict):
        legacy_table = Table(title="Legacy Confusion Counts")
        legacy_table.add_column("label")
        legacy_table.add_column("count", justify="right")
        for label in ("TA", "FA", "FV", "TB", "UA"):
            legacy_table.add_row(label, str(legacy_confusion.get(label, 0)))
        legacy_table.add_row("N_expected_PASS", str(summary.get("n_expected_pass", 0)))
        legacy_table.add_row(
            "N_expected_ABSTAIN", str(summary.get("n_expected_abstain", 0))
        )
        console.print(legacy_table)

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
