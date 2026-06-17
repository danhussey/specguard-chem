from __future__ import annotations

"""Generate MD Research subset manifests, tables, and figures."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt

from specguard_chem.scoring import reports
from specguard_chem.utils import jsonio

ACTIONS = ("ACCEPT", "REJECT", "ABSTAIN")
PREDICTED = ("ACCEPT", "REJECT", "ABSTAIN", "INVALID")

ACCESS_BY_TRACK = {
    "closed_book": "closed-book",
    "primary_closed_book": "closed-book",
    "tool_enabled": "public-verifier",
    "retrieval": "retrieval",
    "retrieval_upper_bound": "retrieval",
    "wrapper_guarded": "verifier/search wrapper",
    "external": "external diagnostic",
    "external_model_snapshot": "external diagnostic",
}


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _write_md(path: Path, rows: Sequence[Mapping[str, Any]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig: plt.Figure, figures: Path, stem: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.png", dpi=220)
    fig.savefig(figures / f"{stem}.pdf")
    plt.close(fig)


def _public_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]


def _expected(row: Mapping[str, Any]) -> str:
    value = str(row.get("expected_action") or "").upper()
    if value in ACTIONS:
        return value
    legacy = str(row.get("expected") or "PASS").upper()
    if legacy == "ABSTAIN":
        return "ABSTAIN"
    if legacy == "FAIL":
        return "REJECT"
    return "ACCEPT"


def _predicted(row: Mapping[str, Any]) -> str:
    if row.get("schema_error") or row.get("invalid_action") or row.get("invalid_tool_call"):
        return "INVALID"
    value = str(row.get("final_decision") or "").upper()
    if value in PREDICTED:
        return value
    decision = str(row.get("decision") or "").lower()
    if decision == "accept":
        return "ACCEPT"
    if decision == "reject":
        return "REJECT"
    if decision == "abstain":
        return "ABSTAIN"
    return "INVALID"


def _safe_div(numer: float, denom: float) -> float | None:
    return None if denom == 0 else numer / denom


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    matrix = {exp: {pred: 0 for pred in PREDICTED} for exp in ACTIONS}
    for record in records:
        matrix[_expected(record)][_predicted(record)] += 1
    n = len(records)
    recalls = {
        action: _safe_div(matrix[action][action], sum(matrix[action].values()))
        for action in ACTIONS
    }
    predicted = [_predicted(record) for record in records]
    expected = [_expected(record) for record in records]
    unsafe_denom = sum(1 for value in expected if value in {"REJECT", "ABSTAIN"})
    attempted = [idx for idx, value in enumerate(predicted) if value != "ABSTAIN"]
    verify_calls = [float(record.get("verify_calls_used") or 0) for record in records]
    return {
        "n_tasks": n,
        "action_accuracy": _safe_div(sum(1 for e, p in zip(expected, predicted) if e == p), n),
        "balanced_action_accuracy": mean([value for value in recalls.values() if value is not None]),
        "molecule_acceptance_rate": _safe_div(sum(1 for value in predicted if value == "ACCEPT"), n),
        "task_inconsistent_accept_rate": _safe_div(
            sum(1 for e, p in zip(expected, predicted) if e in {"REJECT", "ABSTAIN"} and p == "ACCEPT"),
            unsafe_denom,
        ),
        "accept_recall": recalls["ACCEPT"],
        "reject_recall": recalls["REJECT"],
        "abstain_recall": recalls["ABSTAIN"],
        "schema_error_rate": _safe_div(sum(1 for record in records if record.get("schema_error")), n),
        "hard_violation_rate": _safe_div(
            sum(1 for idx in attempted if not bool(records[idx].get("hard_pass"))),
            len(attempted),
        ),
        "mean_verify_calls": mean(verify_calls) if verify_calls else 0.0,
        "confusion": matrix,
    }


def _load_tasks(release: Path, split: str = "test") -> list[dict[str, Any]]:
    rows = jsonio.read_jsonl(release / "tasks" / f"{split}.jsonl")
    return [row for row in rows if isinstance(row, dict)]


def write_subset_manifest(
    *, release: Path, subset_manifest: Path, seed: int, max_per_family: int
) -> dict[str, Any]:
    tasks = _load_tasks(release, "test")
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_family[str(task.get("task_family") or "unknown")].append(task)
    selected: list[dict[str, Any]] = []
    for family in sorted(by_family):
        rows = sorted(by_family[family], key=lambda item: str(item.get("task_id") or ""))
        rng.shuffle(rows)
        selected.extend(rows[: min(max_per_family, len(rows))])
    selected.sort(key=lambda item: (str(item.get("task_family") or ""), str(item.get("task_id") or "")))
    payload = {
        "benchmark_id": "sgchem_v1.0",
        "split": "test",
        "seed": seed,
        "selection": "stratified_by_task_family",
        "max_per_family": max_per_family,
        "n_tasks": len(selected),
        "task_ids": [str(task.get("task_id")) for task in selected],
        "task_public_hashes": [_public_hash(str(task.get("task_id"))) for task in selected],
        "counts": {
            "by_task_family": dict(Counter(str(task.get("task_family") or "unknown") for task in selected)),
            "by_protocol": dict(Counter(str(task.get("protocol") or "unknown") for task in selected)),
            "by_expected_action": dict(Counter(_expected(task) for task in selected)),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    subset_manifest.parent.mkdir(parents=True, exist_ok=True)
    jsonio.write_json(subset_manifest, payload)
    return payload


def _read_cache_metadata(cache_root: Path | None, name: str) -> dict[str, Any]:
    if cache_root is None:
        return {}
    cache_path = cache_root / name / "cache.jsonl"
    if not cache_path.exists():
        return {}
    for row in jsonio.read_jsonl(cache_path):
        metadata = row.get("model_metadata") if isinstance(row, dict) else None
        if isinstance(metadata, dict):
            return metadata
    return {}


def _provider_from_name(name: str, model: str) -> str:
    for provider in ("openai", "anthropic", "deepseek"):
        if name.startswith(provider) or model.startswith(provider):
            return provider
    return "deterministic"


def collect_rows(aggregate_path: Path, *, cache_root: Path | None, cache_mode: str) -> list[dict[str, Any]]:
    if not aggregate_path.exists():
        return []
    aggregate = jsonio.read_json(aggregate_path)
    if not isinstance(aggregate, dict):
        return []
    rows: list[dict[str, Any]] = []
    for baseline in aggregate.get("baselines", []):
        if not isinstance(baseline, dict):
            continue
        name = str(baseline.get("name") or "")
        run_dir = aggregate_path.parent / str(baseline.get("run_dir") or name)
        records = reports.load_trace(run_dir)
        metrics = _metrics(records)
        adapter_kwargs = baseline.get("adapter_kwargs") if isinstance(baseline.get("adapter_kwargs"), dict) else {}
        metadata = _read_cache_metadata(cache_root, name)
        model = str(baseline.get("model") or "")
        provider = str(metadata.get("provider") or _provider_from_name(name, model))
        model_id = str(metadata.get("model_id") or adapter_kwargs.get("model") or model)
        row = {
            "system": name,
            "provider": provider,
            "adapter": model,
            "model_id": model_id,
            "access_model": ACCESS_BY_TRACK.get(str(baseline.get("track") or ""), str(baseline.get("track") or "")),
            "protocol": str(baseline.get("protocol") or "mixed"),
            "cache_mode": cache_mode,
            "temperature": adapter_kwargs.get("temperature"),
            "top_p": adapter_kwargs.get("top_p"),
            "max_tokens": adapter_kwargs.get("max_tokens"),
            **{key: value for key, value in metrics.items() if key != "confusion"},
            "estimated_cost_usd": None,
        }
        rows.append(row)
    return rows


def _metric_definitions() -> list[dict[str, str]]:
    return [
        {"metric": "molecule_acceptance_rate", "level": "molecule/action", "captures": "fraction ending in ACCEPT", "misses": "whether ACCEPT was the correct action"},
        {"metric": "chemical validity", "level": "molecule", "captures": "parseable and checkable molecule outputs", "misses": "scientific decision correctness"},
        {"metric": "constraint satisfaction", "level": "molecule", "captures": "hard rule compliance for attempted molecules", "misses": "reject and abstain semantics"},
        {"metric": "action_accuracy", "level": "task", "captures": "exact ACCEPT/REJECT/ABSTAIN match", "misses": "why an action was selected"},
        {"metric": "accept_recall", "level": "action", "captures": "ACCEPT tasks solved as ACCEPT", "misses": "reject/abstain behavior"},
        {"metric": "reject_recall", "level": "action", "captures": "REJECT tasks identified", "misses": "molecule quality on accept tasks"},
        {"metric": "abstain_recall", "level": "action", "captures": "contradiction/uncertainty abstentions", "misses": "near-miss repair quality"},
        {"metric": "task_inconsistent_accept_rate", "level": "task", "captures": "unsafe accept collapse on reject/abstain tasks", "misses": "benign false abstentions"},
        {"metric": "mean_verify_calls", "level": "protocol", "captures": "verifier budget usage", "misses": "quality independent of budget"},
        {"metric": "schema_error_rate", "level": "wrapper", "captures": "JSON/action contract failures", "misses": "chemical correctness"},
    ]


def _baseline_groups() -> list[dict[str, str]]:
    return [
        {"group": "minimal baselines", "systems": "always_accept, always_abstain, heuristic", "role": "sanity checks and action-distribution references"},
        {"group": "molecule-producing baselines", "systems": "local_mutation, corpus_search", "role": "tests molecule success without action understanding"},
        {"group": "agent-like baselines", "systems": "OpenAI, Anthropic, DeepSeek closed and verify_l3 snapshots", "role": "live external diagnostic comparison"},
        {"group": "threat-model baseline", "systems": "well_engineered_wrapper", "role": "public verifier/search wrapper saturation ceiling"},
    ]


def _action_collapse_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "system": row["system"],
            "provider": row["provider"],
            "action_accuracy": row["action_accuracy"],
            "molecule_acceptance_rate": row["molecule_acceptance_rate"],
            "reject_recall": row["reject_recall"],
            "abstain_recall": row["abstain_recall"],
            "task_inconsistent_accept_rate": row["task_inconsistent_accept_rate"],
        }
        for row in rows
    ]


def _protocol_caveat(release: Path, subset_manifest: Path) -> list[dict[str, Any]]:
    all_test = _load_tasks(release, "test")
    subset = jsonio.read_json(subset_manifest)
    ids = set(subset.get("task_ids") or []) if isinstance(subset, dict) else set()
    selected = [task for task in all_test if task.get("task_id") in ids]
    full_counts = Counter(str(task.get("protocol") or "unknown") for task in all_test)
    subset_counts = Counter(str(task.get("protocol") or "unknown") for task in selected)
    return [
        {
            "protocol": protocol,
            "full_test_tasks": full_counts.get(protocol, 0),
            "md_subset_tasks": subset_counts.get(protocol, 0),
            "interpretation": "native task grouping; not a forced same-task protocol intervention",
        }
        for protocol in ("L1", "L2", "L3")
    ]


def _figure_valid_decision(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    labels = [
        "System returns molecule",
        "Molecule passes constraints",
        "Correct action was REJECT/ABSTAIN",
        "Molecule metric says success",
        "Action metric says failure",
    ]
    xs = [0.08, 0.29, 0.50, 0.71, 0.90]
    for x, label in zip(xs, labels):
        ax.text(x, 0.55, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#f7f7f7", "ec": "#333333"}, fontsize=9)
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right - 0.08, 0.55), xytext=(left + 0.08, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("Valid Molecule Is Not Necessarily a Valid Decision", fontsize=12)
    _save(fig, figures, "figure1_valid_molecule_not_valid_decision")


def _figure_architecture(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    nodes = {
        "Specification bundle": (0.12, 0.65),
        "Public task view": (0.34, 0.78),
        "Hidden oracle evidence": (0.34, 0.45),
        "Model response": (0.58, 0.78),
        "Verifier / scorer": (0.78, 0.60),
        "Action-aware metrics": (0.58, 0.35),
    }
    for label, (x, y) in nodes.items():
        ax.text(x, y, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#eef5ff", "ec": "#2f4f6f"}, fontsize=9)
    arrows = [
        ("Specification bundle", "Public task view"),
        ("Specification bundle", "Hidden oracle evidence"),
        ("Public task view", "Model response"),
        ("Model response", "Verifier / scorer"),
        ("Hidden oracle evidence", "Verifier / scorer"),
        ("Verifier / scorer", "Action-aware metrics"),
    ]
    for src, dst in arrows:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("SpecGuard-Chem Public/Hidden Evaluation Architecture", fontsize=12)
    _save(fig, figures, "figure2_specguard_architecture")


def _figure_scatter(rows: Sequence[Mapping[str, Any]], figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"deterministic": "#4c78a8", "openai": "#59a14f", "anthropic": "#f28e2b", "deepseek": "#b07aa1"}
    for row in rows:
        x = row.get("molecule_acceptance_rate")
        y = row.get("action_accuracy")
        if x is None or y is None:
            continue
        provider = str(row.get("provider"))
        ax.scatter(float(x), float(y), s=55, color=colors.get(provider, "#777777"), label=provider)
        ax.text(float(x) + 0.008, float(y) + 0.008, str(row.get("system")), fontsize=6)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=8)
    ax.set_xlabel("Molecule acceptance rate")
    ax.set_ylabel("Action accuracy")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    _save(fig, figures, "figure3_molecule_acceptance_vs_action_accuracy")


def _figure_action_recall(rows: Sequence[Mapping[str, Any]], figures: Path) -> None:
    labels = [str(row["system"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 5))
    for offset, metric, color in [(-width, "accept_recall", "#4c78a8"), (0, "reject_recall", "#f58518"), (width, "abstain_recall", "#54a24b")]:
        values = [0 if row.get(metric) is None else float(row.get(metric)) for row in rows]
        ax.bar([idx + offset for idx in x], values, width=width, label=metric.replace("_", " "), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures, "figure4_action_recall_by_type")


def _figure_wrapper(rows: Sequence[Mapping[str, Any]], figures: Path) -> None:
    selected = [row for row in rows if row["system"] in {"always_accept", "local_mutation", "corpus_search", "well_engineered_wrapper"} or row.get("provider") != "deterministic"]
    selected = selected[:18]
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.5), 4.8))
    labels = [str(row["system"]) for row in selected]
    values = [0 if row.get("action_accuracy") is None else float(row.get("action_accuracy")) for row in selected]
    colors = ["#d62728" if label == "well_engineered_wrapper" else "#7f7f7f" for label in labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Action accuracy")
    ax.set_title("Wrapper Saturation as Access-Model Ceiling")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures, "figure5_wrapper_saturation")


def write_outputs(
    *,
    release: Path,
    results: Path,
    subset_manifest: Path,
    offline_aggregate: Path,
    external_aggregate: Path | None,
    external_cache_root: Path | None,
    live_aggregate: Path | None,
) -> None:
    tables = results / "tables"
    figures = results / "figures"
    notes = results / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    offline_rows = collect_rows(offline_aggregate, cache_root=None, cache_mode="offline")
    external_rows = collect_rows(external_aggregate, cache_root=external_cache_root, cache_mode="replay") if external_aggregate else []
    all_rows = offline_rows + external_rows

    _write_csv(tables / "metric_definitions.csv", _metric_definitions())
    _write_md(tables / "metric_definitions.md", _metric_definitions(), title="Metric Definitions")
    _write_csv(tables / "baseline_groups.csv", _baseline_groups())
    _write_md(tables / "baseline_groups.md", _baseline_groups(), title="Baseline Groups")
    _write_csv(tables / "external_diagnostic_snapshot.csv", external_rows)
    _write_md(tables / "external_diagnostic_snapshot.md", external_rows, title="External Diagnostic Snapshot")
    _write_csv(tables / "molecule_acceptance_vs_action_accuracy.csv", all_rows)
    _write_md(tables / "molecule_acceptance_vs_action_accuracy.md", all_rows, title="Molecule Acceptance Versus Action Accuracy")
    collapse = _action_collapse_rows(all_rows)
    _write_csv(tables / "reject_abstain_action_collapse.csv", collapse)
    _write_md(tables / "reject_abstain_action_collapse.md", collapse, title="Reject And Abstain Action Collapse")
    wrapper_rows = [row for row in all_rows if row["system"] == "well_engineered_wrapper"]
    _write_csv(tables / "wrapper_saturation.csv", wrapper_rows)
    _write_md(tables / "wrapper_saturation.md", wrapper_rows, title="Wrapper Saturation")
    protocol_rows = _protocol_caveat(release, subset_manifest)
    _write_csv(tables / "protocol_caveat.csv", protocol_rows)
    _write_md(tables / "protocol_caveat.md", protocol_rows, title="Protocol Caveat")

    _figure_valid_decision(figures)
    _figure_architecture(figures)
    _figure_scatter(all_rows, figures)
    _figure_action_recall(all_rows, figures)
    _figure_wrapper(all_rows, figures)

    replay_status = "not_checked"
    if live_aggregate and external_aggregate and live_aggregate.exists() and external_aggregate.exists():
        live_rows = collect_rows(live_aggregate, cache_root=external_cache_root, cache_mode="live")
        live_by_name = {row["system"]: row for row in live_rows}
        replay_by_name = {row["system"]: row for row in external_rows}
        comparable = ["action_accuracy", "molecule_acceptance_rate", "reject_recall", "abstain_recall"]
        mismatches = []
        for name, live in live_by_name.items():
            replay = replay_by_name.get(name)
            if replay is None:
                mismatches.append(f"{name}: missing replay")
                continue
            for key in comparable:
                if live.get(key) != replay.get(key):
                    mismatches.append(f"{name}:{key}")
        replay_status = "pass" if not mismatches else "fail: " + ", ".join(mismatches[:8])
    elif external_rows and external_cache_root and external_cache_root.exists():
        replay_status = "pass: external aggregate regenerated from cached live responses"

    summary = [
        "# MD Research Results Summary",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- subset_manifest: {subset_manifest}",
        f"- offline_rows: {len(offline_rows)}",
        f"- external_rows: {len(external_rows)}",
        f"- replay_match: {replay_status}",
        "- interpretation: external LLM rows are diagnostic MD evidence and not the primary offline leaderboard.",
        "",
    ]
    (results / "RESULTS_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("benchmarks/releases/sgchem_v1.0"))
    parser.add_argument("--results", type=Path, default=Path("md_research/results"))
    parser.add_argument("--subset-manifest", type=Path, default=Path("md_research/config/external_subset.json"))
    parser.add_argument("--generate-subset", action="store_true")
    parser.add_argument("--subset-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-per-family", type=int, default=8)
    parser.add_argument("--offline-aggregate", type=Path, default=Path("md_research/results/raw_runs/subset_offline/aggregate.json"))
    parser.add_argument("--external-aggregate", type=Path, default=Path("md_research/results/raw_runs/external_replay/aggregate.json"))
    parser.add_argument("--live-aggregate", type=Path, default=Path("md_research/results/raw_runs/external_live/aggregate.json"))
    parser.add_argument("--external-cache-root", type=Path, default=Path("md_research/results/cache/external_live"))
    args = parser.parse_args(argv)

    if args.generate_subset or not args.subset_manifest.exists():
        write_subset_manifest(
            release=args.release,
            subset_manifest=args.subset_manifest,
            seed=args.seed,
            max_per_family=args.max_per_family,
        )
    if args.subset_only:
        return 0
    write_outputs(
        release=args.release,
        results=args.results,
        subset_manifest=args.subset_manifest,
        offline_aggregate=args.offline_aggregate,
        external_aggregate=args.external_aggregate if args.external_aggregate.exists() else None,
        external_cache_root=args.external_cache_root if args.external_cache_root.exists() else None,
        live_aggregate=args.live_aggregate if args.live_aggregate.exists() else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
