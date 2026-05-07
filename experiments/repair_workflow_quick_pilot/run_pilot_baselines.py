from __future__ import annotations

import json
import random
from pathlib import Path

from pilot_lib import (
    evaluate_candidate_list,
    load_corpus,
    load_tasks,
    mutation_candidates,
    nearest_passing_corpus,
    render_md_table,
    summarize_baseline,
    write_json,
    write_jsonl,
)

OUT = Path(__file__).resolve().parent


def _construct_first_passing(task: dict, corpus: list[str]) -> list[str | None]:
    return nearest_passing_corpus(task, corpus, k=3)


def _run_baseline(name: str, tasks: list[dict], chooser) -> tuple[dict, list[dict]]:
    rows = []
    for task in tasks:
        candidates = chooser(task)
        row = evaluate_candidate_list(task, candidates)
        row["baseline"] = name
        rows.append(row)
    return summarize_baseline(name, rows), rows


def main() -> int:
    tasks = load_tasks()
    corpus = load_corpus()
    rng = random.Random(7)

    def no_op(task: dict) -> list[str | None]:
        return [task.get("starting_smiles") or None]

    def random_valid_corpus(task: dict) -> list[str | None]:
        sample = list(corpus)
        rng.shuffle(sample)
        return sample[:3]

    def nearest(task: dict) -> list[str | None]:
        return _construct_first_passing(task, corpus)

    def local_mutation(task: dict) -> list[str | None]:
        if task["family"] == "construct_under_rule_card":
            return nearest(task)
        return mutation_candidates(task, depth=1, k=3)

    def verifier_greedy(task: dict) -> list[str | None]:
        if task["family"] == "construct_under_rule_card":
            return nearest(task)
        return mutation_candidates(task, depth=2, k=3)

    summaries = []
    all_rows = []
    for name, chooser in [
        ("no_op_starting_molecule", no_op),
        ("random_valid_corpus", random_valid_corpus),
        ("nearest_passing_corpus_by_similarity", nearest),
        ("local_mutation_search", local_mutation),
        ("verifier_guided_greedy", verifier_greedy),
    ]:
        summary, rows = _run_baseline(name, tasks, chooser)
        summaries.append(summary)
        all_rows.extend(rows)
        write_jsonl(OUT / f"{name}_rows.jsonl", rows)

    write_json(OUT / "baseline_results.json", {"summaries": summaries})
    md = [
        "# Repair Workflow Pilot Baseline Results",
        "",
        render_md_table(summaries),
        "",
        "## Failure Counts",
        "",
    ]
    for summary in summaries:
        md.append(f"### {summary['baseline']}")
        md.append("")
        md.append("```json")
        md.append(json.dumps(summary["failure_counts"], indent=2, sort_keys=True))
        md.append("```")
        md.append("")
    (OUT / "baseline_results.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"summaries": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
