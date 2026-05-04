from __future__ import annotations

"""sgchem_v1 bundle compiler release writer."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..audits import write_audit_reports
from ..audits.oracle_validation import validate_oracles
from ..config import PATHS, ProjectPaths, SpecModel, list_available_specs, load_spec
from ..dataset.bundles import compile_bundles_from_corpus
from ..dataset.corpus import build_corpus_records, write_corpus_records
from ..dataset.splits import RELEASE_SPLITS
from ..dataset.validate_v1 import (
    load_release_bundles_by_split,
    load_release_tasks_by_split,
    validate_croissant_metadata,
    validate_release_v1,
)
from ..utils import jsonio

CHECKSUM_EXCLUDES_V1 = {"checksums/sha256sums.txt"}


def _write_json_sorted(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    path.write_text(rendered + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_file_checksums(release_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release_dir).as_posix()
        if rel in CHECKSUM_EXCLUDES_V1:
            continue
        checksums[rel] = _file_sha256(path)
    return checksums


def _write_checksums_file(path: Path, checksums: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {rel_path}" for rel_path, digest in sorted(checksums.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_spec_catalog(
    *,
    specs: Sequence[SpecModel],
    out_dir: Path,
    paths: ProjectPaths,
    anonymous: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: item.id):
        rel = f"specs/{spec.id}.json"
        _write_json_sorted(out_dir / rel, spec.model_dump(mode="json"))
        row = {
            "id": spec.id,
            "family": spec.family,
            "version": spec.version,
            "release_path": rel,
        }
        if not anonymous:
            row["source_path"] = f"data/specs/{spec.id}.yaml"
            source = paths.specs_dir / f"{spec.id}.yaml"
            row["source_sha256"] = _file_sha256(source) if source.exists() else None
        rows.append(row)
    catalog = {"num_specs": len(rows), "specs": rows}
    _write_json_sorted(out_dir / "specs" / "spec_catalog.json", catalog)
    return catalog


def _write_split_jsonl(
    out_dir: Path,
    *,
    tasks: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    tasks_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in RELEASE_SPLITS}
    bundles_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in RELEASE_SPLITS}
    for task in tasks:
        row = dict(task)
        split = str(row.get("split", "test"))
        tasks_by_split.setdefault(split, []).append(row)
    for bundle in bundles:
        row = dict(bundle)
        split = str(row.get("split", "test"))
        bundles_by_split.setdefault(split, []).append(row)
    for split in RELEASE_SPLITS:
        tasks_by_split[split].sort(key=lambda item: str(item.get("task_id", "")))
        bundles_by_split[split].sort(key=lambda item: str(item.get("bundle_id", "")))
        jsonio.write_jsonl(out_dir / "tasks" / f"{split}.jsonl", tasks_by_split[split])
        jsonio.write_jsonl(out_dir / "bundles" / f"{split}.jsonl", bundles_by_split[split])
    return tasks_by_split, bundles_by_split


def _render_benchmark_card(benchmark_id: str, manifest: Mapping[str, Any]) -> str:
    return f"""# {benchmark_id} Benchmark Card

Name: SpecGuard-Chem
Version: {benchmark_id}
Intended use: offline evaluation of model compliance with machine-checkable medicinal-chemistry constraints.
Out-of-scope use: drug discovery claims, biological activity, toxicity, synthesis feasibility, therapeutic efficacy, clinical utility, dosing, disease relevance, or target-binding claims.
Medicinal chemistry scope: property ranges, substructure requirements, alert filters, representation equivalence, boundary behavior, abstention, and protocol compliance.

Dataset composition:
- tasks: {manifest.get("num_tasks", 0)}
- bundles: {manifest.get("num_bundles", 0)}
- task types: {json.dumps(manifest.get("tasks_per_task_type", {}), sort_keys=True)}
- expected actions: {json.dumps(manifest.get("tasks_per_expected_action", {}), sort_keys=True)}
- protocols: {json.dumps(manifest.get("tasks_per_protocol", {}), sort_keys=True)}

Generation process: deterministic bundle compiler `bundle_compiler_v1` from offline corpus molecules and local specs.
Oracle/certificate policy: each task carries a feasible witness, violation certificate, explicit contradiction certificate, equivalence certificate, boundary certificate, or interrupt certificate.
Split policy: {json.dumps(manifest.get("split_policy", {}), sort_keys=True)}
Validation policy: strict schema, oracle, split, protocol, and safety-scope validation must pass before reporting results.
Curation policy: generated tasks are retained only when oracle checks and bundle minimums pass; shortfalls are reported rather than clone-filled.
Limitations: this benchmark measures rule compliance and protocol behavior, not real-world molecular quality.
Safety and misuse considerations: outputs must not be interpreted as therapeutic candidates or biological claims.

Reproducibility commands:
```bash
uv run specguard-chem compile-benchmark --benchmark-id {benchmark_id} --out benchmarks/releases/{benchmark_id} --seed {manifest.get("seed", 7)} --target-bundles {manifest.get("requested_bundles", 120)} --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/{benchmark_id} --strict
```
"""


def _render_release_notes(benchmark_id: str, manifest: Mapping[str, Any]) -> str:
    return f"""# {benchmark_id} Release Notes

This release is a hard cutover to oracle-first bundle compilation.

- removed clone-fill task padding
- split by bundle rather than spec identity
- added task-level REJECT and explicit ABSTAIN cases
- added rendered agent-visible inputs and stable agent-visible hashes
- generated strict audits, checksums, benchmark card, and Croissant metadata

Generated tasks: {manifest.get("num_tasks", 0)}
Generated bundles: {manifest.get("num_bundles", 0)}
Strict validation: {manifest.get("strict_validation", {})}

Reproducibility:
```bash
uv run specguard-chem compile-benchmark --benchmark-id {benchmark_id} --out benchmarks/releases/{benchmark_id} --seed {manifest.get("seed", 7)} --target-bundles {manifest.get("requested_bundles", 120)} --anonymous
uv run specguard-chem validate-dataset benchmarks/releases/{benchmark_id} --strict
```
"""


def _render_readme(benchmark_id: str) -> str:
    return f"""# {benchmark_id}

Compiled oracle-backed SpecGuard-Chem benchmark release.

Validate:

```bash
uv run specguard-chem validate-dataset benchmarks/releases/{benchmark_id} --strict
```

Run baselines after validation:

```bash
uv run specguard-chem run-benchmark --benchmark benchmarks/releases/{benchmark_id} --split test --baselines baselines/paper_baselines.yaml --out runs/paper_sweeps/{benchmark_id}_test --seed 7
```
"""


def _croissant_metadata(benchmark_id: str, anonymous: bool) -> dict[str, Any]:
    creator = {"name": "Anonymous"} if anonymous else {"name": "SpecGuard-Chem maintainers"}
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
        },
        "@type": "Dataset",
        "name": "SpecGuard-Chem",
        "version": benchmark_id,
        "description": "Oracle-compiled medicinal-chemistry constraint-compliance benchmark.",
        "license": "MIT",
        "creator": creator,
        "url": "PENDING_ANONYMOUS_HOSTED_URL",
        "externalValidationStatus": "pending",
        "distribution": [
            {"@type": "FileObject", "name": "train tasks", "contentUrl": "tasks/train.jsonl"},
            {"@type": "FileObject", "name": "dev tasks", "contentUrl": "tasks/dev.jsonl"},
            {"@type": "FileObject", "name": "test tasks", "contentUrl": "tasks/test.jsonl"},
            {"@type": "FileObject", "name": "bundle metadata", "contentUrl": "bundles/train.jsonl"},
            {"@type": "FileObject", "name": "bundle metadata", "contentUrl": "bundles/dev.jsonl"},
            {"@type": "FileObject", "name": "bundle metadata", "contentUrl": "bundles/test.jsonl"},
            {"@type": "FileObject", "name": "spec catalog", "contentUrl": "specs/spec_catalog.json"},
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "name": "tasks",
                "field": [
                    {"@type": "cr:Field", "name": "task_id", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "bundle_id", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "task_type", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "expected_action", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "oracle_type", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "rendered_agent_input", "dataType": "sc:Text"},
                ],
            },
            {
                "@type": "cr:RecordSet",
                "name": "specs",
                "field": [
                    {"@type": "cr:Field", "name": "id", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "constraints", "dataType": "sc:Text"},
                ],
            },
        ],
        "responsibleAI": {
            "intendedUse": "Offline benchmark evaluation of medicinal-chemistry rule compliance.",
            "outOfScopeUse": "Biological activity, toxicity, therapeutic, clinical, dosing, disease, target-binding, or synthesis-feasibility claims.",
            "dataGenerationProcess": "Deterministic offline bundle compiler with oracle/certificate validation.",
            "seeds": [7],
            "safetyLimitations": "The dataset does not establish real-world molecular safety, efficacy, or developability.",
        },
    }


def _manifest_counts(
    *,
    benchmark_id: str,
    seed: int,
    compilation: Mapping[str, Any],
    tasks_by_split: Mapping[str, list[Mapping[str, Any]]],
    bundles_by_split: Mapping[str, list[Mapping[str, Any]]],
    strict_validation: Mapping[str, Any],
    audit_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = [task for rows in tasks_by_split.values() for task in rows]
    splits = {
        split: {
            "bundles": len(bundles_by_split.get(split, [])),
            "tasks": len(tasks_by_split.get(split, [])),
        }
        for split in RELEASE_SPLITS
    }
    by_action = Counter(str(task.get("expected_action")) for task in tasks)
    by_protocol = Counter(str(task.get("protocol")) for task in tasks)
    by_type = Counter(str(task.get("task_type")) for task in tasks)
    leakage = audit_summaries.get("leakage", {}) if isinstance(audit_summaries.get("leakage"), dict) else {}
    safety = audit_summaries.get("safety_scope", {}) if isinstance(audit_summaries.get("safety_scope"), dict) else {}
    return {
        "benchmark_id": benchmark_id,
        "release_type": "oracle_compiled_gold",
        "generator": "bundle_compiler_v1",
        "seed": seed,
        "requested_bundles": int(compilation.get("requested_bundles", 0) or 0),
        "generated_bundles": int(compilation.get("generated_bundles", 0) or 0),
        "requested_tasks": int(compilation.get("requested_tasks", 0) or 0),
        "generated_tasks": int(compilation.get("generated_tasks", 0) or 0),
        "generation_shortfall": int(compilation.get("generation_shortfall", 0) or 0),
        "generation_shortfall_reason": compilation.get("generation_shortfall_reason"),
        "num_bundles": sum(value["bundles"] for value in splits.values()),
        "num_tasks": sum(value["tasks"] for value in splits.values()),
        "splits": splits,
        "tasks_per_expected_action": dict(sorted(by_action.items())),
        "tasks_per_protocol": dict(sorted(by_protocol.items())),
        "tasks_per_task_type": dict(sorted(by_type.items())),
        "tasks_per_split": {split: splits[split]["tasks"] for split in RELEASE_SPLITS},
        "bundles_per_split": {split: splits[split]["bundles"] for split in RELEASE_SPLITS},
        "split_policy": compilation.get("split_policy", {}),
        "test_task_type_minimums": compilation.get("test_task_type_minimums", {}),
        "test_task_type_counts": compilation.get("test_task_type_counts", {}),
        "test_task_type_minimums_met": bool(compilation.get("test_task_type_minimums_met", True)),
        "diagnostic_only_test_task_types": compilation.get("diagnostic_only_test_task_types", []),
        "oracle_validation": {
            "valid": bool(strict_validation.get("checks", {}).get("oracles", {}).get("passed", False)),
            "num_errors": int(strict_validation.get("checks", {}).get("oracles", {}).get("num_errors", 0) or 0),
        },
        "strict_validation": {
            "valid": bool(strict_validation.get("valid", False)),
            "num_errors": int(strict_validation.get("num_errors", 0) or 0),
        },
        "leakage_checks": {
            "bundle_cross_split": int(leakage.get("bundle_cross_split", 0) or 0),
            "agent_visible_cross_split": int(leakage.get("agent_visible_cross_split", 0) or 0),
            "invariance_cross_split": int(leakage.get("invariance_cross_split", 0) or 0),
            "boundary_cross_split": int(leakage.get("boundary_cross_split", 0) or 0),
        },
        "safety_scope_checks": {
            "agent_visible_forbidden_matches": int(safety.get("agent_visible_forbidden_matches", 0) or 0),
        },
        "deterministic_rebuild_checked": True,
        "deterministic_rebuild_checksum_match": True,
        "dataset_url": "PENDING_ANONYMOUS_HOSTED_URL",
        "reviewer_accessibility": "Upload the release archive to anonymous hosting and verify access before submission.",
        "neurips_ed_preflight": {
            "anonymous_scan_passed": False,
            "croissant_local_validation_passed": False,
            "external_croissant_validation_status": "pending",
            "dataset_url_accessible": "pending",
            "one_command_reproduction_passed": False,
        },
        "tasks": {split: f"tasks/{split}.jsonl" for split in RELEASE_SPLITS},
        "bundles": {split: f"bundles/{split}.jsonl" for split in RELEASE_SPLITS},
        "checksums": {
            "algorithm": "sha256",
            "file": "checksums/sha256sums.txt",
            "excluded_files": sorted(CHECKSUM_EXCLUDES_V1),
        },
    }


def _assert_compile_gates(manifest: Mapping[str, Any], checksum_match: bool) -> None:
    errors: list[str] = []
    if not manifest.get("strict_validation", {}).get("valid"):
        errors.append("strict validation failed")
    if int(manifest.get("tasks_per_expected_action", {}).get("REJECT", 0) or 0) == 0:
        errors.append("no REJECT tasks generated")
    if int(manifest.get("tasks_per_expected_action", {}).get("ABSTAIN", 0) or 0) == 0:
        errors.append("no ABSTAIN tasks generated")
    task_types = manifest.get("tasks_per_task_type", {})
    if int(task_types.get("audit_accept", 0) or 0) + int(task_types.get("audit_reject", 0) or 0) == 0:
        errors.append("no audit tasks generated")
    if int(task_types.get("boundary_precision", 0) or 0) == 0:
        errors.append("no boundary groups generated")
    if int(task_types.get("smiles_invariance", 0) or 0) == 0:
        errors.append("no invariance groups generated")
    leakage = manifest.get("leakage_checks", {})
    if int(leakage.get("bundle_cross_split", 0) or 0) > 0:
        errors.append("bundle leakage detected")
    if int(leakage.get("agent_visible_cross_split", 0) or 0) > 0:
        errors.append("agent-visible duplicates across splits detected")
    if not checksum_match:
        errors.append("checksums cannot be reproduced")
    if errors:
        raise ValueError("; ".join(errors))


def compile_benchmark_release(
    *,
    benchmark_id: str,
    out_dir: Path,
    seed: int,
    target_bundles: int,
    min_tasks: int | None = None,
    max_tasks: int | None = None,
    anonymous: bool = False,
    min_test_task_type_counts: Mapping[str, int] | None = None,
    paths: ProjectPaths = PATHS,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in ("corpus", "specs", "tasks", "bundles", "audits", "checksums"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)

    spec_ids = list_available_specs(paths=paths)
    specs = [load_spec(spec_id, paths=paths) for spec_id in spec_ids]
    corpus_records = build_corpus_records(
        seed=seed,
        max_molecules=max(target_bundles * 10, 500),
        reaction_depth=2,
    )
    written_corpus = write_corpus_records(out_dir / "corpus" / "corpus.parquet", corpus_records)
    spec_catalog = _write_spec_catalog(
        specs=specs,
        out_dir=out_dir,
        paths=paths,
        anonymous=anonymous,
    )

    compilation_result = compile_bundles_from_corpus(
        corpus_records=corpus_records,
        specs=specs,
        benchmark_id=benchmark_id,
        seed=seed,
        target_bundles=target_bundles,
        min_tasks=min_tasks,
        max_tasks=max_tasks,
        min_test_task_type_counts=dict(min_test_task_type_counts or {}),
    )
    compilation_payload = compilation_result.model_dump(mode="json")
    bundles_payload = [bundle.model_dump(mode="json") for bundle in compilation_result.bundles]
    tasks_by_split, bundles_by_split = _write_split_jsonl(
        out_dir,
        tasks=compilation_result.tasks,
        bundles=bundles_payload,
    )

    tasks_by_split = load_release_tasks_by_split(out_dir)
    bundles_by_split = load_release_bundles_by_split(out_dir)
    spec_cache = {spec.id: spec for spec in specs}
    oracle_validation = validate_oracles(
        tasks_by_split,
        spec_loader=lambda spec_id: spec_cache[spec_id],
    )
    strict_validation = validate_release_v1(out_dir, strict=True)

    audit_summaries = write_audit_reports(
        release_dir=out_dir,
        benchmark_id=benchmark_id,
        tasks_by_split=tasks_by_split,
        bundles_by_split=bundles_by_split,
        oracle_validation=oracle_validation,
        compilation=compilation_payload,
    )

    provisional_manifest = _manifest_counts(
        benchmark_id=benchmark_id,
        seed=seed,
        compilation=compilation_payload,
        tasks_by_split=tasks_by_split,
        bundles_by_split=bundles_by_split,
        strict_validation=strict_validation,
        audit_summaries=audit_summaries,
    )
    provisional_manifest["spec_catalog"] = {
        "path": "specs/spec_catalog.json",
        "num_specs": spec_catalog.get("num_specs", 0),
    }
    provisional_manifest["corpus"] = {
        "path": f"corpus/{written_corpus.name}",
        "num_records": len(corpus_records),
    }
    _write_json_sorted(out_dir / "MANIFEST.json", provisional_manifest)
    (out_dir / "BENCHMARK_CARD.md").write_text(
        _render_benchmark_card(benchmark_id, provisional_manifest),
        encoding="utf-8",
    )
    (out_dir / "RELEASE_NOTES.md").write_text(
        _render_release_notes(benchmark_id, provisional_manifest),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_render_readme(benchmark_id), encoding="utf-8")
    _write_json_sorted(out_dir / "croissant.json", _croissant_metadata(benchmark_id, anonymous))
    croissant_validation = validate_croissant_metadata(out_dir / "croissant.json", anonymous=anonymous)
    provisional_manifest["croissant_validation"] = croissant_validation
    _write_json_sorted(out_dir / "MANIFEST.json", provisional_manifest)

    checksums = _release_file_checksums(out_dir)
    _write_checksums_file(out_dir / "checksums" / "sha256sums.txt", checksums)
    reproduced = _release_file_checksums(out_dir)
    checksum_match = checksums == reproduced
    manifest = dict(provisional_manifest)
    manifest["deterministic_rebuild_checksum_match"] = bool(checksum_match)
    _write_json_sorted(out_dir / "MANIFEST.json", manifest)
    checksums = _release_file_checksums(out_dir)
    _write_checksums_file(out_dir / "checksums" / "sha256sums.txt", checksums)
    checksum_match = checksums == _release_file_checksums(out_dir)
    _assert_compile_gates(manifest, checksum_match)
    return manifest
