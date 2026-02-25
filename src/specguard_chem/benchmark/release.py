from __future__ import annotations

"""Frozen benchmark release creation and loading."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from rdkit import rdBase

from ..config import PATHS, ProjectPaths, SpecModel, TaskModel, list_available_specs, load_spec
from ..dataset.corpus import build_corpus_records, write_corpus_records
from ..dataset.tasks import generate_tasks_from_corpus, write_tasks_jsonl
from ..dataset.validate import validate_dataset_records
from ..utils import jsonio

RELEASE_SPLITS: tuple[str, ...] = ("train", "dev", "test")
CHECKSUM_EXCLUDES = {"MANIFEST.json", "checksums/sha256sums.txt"}


def _write_json_sorted(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    path.write_text(rendered + "\n", encoding="utf-8")


def _stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _stable_hash_bytes(path.read_bytes())


def _git_commit(project_root: Path) -> Optional[str]:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except OSError:
        return None
    return output or None


def _build_spec_catalog(
    *,
    specs: Iterable[SpecModel],
    out_dir: Path,
    paths: ProjectPaths,
) -> Dict[str, Any]:
    specs_dir = out_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: item.id):
        payload = spec.model_dump(mode="json")
        release_rel_path = f"specs/{spec.id}.json"
        _write_json_sorted(out_dir / release_rel_path, payload)
        source_path = paths.specs_dir / f"{spec.id}.yaml"
        rows.append(
            {
                "id": spec.id,
                "family": spec.family,
                "spec_split": spec.spec_split,
                "version": spec.version,
                "release_path": release_rel_path,
                "source_path": str(source_path),
                "source_sha256": _file_sha256(source_path) if source_path.exists() else None,
            }
        )
    catalog = {
        "num_specs": len(rows),
        "specs": rows,
    }
    _write_json_sorted(out_dir / "specs" / "spec_catalog.json", catalog)
    return catalog


def _task_split_counts(tasks_by_split: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    by_split = {split: len(tasks_by_split.get(split, [])) for split in RELEASE_SPLITS}
    by_suite: Dict[str, int] = Counter()
    by_protocol: Dict[str, int] = Counter()
    by_family: Dict[str, int] = Counter()
    by_split_suite_protocol: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for split in RELEASE_SPLITS:
        for task in tasks_by_split.get(split, []):
            suite = str(task.get("suite", "unknown"))
            protocol = str(task.get("protocol", "unknown"))
            family = str(task.get("task_family") or "unspecified")
            by_suite[suite] += 1
            by_protocol[protocol] += 1
            by_family[family] += 1
            by_split_suite_protocol[split][suite][protocol] += 1

    nested_counts = {
        split: {
            suite: dict(sorted(protocol_counts.items()))
            for suite, protocol_counts in sorted(by_split_suite_protocol[split].items())
        }
        for split in sorted(by_split_suite_protocol)
    }

    return {
        "total_tasks": sum(by_split.values()),
        "by_split": by_split,
        "by_suite": dict(sorted(by_suite.items())),
        "by_protocol": dict(sorted(by_protocol.items())),
        "by_task_family": dict(sorted(by_family.items())),
        "by_split_suite_protocol": nested_counts,
    }


def _release_file_checksums(release_dir: Path) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release_dir).as_posix()
        if rel in CHECKSUM_EXCLUDES:
            continue
        checksums[rel] = _file_sha256(path)
    return checksums


def _write_checksums_file(path: Path, checksums: Mapping[str, str]) -> None:
    lines = [f"{digest}  {rel_path}" for rel_path, digest in sorted(checksums.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _release_readme(
    *,
    benchmark_id: str,
    target_tasks: int,
    seed: int,
    counts: Mapping[str, Any],
) -> str:
    return (
        f"# {benchmark_id}\n\n"
        "Frozen benchmark release generated by `specguard-chem freeze-benchmark`.\n\n"
        f"- target_tasks: {target_tasks}\n"
        f"- seed: {seed}\n"
        f"- total_tasks: {counts.get('total_tasks', 0)}\n\n"
        "Use `specguard-chem run-benchmark` against this directory.\n"
    )


def _load_split_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = jsonio.read_jsonl(path)
    rows.sort(key=lambda item: str(item.get("task_id", "")))
    return rows


def validate_release_directory(
    release_dir: Path,
    *,
    repair_start_hard_fail_threshold: float = 0.70,
    near_miss_margin_band: float = 5.0,
    boundary_margin_band: float = 1.0,
    min_counts: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    tasks_by_split = {
        split: _load_split_file(release_dir / "tasks" / f"{split}.jsonl")
        for split in RELEASE_SPLITS
    }
    merged: List[Dict[str, Any]] = []
    for split in RELEASE_SPLITS:
        merged.extend(tasks_by_split.get(split, []))

    result = validate_dataset_records(
        merged,
        near_miss_margin_band=near_miss_margin_band,
        boundary_margin_band=boundary_margin_band,
        repair_start_hard_fail_threshold=repair_start_hard_fail_threshold,
        min_counts=min_counts,
    )
    split_counts = {split: len(tasks_by_split[split]) for split in RELEASE_SPLITS}
    result["split_counts"] = split_counts
    result["nonempty_splits"] = sorted(
        split for split, count in split_counts.items() if count > 0
    )
    return result


def freeze_benchmark_release(
    *,
    benchmark_id: str,
    out_dir: Path,
    target_tasks: int,
    seed: int,
    paths: ProjectPaths = PATHS,
    near_miss_margin_band: float = 5.0,
    boundary_margin_band: float = 1.0,
    repair_start_hard_fail_threshold: float = 0.70,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_ids = list_available_specs(paths=paths)
    specs = [load_spec(spec_id, paths=paths) for spec_id in spec_ids]

    corpus_records = build_corpus_records(
        seed=seed,
        max_molecules=max(target_tasks * 4, 500),
        reaction_depth=2,
    )
    corpus_path = out_dir / "corpus" / "corpus.parquet"
    written_corpus = write_corpus_records(corpus_path, corpus_records)

    generated_tasks = generate_tasks_from_corpus(
        corpus_records=corpus_records,
        specs=specs,
        target_tasks=target_tasks,
        seed=seed,
        suite_name=benchmark_id,
        near_miss_margin_band=near_miss_margin_band,
        boundary_margin_band=boundary_margin_band,
    )

    split_by_spec = {spec.id: spec.spec_split for spec in specs}
    tasks_by_split: Dict[str, List[Dict[str, Any]]] = {split: [] for split in RELEASE_SPLITS}
    for task in generated_tasks:
        spec_id = str(task.get("spec_id", ""))
        split = split_by_spec.get(spec_id, "test")
        tasks_by_split.setdefault(split, []).append(task)
    for split in RELEASE_SPLITS:
        tasks_by_split[split].sort(key=lambda item: str(item.get("task_id", "")))
        write_tasks_jsonl(out_dir / "tasks" / f"{split}.jsonl", tasks_by_split[split])

    spec_catalog = _build_spec_catalog(specs=specs, out_dir=out_dir, paths=paths)
    counts = _task_split_counts(tasks_by_split)
    (out_dir / "README.md").write_text(
        _release_readme(
            benchmark_id=benchmark_id,
            target_tasks=target_tasks,
            seed=seed,
            counts=counts,
        ),
        encoding="utf-8",
    )

    validation = validate_release_directory(
        out_dir,
        repair_start_hard_fail_threshold=repair_start_hard_fail_threshold,
        near_miss_margin_band=near_miss_margin_band,
        boundary_margin_band=boundary_margin_band,
    )
    if not validation.get("valid", False):
        first_errors = validation.get("errors", [])[:10]
        rendered = "; ".join(str(item) for item in first_errors)
        raise ValueError(f"Benchmark release validation failed: {rendered}")

    checksums = _release_file_checksums(out_dir)
    checksums_path = out_dir / "checksums" / "sha256sums.txt"
    _write_checksums_file(checksums_path, checksums)

    manifest = {
        "benchmark_id": benchmark_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(paths.project_root),
        "rdkit_version": rdBase.rdkitVersion,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "generation": {
            "seed": seed,
            "target_tasks": target_tasks,
            "suite_mix": dict(sorted(counts.get("by_suite", {}).items())),
            "near_miss_margin_band": near_miss_margin_band,
            "boundary_margin_band": boundary_margin_band,
            "repair_start_hard_fail_threshold": repair_start_hard_fail_threshold,
        },
        "counts": counts,
        "spec_catalog": {
            "path": "specs/spec_catalog.json",
            "num_specs": spec_catalog.get("num_specs", 0),
        },
        "corpus": {
            "path": (
                "corpus/corpus.parquet"
                if written_corpus.name == "corpus.parquet"
                else f"corpus/{written_corpus.name}"
            ),
            "num_records": len(corpus_records),
        },
        "tasks": {split: f"tasks/{split}.jsonl" for split in RELEASE_SPLITS},
        "validator": validation,
        "checksums": {
            "algorithm": "sha256",
            "file": "checksums/sha256sums.txt",
            "excluded_files": sorted(CHECKSUM_EXCLUDES),
            "files": checksums,
        },
    }
    _write_json_sorted(out_dir / "MANIFEST.json", manifest)
    return manifest


@dataclass(frozen=True)
class BenchmarkRelease:
    benchmark_id: str
    release_dir: Path
    manifest: Dict[str, Any]
    spec_catalog: Dict[str, Any]
    spec_loader: Callable[[str], SpecModel]

    def load_split_tasks(self, split: str) -> List[TaskModel]:
        split_path = self.release_dir / "tasks" / f"{split}.jsonl"
        rows = jsonio.read_jsonl(split_path)
        tasks = [TaskModel.model_validate(row) for row in rows]
        tasks.sort(key=lambda item: item.task_id)
        return tasks


def load_benchmark_release(release_dir: Path) -> BenchmarkRelease:
    manifest_path = release_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing release manifest at {manifest_path}")
    manifest = jsonio.read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Release manifest must be a JSON object")

    catalog_path = release_dir / "specs" / "spec_catalog.json"
    spec_catalog = jsonio.read_json(catalog_path)
    if not isinstance(spec_catalog, dict):
        raise ValueError("spec_catalog.json must be a JSON object")

    spec_cache: Dict[str, SpecModel] = {}

    def _spec_loader(spec_id: str) -> SpecModel:
        if spec_id in spec_cache:
            return spec_cache[spec_id]
        spec_path = release_dir / "specs" / f"{spec_id}.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"Missing frozen spec '{spec_id}' at {spec_path}")
        payload = jsonio.read_json(spec_path)
        spec = SpecModel.model_validate(payload)
        spec_cache[spec_id] = spec
        return spec

    benchmark_id = str(manifest.get("benchmark_id") or release_dir.name)
    return BenchmarkRelease(
        benchmark_id=benchmark_id,
        release_dir=release_dir,
        manifest=manifest,
        spec_catalog=spec_catalog,
        spec_loader=_spec_loader,
    )
