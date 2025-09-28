from __future__ import annotations

"""Configuration, schema models, and data-loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

from .utils import jsonio

ConstraintType = Literal["hard", "soft"]


class ConstraintModel(BaseModel):
    """Represents a single constraint entry loaded from a spec file."""

    id: str
    type: ConstraintType
    check: str
    params: Dict[str, Any] = Field(default_factory=dict)
    severity: Optional[str] = None
    weight: float = 1.0


class BehaviourModel(BaseModel):
    """Behavioural policy block embedded in specs."""

    interrupt_policy: str
    abstain_policy: Optional[Dict[str, Any]] = None


class SpecModel(BaseModel):
    """Top-level spec definition."""

    id: str
    version: int
    constraints: List[ConstraintModel]
    behaviour: BehaviourModel


class TaskScoringModel(BaseModel):
    primary: str
    secondary: Optional[str] = None


class TaskInputModel(BaseModel):
    smiles: Optional[str] = None


class TaskModel(BaseModel):
    """Schema for a single evaluation task."""

    task_id: str
    suite: str
    protocol: Literal["L1", "L2", "L3"]
    prompt: str
    input: TaskInputModel
    spec_id: str
    scoring: TaskScoringModel
    interrupt_at_step: Optional[int] = None


class FailureItem(BaseModel):
    """One entry in a failure vector."""

    id: str
    detail: Optional[str] = None
    delta: Optional[float] = None
    distance_to_bound: Optional[float] = None


class FailureVector(BaseModel):
    """Structured view of constraint outcomes handed back to an agent."""

    hard_fails: List[FailureItem] = Field(default_factory=list)
    soft_misses: List[FailureItem] = Field(default_factory=list)
    margins: List[FailureItem] = Field(default_factory=list)
    round: int


@dataclass(frozen=True)
class ProjectPaths:
    """Convenience holder for canonical project directories."""

    project_root: Path
    data_dir: Path
    specs_dir: Path
    suites_dir: Path


def _default_paths() -> ProjectPaths:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    return ProjectPaths(
        project_root=root,
        data_dir=data_dir,
        specs_dir=data_dir / "specs",
        suites_dir=root / "tasks" / "suites",
    )


PATHS = _default_paths()


class SpecNotFoundError(FileNotFoundError):
    """Raised when a spec identifier cannot be located."""


class TaskSuiteNotFoundError(FileNotFoundError):
    """Raised when a task suite cannot be located."""


def load_spec(spec_id: str, *, paths: ProjectPaths = PATHS) -> SpecModel:
    """Load a spec by identifier from the data directory."""

    spec_path = paths.specs_dir / f"{spec_id}.yaml"
    if not spec_path.exists():
        raise SpecNotFoundError(f"Unknown spec '{spec_id}' at {spec_path}")
    with spec_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    try:
        return SpecModel.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - validation details vary
        raise ValueError(f"Invalid spec data: {exc}") from exc


def load_tasks_for_suite(
    suite_name: str, *, paths: ProjectPaths = PATHS
) -> List[TaskModel]:
    """Read all tasks for a given suite."""

    suite_path = paths.suites_dir / f"{suite_name}.jsonl"
    if not suite_path.exists():
        raise TaskSuiteNotFoundError(f"Unknown task suite '{suite_name}' at {suite_path}")
    tasks: List[TaskModel] = []
    for entry in jsonio.read_jsonl(suite_path):
        try:
            tasks.append(TaskModel.model_validate(entry))
        except ValidationError as exc:
            raise ValueError(f"Invalid task payload in {suite_path}: {exc}") from exc
    return tasks


def ensure_dirs(path: Path) -> None:
    """Create parent directories for *path* if they do not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def list_available_specs(paths: ProjectPaths = PATHS) -> List[str]:
    """Return the set of known spec identifiers."""

    return sorted(p.stem for p in paths.specs_dir.glob("*.yaml"))


def list_available_suites(paths: ProjectPaths = PATHS) -> List[str]:
    """Return the set of known suite identifiers."""

    return sorted(p.stem for p in paths.suites_dir.glob("*.jsonl"))


def select_tasks(
    tasks: Iterable[TaskModel],
    *,
    protocol: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[TaskModel]:
    """Filter tasks by protocol and limit taken from the front of the list."""

    filtered = [t for t in tasks if protocol is None or t.protocol == protocol]
    if limit is not None:
        return filtered[: max(limit, 0)]
    return filtered
