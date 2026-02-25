from __future__ import annotations

"""Configuration, schema models, and data-loading utilities."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .utils import jsonio

ConstraintType = Literal["hard", "soft"]
ExpectedOutcome = Literal["PASS", "ABSTAIN", "FAIL"]
ExpectedAction = Literal["ACCEPT", "ABSTAIN", "REJECT"]
InterruptAction = Literal["ABSTAIN", "CONTINUE"]
SpecCheck = Literal[
    "property_bounds",
    "alert_set_absent",
    "alert_set_present",
    "substructure_present",
    "substructure_absent",
    "sa_proxy_max",
    "similarity_min_to_input",
]


def legacy_expected_to_action(expected: ExpectedOutcome) -> ExpectedAction:
    """Translate legacy expected labels into decision-level actions."""

    if expected == "PASS":
        return "ACCEPT"
    if expected == "ABSTAIN":
        return "ABSTAIN"
    return "REJECT"


class BoundsModel(BaseModel):
    """Inclusive numeric bound."""

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float

    @model_validator(mode="after")
    def _validate_range(self) -> "BoundsModel":
        if self.min > self.max:
            raise ValueError("Bounds require min <= max")
        return self


class CountRangeModel(BaseModel):
    """Inclusive integer count bounds."""

    model_config = ConfigDict(extra="forbid")

    min: Optional[int] = Field(default=None, ge=0)
    max: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_count_range(self) -> "CountRangeModel":
        if self.min is None and self.max is None:
            raise ValueError("Count range requires at least one of min/max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Count range requires min <= max")
        return self


class PropertyBoundsParamsModel(BaseModel):
    """Parameter schema for property_bounds checks."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["all", "any"] = "all"
    bounds: Dict[str, BoundsModel]

    @model_validator(mode="before")
    @classmethod
    def _normalize_bounds(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError("property_bounds params must be an object")
        bounds = data.get("bounds")
        if not isinstance(bounds, dict):
            raise ValueError("property_bounds params require a bounds object")
        normalized: Dict[str, Dict[str, float]] = {}
        for name, raw in bounds.items():
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                normalized[str(name)] = {"min": float(raw[0]), "max": float(raw[1])}
                continue
            if isinstance(raw, dict) and "min" in raw and "max" in raw:
                normalized[str(name)] = {
                    "min": float(raw["min"]),
                    "max": float(raw["max"]),
                }
                continue
            raise ValueError(
                f"Invalid bounds payload for property '{name}'; expected [min,max] or object"
            )
        payload = dict(data)
        payload["bounds"] = normalized
        return payload

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "PropertyBoundsParamsModel":
        if not self.bounds:
            raise ValueError("property_bounds requires at least one bound")
        return self


class AlertSetParamsModel(BaseModel):
    """Parameter schema for alert_set_absent/present checks."""

    model_config = ConfigDict(extra="forbid")

    alert_set: str
    min_hits: Optional[int] = Field(default=None, ge=0)


class SubstructureParamsModel(BaseModel):
    """Parameter schema for substructure present/absent checks."""

    model_config = ConfigDict(extra="forbid")

    smarts_id: str
    count: Optional[CountRangeModel] = None


class SAProxyMaxParamsModel(BaseModel):
    """Parameter schema for sa_proxy_max checks."""

    model_config = ConfigDict(extra="forbid")

    max: float


class SimilarityMinToInputParamsModel(BaseModel):
    """Parameter schema for similarity_min_to_input checks."""

    model_config = ConfigDict(extra="forbid")

    min: float = Field(ge=0.0, le=1.0)
    fp: Literal["morgan"] = "morgan"
    radius: int = Field(default=2, ge=1, le=4)
    nBits: int = Field(default=2048, ge=128, le=8192)


class ConstraintModel(BaseModel):
    """Represents a single strict v2 constraint entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: ConstraintType
    check: SpecCheck
    params: Dict[str, Any] = Field(default_factory=dict)
    severity: Optional[str] = None
    weight: float = 1.0

    @model_validator(mode="after")
    def _validate_params(self) -> "ConstraintModel":
        if self.check == "property_bounds":
            parsed = PropertyBoundsParamsModel.model_validate(self.params)
            self.params = parsed.model_dump(mode="json")
            return self

        if self.check in {"alert_set_absent", "alert_set_present"}:
            parsed = AlertSetParamsModel.model_validate(self.params)
            if self.check == "alert_set_present" and parsed.min_hits is None:
                parsed.min_hits = 1
            self.params = parsed.model_dump(mode="json", exclude_none=True)
            return self

        if self.check in {"substructure_present", "substructure_absent"}:
            parsed = SubstructureParamsModel.model_validate(self.params)
            self.params = parsed.model_dump(mode="json", exclude_none=True)
            return self

        if self.check == "sa_proxy_max":
            parsed = SAProxyMaxParamsModel.model_validate(self.params)
            self.params = parsed.model_dump(mode="json")
            return self

        if self.check == "similarity_min_to_input":
            parsed = SimilarityMinToInputParamsModel.model_validate(self.params)
            self.params = parsed.model_dump(mode="json")
            return self

        raise ValueError(f"Unsupported check type '{self.check}'")


class BehaviourModel(BaseModel):
    """Behavioural policy block embedded in specs."""

    model_config = ConfigDict(extra="forbid")

    interrupt_policy: str
    abstain_policy: Optional[Dict[str, Any]] = None


class SpecModel(BaseModel):
    """Top-level strict v2 spec definition."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int = Field(ge=2)
    family: str = "default"
    spec_split: Literal["train", "dev", "test"] = "test"
    constraints: List[ConstraintModel]
    behaviour: BehaviourModel

    @model_validator(mode="after")
    def _validate_constraint_ids(self) -> "SpecModel":
        seen: set[str] = set()
        duplicates: list[str] = []
        for constraint in self.constraints:
            if constraint.id in seen:
                duplicates.append(constraint.id)
            else:
                seen.add(constraint.id)
        if duplicates:
            rendered = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Duplicate constraint ids are not allowed: {rendered}")
        return self


class TaskScoringModel(BaseModel):
    primary: str
    secondary: Optional[str] = None


class TaskInputModel(BaseModel):
    smiles: Optional[str] = None


class TaskEvidenceModel(BaseModel):
    """Feasibility or contradiction evidence attached to generated tasks."""

    model_config = ConfigDict(extra="forbid")

    feasible_witness_smiles: Optional[str] = None
    contradiction_proof: Optional[Dict[str, Any]] = None
    budget_infeasible_note: Optional[str] = None
    invariance_group_id: Optional[str] = None
    invariance_canonical_smiles: Optional[str] = None
    invariance_variant_label: Optional[str] = None
    boundary_property: Optional[str] = None
    boundary_side: Optional[Literal["lower", "upper"]] = None
    boundary_distance: Optional[float] = Field(default=None, ge=0.0)


class TaskBudgetsModel(BaseModel):
    """Execution budgets enforced by the runner."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1)
    max_proposals: int = Field(ge=0)
    max_verify_calls: int = Field(ge=0)
    max_total_verifier_calls: int = Field(ge=0)
    max_edit_cost: Optional[float] = Field(default=None, ge=0)


class TaskConstraintOverrideModel(BaseModel):
    """Deterministic per-constraint override block merged into a base spec."""

    model_config = ConfigDict(extra="forbid")

    check: Optional[SpecCheck] = None
    type: Optional[ConstraintType] = None
    severity: Optional[str] = None
    weight: Optional[float] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class TaskConstraintsModel(BaseModel):
    """Task-level patch for constructing an effective spec."""

    model_config = ConfigDict(extra="forbid")

    overrides: Dict[str, TaskConstraintOverrideModel] = Field(default_factory=dict)
    additions: List[ConstraintModel] = Field(default_factory=list)


def default_task_budgets(protocol: str) -> TaskBudgetsModel:
    """Protocol defaults preserve the original step limits."""

    if protocol == "L1":
        return TaskBudgetsModel(
            max_steps=1,
            max_proposals=1,
            max_verify_calls=0,
            max_total_verifier_calls=1,
        )
    if protocol == "L2":
        return TaskBudgetsModel(
            max_steps=3,
            max_proposals=3,
            max_verify_calls=0,
            max_total_verifier_calls=3,
        )
    return TaskBudgetsModel(
        max_steps=4,
        max_proposals=4,
        max_verify_calls=4,
        max_total_verifier_calls=8,
    )


class InterruptExpectedModel(BaseModel):
    """Expected interrupt-handling behaviour for scoring."""

    must_ack: bool = True
    must_restate_goal: bool = False
    must_report_state: bool = True
    allowed_actions: List[InterruptAction] = Field(
        default_factory=lambda: ["ABSTAIN"]
    )


class InterruptModel(BaseModel):
    """Interrupt configuration for a task."""

    enabled: bool = False
    after_step: Optional[int] = None
    signal_text: Optional[str] = None
    expected_behavior: InterruptExpectedModel = Field(
        default_factory=InterruptExpectedModel
    )


class ExpectedFieldsModel(BaseModel):
    """Optional output-format expectations for tasks."""

    must_return_smiles: bool = True


class TaskModel(BaseModel):
    """Schema for a single evaluation task."""

    task_id: str
    suite: str
    protocol: Literal["L1", "L2", "L3"]
    prompt: str
    input: TaskInputModel
    spec_id: str
    scoring: TaskScoringModel
    task_family: Optional[str] = None
    evidence: Optional[TaskEvidenceModel] = None
    task_constraints: Optional[TaskConstraintsModel] = None
    budgets: Optional[TaskBudgetsModel] = None
    interrupt_at_step: Optional[int] = None
    expected: ExpectedOutcome = "PASS"
    expected_action: Optional[ExpectedAction] = None
    interrupt: Optional[InterruptModel] = None
    expected_fields: Optional[ExpectedFieldsModel] = None

    @model_validator(mode="after")
    def _populate_expected_action(self) -> "TaskModel":
        if self.expected_action is None:
            self.expected_action = legacy_expected_to_action(self.expected)
        if self.budgets is None:
            self.budgets = default_task_budgets(self.protocol)
        return self


class FailureItem(BaseModel):
    """One legacy entry in a failure vector."""

    id: str
    detail: Optional[str] = None
    delta: Optional[float] = None
    distance_to_bound: Optional[float] = None


class AlertHitModel(BaseModel):
    """Matched alert descriptor."""

    model_config = ConfigDict(extra="forbid")

    id: str
    family: str


class PropertyDetailModel(BaseModel):
    """Constraint-aligned property margin detail."""

    model_config = ConfigDict(extra="forbid")

    property: str
    value: float
    bounds: BoundsModel
    signed_margin: float


class ConstraintResultModel(BaseModel):
    """Per-constraint status emitted in failure vectors."""

    model_config = ConfigDict(extra="forbid")

    constraint_id: str
    check: str
    status: Literal["pass", "fail"]
    detail: Optional[str] = None
    property_details: List[PropertyDetailModel] = Field(default_factory=list)
    hit_count: Optional[int] = None
    hits: List[AlertHitModel] = Field(default_factory=list)


class FailureVector(BaseModel):
    """Structured view of constraint outcomes handed back to an agent."""

    hard_fails: List[FailureItem] = Field(default_factory=list)
    soft_misses: List[FailureItem] = Field(default_factory=list)
    margins: List[FailureItem] = Field(default_factory=list)
    constraint_results: List[ConstraintResultModel] = Field(default_factory=list)
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


def _normalize_bounds_payload(raw_bounds: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(raw_bounds, dict):
        raise ValueError("Bounds payload must be an object")
    normalized: Dict[str, Dict[str, float]] = {}
    for name, raw in raw_bounds.items():
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            lower, upper = raw
        elif isinstance(raw, dict) and "min" in raw and "max" in raw:
            lower, upper = raw["min"], raw["max"]
        else:
            raise ValueError(
                f"Invalid bounds payload for '{name}'; expected [min,max] or object"
            )
        normalized[str(name)] = {"min": float(lower), "max": float(upper)}
    return normalized


def migrate_spec_v1_to_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy v1 spec data into strict v2 internal shape."""

    if not isinstance(payload, dict):
        raise ValueError("Spec payload must be an object")
    normalized = deepcopy(payload)
    raw_constraints = normalized.get("constraints")
    if not isinstance(raw_constraints, list):
        raise ValueError("Spec constraints must be a list")

    migrated_constraints: List[Dict[str, Any]] = []
    saw_legacy = False
    for raw_constraint in raw_constraints:
        if not isinstance(raw_constraint, dict):
            raise ValueError("Constraint payload must be an object")
        constraint = deepcopy(raw_constraint)
        check = constraint.get("check")
        params = constraint.get("params")
        if not isinstance(params, dict):
            params = {}

        if check == "property_bounds_all":
            saw_legacy = True
            constraint["check"] = "property_bounds"
            constraint["params"] = {
                "mode": "all",
                "bounds": _normalize_bounds_payload(params.get("bounds", {})),
            }
        elif check == "property_bounds_any":
            saw_legacy = True
            bounds_payload = params.get("bounds") if "bounds" in params else params
            constraint["check"] = "property_bounds"
            constraint["params"] = {
                "mode": "any",
                "bounds": _normalize_bounds_payload(bounds_payload),
            }
        elif check == "substructure_absent" and "alert_set" in params:
            saw_legacy = True
            migrated_params: Dict[str, Any] = {"alert_set": params["alert_set"]}
            if "min_hits" in params:
                migrated_params["min_hits"] = params["min_hits"]
            constraint["check"] = "alert_set_absent"
            constraint["params"] = migrated_params
        else:
            constraint["params"] = params
        migrated_constraints.append(constraint)

    normalized["constraints"] = migrated_constraints
    try:
        version = int(normalized.get("version", 1))
    except (TypeError, ValueError):
        version = 1
    if version < 2 or saw_legacy:
        normalized["version"] = 2
    return normalized


def load_spec(spec_id: str, *, paths: ProjectPaths = PATHS) -> SpecModel:
    """Load a spec by identifier from the data directory."""

    spec_path = paths.specs_dir / f"{spec_id}.yaml"
    if not spec_path.exists():
        raise SpecNotFoundError(f"Unknown spec '{spec_id}' at {spec_path}")
    with spec_path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle)
    try:
        data = migrate_spec_v1_to_v2(raw_data)
        return SpecModel.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - validation details vary
        raise ValueError(f"Invalid spec data: {exc}") from exc


def load_tasks_for_suite(
    suite_name: str, *, paths: ProjectPaths = PATHS
) -> List[TaskModel]:
    """Read all tasks for a given suite."""

    validate_unique_task_ids(paths=paths)
    suite_path = paths.suites_dir / f"{suite_name}.jsonl"
    if not suite_path.exists():
        raise TaskSuiteNotFoundError(
            f"Unknown task suite '{suite_name}' at {suite_path}"
        )
    tasks: List[TaskModel] = []
    for entry in jsonio.read_jsonl(suite_path):
        try:
            tasks.append(TaskModel.model_validate(entry))
        except ValidationError as exc:
            raise ValueError(f"Invalid task payload in {suite_path}: {exc}") from exc
    return tasks


def validate_unique_task_ids(*, paths: ProjectPaths = PATHS) -> None:
    """Fail fast if any task_id appears in more than one suite (or twice in one)."""

    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for suite_path in sorted(paths.suites_dir.glob("*.jsonl")):
        suite_name = suite_path.stem
        for line_number, entry in enumerate(jsonio.read_jsonl(suite_path), start=1):
            task_id = entry.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(
                    f"Invalid task_id in {suite_path}:{line_number}; expected non-empty string."
                )
            owner = f"{suite_name}:{line_number}"
            previous = seen.get(task_id)
            if previous is not None:
                duplicates.append(f"{task_id} ({previous}, {owner})")
            else:
                seen[task_id] = owner
    if duplicates:
        rendered = "; ".join(duplicates[:10])
        if len(duplicates) > 10:
            rendered += f"; ... and {len(duplicates) - 10} more"
        raise ValueError(f"Duplicate task_id values across suites: {rendered}")


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
