from __future__ import annotations

"""Constraint evaluation primitives and protocol helpers."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from ..config import (
    AlertHitModel,
    BoundsModel,
    ConstraintModel,
    ConstraintResultModel,
    FailureItem,
    FailureVector,
    PropertyDetailModel,
    SpecModel,
)
from ..verifiers import (
    alert_hits,
    canonicalize_smiles,
    check_property_bounds_all,
    check_property_bounds_any,
    compute_properties,
    margins_to_bounds,
    parse_smiles,
    synthetic_accessibility_score,
)

_SMARTS_LIBRARY: Dict[str, str] = {
    "phenol": "c1ccc(O)cc1",
    "aniline": "c1ccc(N)cc1",
    "carboxylic_acid": "C(=O)[O;H,-]",
    "amide": "NC(=O)",
}
_SMARTS_CACHE: Dict[str, Optional[Chem.Mol]] = {}


def _resolve_smarts(smarts_id: str) -> Optional[Chem.Mol]:
    if smarts_id not in _SMARTS_CACHE:
        smarts = _SMARTS_LIBRARY.get(smarts_id)
        _SMARTS_CACHE[smarts_id] = Chem.MolFromSmarts(smarts) if smarts else None
    return _SMARTS_CACHE[smarts_id]


def _count_matches(mol: Chem.Mol, pattern: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(pattern))


def _count_in_range(count: int, count_range: Optional[Dict[str, Any]]) -> bool:
    if not count_range:
        return count > 0
    lower = count_range.get("min")
    upper = count_range.get("max")
    if lower is not None and count < int(lower):
        return False
    if upper is not None and count > int(upper):
        return False
    return True


@dataclass
class ConstraintOutcome:
    constraint: ConstraintModel
    passed: bool
    detail: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def as_failure_item(self) -> FailureItem:
        return FailureItem(id=self.constraint.id, detail=self.detail)

    def as_constraint_result(self) -> ConstraintResultModel:
        raw_property_details = self.info.get("property_details") or []
        property_details = [
            PropertyDetailModel.model_validate(item) for item in raw_property_details
        ]
        raw_hits = self.info.get("hits") or []
        hits = [AlertHitModel.model_validate(hit) for hit in raw_hits]
        hit_count = self.info.get("hit_count")
        return ConstraintResultModel(
            constraint_id=self.constraint.id,
            check=self.constraint.check,
            status="pass" if self.passed else "fail",
            detail=self.detail,
            property_details=property_details,
            hit_count=int(hit_count) if hit_count is not None else None,
            hits=hits,
        )


@dataclass
class EvaluationResult:
    input_smiles: str
    canonical_smiles: Optional[str]
    valid: bool
    hard_outcomes: List[ConstraintOutcome]
    soft_outcomes: List[ConstraintOutcome]
    constraint_outcomes: List[ConstraintOutcome]
    properties: Dict[str, float]
    property_margins: Dict[str, float]
    alerts: List[Dict[str, str]]
    sa_score: Optional[float]

    @property
    def hard_pass(self) -> bool:
        return all(outcome.passed for outcome in self.hard_outcomes) and self.valid

    def soft_score_terms(self) -> List[Tuple[float, float]]:
        terms: List[Tuple[float, float]] = []
        for outcome in self.soft_outcomes:
            score = 1.0 if outcome.passed else 0.0
            weight = outcome.constraint.weight or 1.0
            terms.append((score, weight))
        return terms

    def build_failure_vector(self, round_id: int) -> FailureVector:
        hard_items = [
            outcome.as_failure_item()
            for outcome in self.hard_outcomes
            if not outcome.passed
        ]
        soft_items: List[FailureItem] = []
        for outcome in self.soft_outcomes:
            if outcome.passed:
                continue
            delta = outcome.info.get("delta")
            soft_items.append(
                FailureItem(
                    id=outcome.constraint.id, detail=outcome.detail, delta=delta
                )
            )
        margin_items = [
            FailureItem(id=name, distance_to_bound=margin)
            for name, margin in sorted(self.property_margins.items())
        ]
        constraint_results = [
            outcome.as_constraint_result() for outcome in self.constraint_outcomes
        ]
        return FailureVector(
            hard_fails=hard_items,
            soft_misses=soft_items,
            margins=margin_items,
            constraint_results=constraint_results,
            round=round_id,
        )


class ConstraintEvaluator:
    """Evaluate specs against SMILES strings."""

    def __init__(self, spec: SpecModel):
        self.spec = spec

    def evaluate(self, smiles: str) -> EvaluationResult:
        canonical = canonicalize_smiles(smiles) if smiles else None
        mol = parse_smiles(smiles) if smiles else None
        valid = mol is not None
        properties: Dict[str, float] = {}
        property_margins: Dict[str, float] = {}
        alerts_seen: set[tuple[str, str]] = set()
        sa_score: Optional[float] = None
        hard_outcomes: List[ConstraintOutcome] = []
        soft_outcomes: List[ConstraintOutcome] = []
        constraint_outcomes: List[ConstraintOutcome] = []

        if not valid:
            for constraint in self.spec.constraints:
                outcome = ConstraintOutcome(
                    constraint=constraint,
                    passed=False,
                    detail="Invalid SMILES",
                )
                constraint_outcomes.append(outcome)
                if constraint.type == "hard":
                    hard_outcomes.append(outcome)
                else:
                    soft_outcomes.append(outcome)
            return EvaluationResult(
                input_smiles=smiles,
                canonical_smiles=canonical,
                valid=False,
                hard_outcomes=hard_outcomes,
                soft_outcomes=soft_outcomes,
                constraint_outcomes=constraint_outcomes,
                properties=properties,
                property_margins=property_margins,
                alerts=[],
                sa_score=sa_score,
            )

        properties = compute_properties(mol)
        sa_score = synthetic_accessibility_score(mol)

        for constraint in self.spec.constraints:
            if constraint.check in {"alert_set_absent", "alert_set_present"}:
                set_name = str(constraint.params.get("alert_set", "PAINS_A"))
                hits = alert_hits(mol, set_name)
                for hit in hits:
                    alerts_seen.add((hit["id"], hit["family"]))
                hit_count = len(hits)
                min_hits = int(constraint.params.get("min_hits", 1))
                if constraint.check == "alert_set_absent":
                    passed = hit_count == 0
                    detail = None if passed else f"{hits[0]['id']} alert"
                else:
                    passed = hit_count >= min_hits
                    detail = None if passed else f"Expected at least {min_hits} alert hit(s)"
                outcome = ConstraintOutcome(
                    constraint=constraint,
                    passed=passed,
                    detail=detail,
                    info={"hits": hits, "hit_count": hit_count},
                )
            elif constraint.check == "property_bounds":
                mode = str(constraint.params.get("mode", "all"))
                raw_bounds = constraint.params.get("bounds", {})
                bounds = {
                    name: (float(entry["min"]), float(entry["max"]))
                    for name, entry in raw_bounds.items()
                }
                if mode == "any":
                    passed = check_property_bounds_any(properties, bounds)
                else:
                    passed = check_property_bounds_all(properties, bounds)
                margins = margins_to_bounds(properties, bounds)
                self._update_margins(property_margins, margins)
                property_details = []
                violations = []
                for name, (lower, upper) in sorted(bounds.items()):
                    value = float(properties.get(name, float("nan")))
                    margin = float(margins.get(name, 0.0))
                    property_details.append(
                        PropertyDetailModel(
                            property=name,
                            value=value,
                            bounds=BoundsModel(min=lower, max=upper),
                            signed_margin=margin,
                        ).model_dump(mode="json")
                    )
                    if value < lower or value > upper:
                        violations.append(
                            f"{name}={value:.3f} outside [{lower:.3f},{upper:.3f}]"
                        )
                detail = None
                if not passed:
                    if mode == "any":
                        detail = "No properties satisfy any target window"
                    else:
                        detail = "; ".join(violations) or "Property outside bounds"
                info: Dict[str, Any] = {
                    "margins": margins,
                    "property_details": property_details,
                }
                if mode == "any":
                    info["delta"] = max(margins.values()) if margins else None
                outcome = ConstraintOutcome(
                    constraint=constraint,
                    passed=passed,
                    detail=detail,
                    info=info,
                )
            elif constraint.check in {"substructure_present", "substructure_absent"}:
                smarts_id = str(constraint.params.get("smarts_id", ""))
                pattern = _resolve_smarts(smarts_id)
                if pattern is None:
                    raise ValueError(f"Unknown SMARTS id '{smarts_id}'")
                match_count = _count_matches(mol, pattern)
                if constraint.check == "substructure_present":
                    passed = _count_in_range(match_count, constraint.params.get("count"))
                    detail = None if passed else f"Expected presence of '{smarts_id}'"
                else:
                    if constraint.params.get("count"):
                        passed = _count_in_range(
                            match_count, constraint.params.get("count")
                        )
                    else:
                        passed = match_count == 0
                    detail = None if passed else f"Unexpected '{smarts_id}' matches"
                outcome = ConstraintOutcome(
                    constraint=constraint,
                    passed=passed,
                    detail=detail,
                    info={"match_count": match_count},
                )
            elif constraint.check == "sa_proxy_max":
                max_value = float(constraint.params.get("max", 10.0))
                passed = sa_score is not None and sa_score <= max_value
                detail = None if passed else f"SA={sa_score:.3f} exceeds {max_value:.3f}"
                outcome = ConstraintOutcome(
                    constraint=constraint,
                    passed=passed,
                    detail=detail,
                    info={"value": sa_score, "max": max_value},
                )
            else:  # pragma: no cover - guarded by strict schema
                raise NotImplementedError(f"Unsupported check '{constraint.check}'")

            constraint_outcomes.append(outcome)
            if constraint.type == "hard":
                hard_outcomes.append(outcome)
            else:
                if "delta" not in outcome.info:
                    outcome.info["delta"] = max(
                        outcome.info.get("margins", {}).values(),
                        default=None,
                    )
                soft_outcomes.append(outcome)

        alerts = [
            {"id": alert_id, "family": family}
            for alert_id, family in sorted(alerts_seen)
        ]
        return EvaluationResult(
            input_smiles=smiles,
            canonical_smiles=canonical,
            valid=True,
            hard_outcomes=hard_outcomes,
            soft_outcomes=soft_outcomes,
            constraint_outcomes=constraint_outcomes,
            properties=properties,
            property_margins=property_margins,
            alerts=alerts,
            sa_score=sa_score,
        )

    @staticmethod
    def _update_margins(target: Dict[str, float], margins: Dict[str, float]) -> None:
        for name, margin in margins.items():
            current = target.get(name)
            if current is None or abs(margin) < abs(current):
                target[name] = margin
