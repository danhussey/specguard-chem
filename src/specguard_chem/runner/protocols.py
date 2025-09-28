from __future__ import annotations

"""Constraint evaluation primitives and protocol helpers."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..config import ConstraintModel, FailureItem, FailureVector, SpecModel
from ..verifiers import (
    canonicalize_smiles,
    check_property_bounds_all,
    check_property_bounds_any,
    compute_properties,
    margins_to_bounds,
    pains_alerts,
    parse_smiles,
    synthetic_accessibility_score,
)


@dataclass
class ConstraintOutcome:
    constraint: ConstraintModel
    passed: bool
    detail: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def as_failure_item(self) -> FailureItem:
        return FailureItem(id=self.constraint.id, detail=self.detail)


@dataclass
class EvaluationResult:
    input_smiles: str
    canonical_smiles: Optional[str]
    valid: bool
    hard_outcomes: List[ConstraintOutcome]
    soft_outcomes: List[ConstraintOutcome]
    properties: Dict[str, float]
    property_margins: Dict[str, float]
    alerts: List[str]
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
            outcome.as_failure_item() for outcome in self.hard_outcomes if not outcome.passed
        ]
        soft_items: List[FailureItem] = []
        for outcome in self.soft_outcomes:
            if outcome.passed:
                continue
            delta = outcome.info.get("delta")
            soft_items.append(
                FailureItem(id=outcome.constraint.id, detail=outcome.detail, delta=delta)
            )
        margin_items = [
            FailureItem(id=name, distance_to_bound=margin)
            for name, margin in sorted(self.property_margins.items())
        ]
        return FailureVector(
            hard_fails=hard_items,
            soft_misses=soft_items,
            margins=margin_items,
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
        alerts: List[str] = []
        sa_score: Optional[float] = None
        hard_outcomes: List[ConstraintOutcome] = []
        soft_outcomes: List[ConstraintOutcome] = []

        if not valid:
            hard_outcomes.append(
                ConstraintOutcome(
                    constraint=ConstraintModel(
                        id="smiles_valid",
                        type="hard",
                        check="smiles_valid",
                        params={},
                    ),
                    passed=False,
                    detail="Invalid SMILES",
                )
            )
            return EvaluationResult(
                input_smiles=smiles,
                canonical_smiles=canonical,
                valid=False,
                hard_outcomes=hard_outcomes,
                soft_outcomes=soft_outcomes,
                properties=properties,
                property_margins=property_margins,
                alerts=alerts,
                sa_score=sa_score,
            )

        properties = compute_properties(mol)
        sa_score = synthetic_accessibility_score(mol)

        for constraint in self.spec.constraints:
            if constraint.check == "substructure_absent":
                set_name = constraint.params.get("alert_set", "PAINS_A")
                hits = pains_alerts(mol, set_name)
                alerts.extend(hits)
                passed = len(hits) == 0
                detail = None if passed else f"{hits[0]} alert"
                outcome = ConstraintOutcome(
                    constraint=constraint,
                    passed=passed,
                    detail=detail,
                    info={"alerts": hits},
                )
            elif constraint.check == "property_bounds_all":
                bounds = {
                    key: tuple(value) if not isinstance(value, tuple) else value
                    for key, value in constraint.params.get("bounds", {}).items()
                }
                passed = check_property_bounds_all(properties, bounds)
                margins = margins_to_bounds(properties, bounds)
                self._update_margins(property_margins, margins)
                detail = None
                if not passed:
                    violations = [
                        f"{name}={properties.get(name, float('nan')):.1f} outside [{lower:.1f},{upper:.1f}]"
                        for name, (lower, upper) in bounds.items()
                        if name in properties
                        and not (lower <= properties[name] <= upper)
                    ]
                    detail = "; ".join(violations) or "Property outside bounds"
                outcome = ConstraintOutcome(
                    constraint=constraint,
                    passed=passed,
                    detail=detail,
                    info={"margins": margins},
                )
            elif constraint.check == "property_bounds_any":
                bounds = {
                    key: tuple(value) if not isinstance(value, tuple) else value
                    for key, value in constraint.params.items()
                }
                passed = check_property_bounds_any(properties, bounds)
                margins = margins_to_bounds(properties, bounds)
                self._update_margins(property_margins, margins)
                detail = None if passed else "All properties outside target window"
                delta = max(margins.values()) if margins else None
                outcome = ConstraintOutcome(
                    constraint=constraint,
                    passed=passed,
                    detail=detail,
                    info={"margins": margins, "delta": delta},
                )
            else:  # pragma: no cover - future spec types
                raise NotImplementedError(f"Unsupported check '{constraint.check}'")

            if constraint.type == "hard":
                hard_outcomes.append(outcome)
            else:
                if "delta" not in outcome.info:
                    outcome.info["delta"] = max(
                        outcome.info.get("margins", {}).values(),
                        default=None,
                    )
                soft_outcomes.append(outcome)

        return EvaluationResult(
            input_smiles=smiles,
            canonical_smiles=canonical,
            valid=True,
            hard_outcomes=hard_outcomes,
            soft_outcomes=soft_outcomes,
            properties=properties,
            property_margins=property_margins,
            alerts=alerts,
            sa_score=sa_score,
        )

    @staticmethod
    def _update_margins(
        target: Dict[str, float], margins: Dict[str, float]
    ) -> None:
        for name, margin in margins.items():
            current = target.get(name)
            if current is None or abs(margin) < abs(current):
                target[name] = margin
