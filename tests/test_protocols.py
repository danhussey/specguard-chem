from __future__ import annotations

import pytest

from specguard_chem.config import SpecModel
from specguard_chem.runner.protocols import ConstraintEvaluator


def test_failure_vector_property_margin_is_constraint_aligned() -> None:
    spec = SpecModel.model_validate(
        {
            "id": "spec_bounds",
            "version": 2,
            "constraints": [
                {
                    "id": "logp_cap",
                    "type": "hard",
                    "check": "property_bounds",
                    "params": {
                        "mode": "all",
                        "bounds": {"logP": {"min": 0.0, "max": 1.0}},
                    },
                }
            ],
            "behaviour": {"interrupt_policy": "confirm_then_continue"},
        }
    )
    result = ConstraintEvaluator(spec).evaluate("CCCCCCCC")
    fv = result.build_failure_vector(round_id=1)

    constraint_result = fv.constraint_results[0]
    assert constraint_result.constraint_id == "logp_cap"
    assert constraint_result.check == "property_bounds"
    assert constraint_result.status == "fail"
    assert len(constraint_result.property_details) == 1

    detail = constraint_result.property_details[0]
    assert detail.property == "logP"
    assert detail.signed_margin < 0
    assert detail.signed_margin == pytest.approx(detail.bounds.max - detail.value)


def test_failure_vector_alert_constraint_includes_hit_details() -> None:
    spec = SpecModel.model_validate(
        {
            "id": "spec_alerts",
            "version": 2,
            "constraints": [
                {
                    "id": "alerts_block",
                    "type": "hard",
                    "check": "alert_set_absent",
                    "params": {"alert_set": "PAINS_A"},
                }
            ],
            "behaviour": {"interrupt_policy": "confirm_then_continue"},
        }
    )
    result = ConstraintEvaluator(spec).evaluate("O=C1NC(=S)SC1=Cc1ccccc1")
    fv = result.build_failure_vector(round_id=1)

    constraint_result = fv.constraint_results[0]
    assert constraint_result.constraint_id == "alerts_block"
    assert constraint_result.status == "fail"
    assert constraint_result.hit_count is not None
    assert constraint_result.hit_count >= 1
    assert len(constraint_result.hits) == constraint_result.hit_count
