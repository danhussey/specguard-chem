from __future__ import annotations

import pytest
from rdkit import Chem

from specguard_chem.verifiers import (
    BOUNDS_TOLERANCE,
    alert_hits,
    available_alert_sets,
    brics_fragment_edit_distance,
    check_property_bounds_all,
    check_property_bounds_any,
    compute_properties,
    margins_to_bounds,
    morgan_tanimoto,
    pains_alerts,
    synthetic_accessibility_score,
)


def test_property_bounds_all_pass() -> None:
    mol = Chem.MolFromSmiles("CC(=O)NC1=CC=CC=C1O")
    props = compute_properties(mol)
    bounds = {
        "MW": (100, 500),
        "logP": (0.0, 5.0),
        "HBD": (0, 5),
        "HBA": (0, 10),
    }
    assert check_property_bounds_all(props, bounds)
    margins = margins_to_bounds(props, bounds)
    assert all(margin >= 0 for margin in margins.values())


def test_property_bounds_any_window() -> None:
    mol = Chem.MolFromSmiles("CC(=O)NC1=CC=CC=C1O")
    props = compute_properties(mol)
    bounds = {"TPSA": (20, 120)}
    assert check_property_bounds_any(props, bounds)


def test_boundary_inclusive_tolerance_policy() -> None:
    bounds = {"logP": (0.0, 1.0)}
    assert check_property_bounds_all({"logP": -0.5 * BOUNDS_TOLERANCE}, bounds)
    assert check_property_bounds_all({"logP": 1.0 + 0.5 * BOUNDS_TOLERANCE}, bounds)
    assert not check_property_bounds_all({"logP": -0.01}, bounds)


def test_pains_block_detects_catechol() -> None:
    mol = Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc1ccccc1")
    alerts = pains_alerts(mol, "PAINS_A")
    assert alerts


def test_alert_hits_include_id_and_family_and_are_deterministic() -> None:
    mol = Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc1ccccc1")
    first = alert_hits(mol, "PAINS_A")
    second = alert_hits(mol, "PAINS_A")
    assert first == second
    assert all("id" in hit and "family" in hit for hit in first)


def test_available_alert_sets_include_expanded_families() -> None:
    sets = set(available_alert_sets())
    assert {"PAINS_A", "PAINS_B", "PAINS_C", "BRENK"}.issubset(sets)


def test_sa_score_is_bounded() -> None:
    mol = Chem.MolFromSmiles("CC(=O)NC1=CC=CC=C1O")
    score = synthetic_accessibility_score(mol)
    assert 1.0 <= score <= 10.0


def test_morgan_tanimoto_identity() -> None:
    similarity = morgan_tanimoto("CCO", "CCO")
    assert similarity == pytest.approx(1.0)


def test_brics_fragment_edit_distance_identity() -> None:
    assert brics_fragment_edit_distance("CCOC(=O)N", "CCOC(=O)N") == 0


def test_brics_fragment_edit_distance_detects_change() -> None:
    distance = brics_fragment_edit_distance("CCOC(=O)N", "CCOC(=O)NC")
    assert distance is not None
    assert distance > 0
