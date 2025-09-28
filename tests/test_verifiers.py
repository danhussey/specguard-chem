from __future__ import annotations

from rdkit import Chem

from specguard_chem.verifiers import (
    check_property_bounds_all,
    check_property_bounds_any,
    compute_properties,
    margins_to_bounds,
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


def test_pains_block_detects_catechol() -> None:
    mol = Chem.MolFromSmiles("Oc1ccc(O)cc1")  # catechol motif
    alerts = pains_alerts(mol, "PAINS_A")
    assert "catechol_A" in alerts


def test_sa_score_is_bounded() -> None:
    mol = Chem.MolFromSmiles("CC(=O)NC1=CC=CC=C1O")
    score = synthetic_accessibility_score(mol)
    assert 1.0 <= score <= 10.0
