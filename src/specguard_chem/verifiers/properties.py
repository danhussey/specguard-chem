from __future__ import annotations

"""Property calculators leveraging RDKit descriptors."""

from typing import Dict, Mapping

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

PROPERTY_NAMES = ("MW", "logP", "TPSA", "HBD", "HBA", "ROTB")


def compute_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Return baseline physicochemical descriptors for *mol*."""

    if mol is None:
        raise ValueError("Molecule is None; cannot compute properties")
    props = {
        "MW": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "ROTB": Lipinski.NumRotatableBonds(mol),
    }
    return props


def check_property_bounds_all(
    props: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]
) -> bool:
    """True iff all properties fall within inclusive bounds."""

    for name, (lower, upper) in bounds.items():
        value = props.get(name)
        if value is None or value < lower or value > upper:
            return False
    return True


def check_property_bounds_any(
    props: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]
) -> bool:
    """True if any property is inside the inclusive bounds block."""

    for name, (lower, upper) in bounds.items():
        value = props.get(name)
        if value is not None and lower <= value <= upper:
            return True
    return False


def margins_to_bounds(
    props: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]
) -> Dict[str, float]:
    """Distance to the nearest bound per property.

    Positive values indicate how far inside the admissible window the property
    lies; negative values indicate the magnitude of the violation.
    """

    margins: Dict[str, float] = {}
    for name, (lower, upper) in bounds.items():
        value = props.get(name)
        if value is None:
            continue
        if value < lower:
            margins[name] = value - lower
        elif value > upper:
            margins[name] = upper - value
        else:
            margins[name] = min(value - lower, upper - value)
    return margins
