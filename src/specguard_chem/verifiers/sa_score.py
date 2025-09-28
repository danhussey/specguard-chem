from __future__ import annotations

"""Lightweight synthetic accessibility proxy."""

from math import tanh

from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors


def synthetic_accessibility_score(mol: Chem.Mol) -> float:
    """Return a heuristic SA score on the 1-10 scale (lower is easier)."""

    heavy = mol.GetNumHeavyAtoms()
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    fused = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    score = 1.0
    score += 0.1 * heavy
    score += 0.3 * rot_bonds
    score += 0.4 * spiro
    score += 0.4 * fused
    score += 0.2 * max(0, rings - 1)
    normalized = 5.0 * (tanh((score - 5) / 5) + 1)
    return max(1.0, min(10.0, normalized))
