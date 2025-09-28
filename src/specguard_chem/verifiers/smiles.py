from __future__ import annotations

"""SMILES helpers."""

from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdchem


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES; returns None if invalid."""

    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except (ValueError, rdchem.KekulizeException):
        return None
    return mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except ValueError:
        return None


def is_valid_smiles(smiles: str) -> bool:
    return parse_smiles(smiles) is not None
