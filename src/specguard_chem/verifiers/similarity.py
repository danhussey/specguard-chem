from __future__ import annotations

"""Similarity helpers for edit-economy metrics."""

from typing import Optional

from rdkit import DataStructs
from rdkit.Chem import AllChem

from .smiles import parse_smiles


def morgan_tanimoto(
    smiles_a: str,
    smiles_b: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[float]:
    """Return Morgan fingerprint Tanimoto similarity for two SMILES."""

    mol_a = parse_smiles(smiles_a)
    mol_b = parse_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
