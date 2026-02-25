from __future__ import annotations

"""Similarity helpers for edit-economy metrics."""

from typing import Optional

from rdkit import DataStructs, RDLogger
from rdkit.Chem import AllChem, BRICS, MolToSmiles

from .smiles import parse_smiles

# RDKit emits a deprecation warning on each legacy Morgan fingerprint call.
RDLogger.DisableLog("rdApp.warning")


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


def brics_fragment_edit_distance(
    smiles_a: str,
    smiles_b: str,
) -> Optional[int]:
    """Symmetric fragment-change count from BRICS decomposition."""

    mol_a = parse_smiles(smiles_a)
    mol_b = parse_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None

    fragments_a = BRICS.BRICSDecompose(mol_a)
    fragments_b = BRICS.BRICSDecompose(mol_b)
    if not fragments_a:
        fragments_a = {MolToSmiles(mol_a, canonical=True)}
    if not fragments_b:
        fragments_b = {MolToSmiles(mol_b, canonical=True)}

    return len(fragments_a.difference(fragments_b)) + len(
        fragments_b.difference(fragments_a)
    )
