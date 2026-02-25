from __future__ import annotations

"""Identity/equivalence helpers for invariance policies."""

from typing import Any, Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .smiles import parse_smiles


def _prepare_mol(
    smiles: str,
    *,
    require_stereo: bool,
    tautomer_invariant: bool,
    charge_invariant: bool,
    normalize: str,
) -> Optional[Chem.Mol]:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    work = Chem.Mol(mol)

    if normalize in {"rdkit_cleanup", "rdkit_cleanup_plus_tautomer_canon"}:
        try:
            work = rdMolStandardize.Cleanup(work)
        except Exception:
            return None

    if tautomer_invariant or normalize == "rdkit_cleanup_plus_tautomer_canon":
        try:
            enumerator = rdMolStandardize.TautomerEnumerator()
            work = enumerator.Canonicalize(work)
        except Exception:
            return None

    if charge_invariant:
        try:
            work = rdMolStandardize.Uncharger().uncharge(work)
        except Exception:
            return None

    if not require_stereo:
        Chem.RemoveStereochemistry(work)

    return work


def equivalence_key(
    smiles: str,
    *,
    require_stereo: bool,
    tautomer_invariant: bool,
    charge_invariant: bool,
    normalize: str,
    key: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    work = _prepare_mol(
        smiles,
        require_stereo=require_stereo,
        tautomer_invariant=tautomer_invariant,
        charge_invariant=charge_invariant,
        normalize=normalize,
    )
    if work is None:
        return None, {
            "valid": False,
            "normalize": normalize,
            "key": key,
        }

    normalized_smiles = Chem.MolToSmiles(
        work,
        canonical=True,
        isomericSmiles=require_stereo,
    )
    if key == "inchi_key":
        try:
            key_value = Chem.MolToInchiKey(work)
        except Exception:
            return None, {
                "valid": False,
                "normalize": normalize,
                "key": key,
                "normalized_smiles": normalized_smiles,
            }
    elif key == "canonical_smiles_after_normalization":
        key_value = normalized_smiles
    else:
        raise ValueError(f"Unsupported equivalence key '{key}'")

    return key_value, {
        "valid": True,
        "normalize": normalize,
        "key": key,
        "normalized_smiles": normalized_smiles,
        "formal_charge": int(Chem.GetFormalCharge(work)),
    }


def equivalent_smiles(
    reference_smiles: str,
    candidate_smiles: str,
    *,
    require_stereo: bool,
    tautomer_invariant: bool,
    charge_invariant: bool,
    normalize: str,
    key: str,
) -> Tuple[bool, Dict[str, Any]]:
    reference_key, reference_meta = equivalence_key(
        reference_smiles,
        require_stereo=require_stereo,
        tautomer_invariant=tautomer_invariant,
        charge_invariant=charge_invariant,
        normalize=normalize,
        key=key,
    )
    candidate_key, candidate_meta = equivalence_key(
        candidate_smiles,
        require_stereo=require_stereo,
        tautomer_invariant=tautomer_invariant,
        charge_invariant=charge_invariant,
        normalize=normalize,
        key=key,
    )

    passed = (
        reference_key is not None
        and candidate_key is not None
        and reference_key == candidate_key
    )
    return passed, {
        "reference": {
            "smiles": reference_smiles,
            "key": reference_key,
            **reference_meta,
        },
        "candidate": {
            "smiles": candidate_smiles,
            "key": candidate_key,
            **candidate_meta,
        },
    }
