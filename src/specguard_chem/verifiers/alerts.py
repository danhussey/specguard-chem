from __future__ import annotations

"""Substructure alert helpers (minimal PAINS-like set)."""

from typing import Dict, Iterable, List

from rdkit import Chem

# Minimal illustrative subset of PAINS-style SMARTS motifs. The goal is to flag
# molecules with prototypical problematic scaffolds rather than enumerate the
# full reference list (which is large and often redistributed under licensing
# constraints).
_PAINS_SMARTS: Dict[str, Dict[str, str]] = {
    "PAINS_A": {
        "catechol_A": "c1cc(O)ccc1O",
        "hydroquinone_A": "c1cc(O)cc(O)c1",
        "quinone_methide": "O=C1C=CC(=O)C=C1",
    }
}

_ALERT_CACHE: Dict[str, Dict[str, Chem.Mol]] = {}


def _load_alert_set(set_name: str) -> Dict[str, Chem.Mol]:
    if set_name not in _PAINS_SMARTS:
        raise KeyError(f"Unknown alert set '{set_name}'")
    if set_name not in _ALERT_CACHE:
        _ALERT_CACHE[set_name] = {
            alert_id: Chem.MolFromSmarts(smarts)
            for alert_id, smarts in _PAINS_SMARTS[set_name].items()
        }
    return _ALERT_CACHE[set_name]


def pains_alerts(mol: Chem.Mol, set_name: str = "PAINS_A") -> List[str]:
    """Return the identifiers of matched alert motifs."""

    patterns = _load_alert_set(set_name)
    matches: List[str] = []
    for alert_id, pattern in patterns.items():
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            matches.append(alert_id)
    return matches


def substructure_absent(mol: Chem.Mol, alert_set: str) -> bool:
    """True iff no alerts from *alert_set* are present in *mol*."""

    return len(pains_alerts(mol, alert_set)) == 0


def available_alert_sets() -> Iterable[str]:
    """Return the names of registered alert sets."""

    return _PAINS_SMARTS.keys()
