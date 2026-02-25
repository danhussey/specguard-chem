from __future__ import annotations

"""Substructure alert helpers (RDKit catalog-backed with deterministic fallback)."""

from collections import defaultdict
from typing import Dict, Iterable, List
import warnings

from rdkit import Chem

try:  # pragma: no cover - environment dependent
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "to-Python converter for boost::shared_ptr<RDKit::FilterHierarchyMatcher> "
                "already registered; second conversion method ignored."
            ),
            category=RuntimeWarning,
        )
        from rdkit.Chem import FilterCatalog
except Exception:  # pragma: no cover - environment dependent
    FilterCatalog = None  # type: ignore[assignment]

_FALLBACK_ALERT_SMARTS: Dict[str, Dict[str, str]] = {
    "PAINS_A": {
        "quinone_A": "O=C1C=CC(=O)C=C1",
        "ene_rhod_A": "O=C1NC(=S)SC1=C",
    },
    "PAINS_B": {
        "catechol_B": "Oc1ccc(O)cc1",
        "hydroquinone_B": "c1cc(O)cc(O)c1",
    },
    "PAINS_C": {
        "anilide_C": "NC(=O)c1ccccc1",
    },
    "BRENK": {
        "nitro_like": "[N+](=O)[O-]",
    },
}

_FALLBACK_CACHE: Dict[str, Dict[str, Chem.Mol]] = {}
_CATALOG_CACHE: Dict[str, "FilterCatalog.FilterCatalog"] = {}


def _catalog_map() -> Dict[str, object]:
    if FilterCatalog is None:
        return {}
    catalogs = FilterCatalog.FilterCatalogParams.FilterCatalogs
    return {
        "PAINS": catalogs.PAINS,
        "PAINS_A": catalogs.PAINS_A,
        "PAINS_B": catalogs.PAINS_B,
        "PAINS_C": catalogs.PAINS_C,
        "BRENK": catalogs.BRENK,
    }


def _load_fallback_alert_set(set_name: str) -> Dict[str, Chem.Mol]:
    if set_name not in _FALLBACK_ALERT_SMARTS:
        raise KeyError(f"Unknown alert set '{set_name}'")
    if set_name not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[set_name] = {
            alert_id: Chem.MolFromSmarts(smarts)
            for alert_id, smarts in _FALLBACK_ALERT_SMARTS[set_name].items()
        }
    return _FALLBACK_CACHE[set_name]


def _load_catalog(set_name: str):
    catalog_map = _catalog_map()
    if set_name not in catalog_map:
        raise KeyError(f"Unknown alert set '{set_name}'")
    if set_name not in _CATALOG_CACHE:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(catalog_map[set_name])
        _CATALOG_CACHE[set_name] = FilterCatalog.FilterCatalog(params)
    return _CATALOG_CACHE[set_name]


def alert_hits(mol: Chem.Mol, set_name: str = "PAINS_A") -> List[Dict[str, str]]:
    """Return deterministic alert hits as dictionaries with id/family."""

    hits: set[tuple[str, str]] = set()
    if FilterCatalog is not None:
        try:
            catalog = _load_catalog(set_name)
            for entry in catalog.GetMatches(mol):
                description = entry.GetDescription().strip()
                props = {key: entry.GetProp(key) for key in entry.GetPropList()}
                family = props.get("FilterSet", set_name)
                if description:
                    hits.add((description, family))
        except KeyError:
            pass

    if hits:
        return [{"id": alert_id, "family": family} for alert_id, family in sorted(hits)]

    if set_name not in _FALLBACK_ALERT_SMARTS:
        return []
    patterns = _load_fallback_alert_set(set_name)
    fallback_hits: List[Dict[str, str]] = []
    for alert_id, pattern in sorted(patterns.items()):
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            fallback_hits.append({"id": alert_id, "family": set_name})
    return fallback_hits


def alert_counts_by_family(alerts: Iterable[Dict[str, str]]) -> Dict[str, int]:
    """Count alert hits by family."""

    counts: Dict[str, int] = defaultdict(int)
    for hit in alerts:
        family = str(hit.get("family", "UNKNOWN"))
        counts[family] += 1
    return dict(sorted(counts.items()))


def pains_alerts(mol: Chem.Mol, set_name: str = "PAINS_A") -> List[str]:
    """Backward-compatible list of alert IDs for matched motifs."""

    return [hit["id"] for hit in alert_hits(mol, set_name)]


def substructure_absent(mol: Chem.Mol, alert_set: str) -> bool:
    """True iff no alerts from *alert_set* are present in *mol*."""

    return len(pains_alerts(mol, alert_set)) == 0


def available_alert_sets() -> Iterable[str]:
    """Return the names of registered alert sets."""

    names = set(_FALLBACK_ALERT_SMARTS)
    names.update(_catalog_map().keys())
    return sorted(names)
