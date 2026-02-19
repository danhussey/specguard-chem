from __future__ import annotations

"""Deterministic offline corpus construction utilities."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..utils import jsonio
from ..verifiers import (
    alert_counts_by_family,
    alert_hits,
    canonicalize_smiles,
    compute_properties,
    synthetic_accessibility_score,
)

SEED_SCAFFOLDS: Tuple[str, ...] = (
    "c1ccccc1",
    "c1ccncc1",
    "c1ncccc1",
    "c1ccc(cc1)O",
    "c1ccc(cc1)N",
    "c1ccc(cc1)Cl",
    "c1ccc(cc1)F",
    "c1ccc(cc1)C(=O)N",
    "c1ccc(cc1)C(=O)O",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "COc1ccccc1",
    "CCOc1ccccc1",
    "CCN(CC)CC",
    "CCOC(=O)N",
    "CCOC(=O)C",
    "CC(=O)Nc1ccccc1",
    "CC(=O)Oc1ccccc1",
    "CC(=O)NCCO",
    "CCOC(=O)NCCO",
    "CC(=O)NCCN",
    "CCNCCO",
    "CCNCCN",
    "CCO",
    "CCN",
    "CCCO",
    "CCCN",
    "CC(C)O",
    "CC(C)N",
    "CCOC",
    "CCNC",
    "CC(=O)O",
    "CC(=O)N",
    "CCC(=O)O",
    "CCC(=O)N",
    "CCCCO",
    "CCCCN",
    "CCCOC",
    "CCN(C)C",
    "CCOC(=O)NC",
    "CC(=O)NC",
    "CCS",
    "CCCl",
    "CCF",
    "O=C(O)c1ccccc1",
    "NC(=O)c1ccccc1",
    "O=C(N)c1ccncc1",
    "COc1ncccc1",
    "CCOc1ncccc1",
    "O=C1NC(=S)SC1=Cc1ccccc1",
    "O=C1C=CC(=O)C=C1",
    "NCC(=O)O",
    "CCN(CC)CCO",
    "CCN(CC)CCN",
    "CCOC(=O)N(CC)CCO",
    "CCOC(=O)N(CC)CCN",
)

REACTION_SMARTS: Tuple[str, ...] = (
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]Cl",
    "[cH:1]>>[c:1]N",
    "[cH:1]>>[c:1]C(=O)N",
    "[CH3:1]>>[CH2:1]F",
    "[CH3:1]>>[CH2:1]Cl",
    "[CH3:1]>>[CH2:1]N",
    "[CH2:1]>>[CH:1](C)",
)

ALERT_SETS: Tuple[str, ...] = ("PAINS_A", "PAINS_B", "PAINS_C", "BRENK")


def _reaction_objects() -> List[rdChemReactions.ChemicalReaction]:
    return [rdChemReactions.ReactionFromSmarts(smarts) for smarts in REACTION_SMARTS]


def _sanitize_to_canonical(mol: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _apply_reactions(smiles: str, reactions: Sequence[rdChemReactions.ChemicalReaction]) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    products: set[str] = set()
    for reaction in reactions:
        try:
            product_sets = reaction.RunReactants((mol,))
        except Exception:
            continue
        for product_tuple in product_sets:
            if not product_tuple:
                continue
            candidate = _sanitize_to_canonical(product_tuple[0])
            if candidate:
                products.add(candidate)
    return sorted(products)


def _fallback_linear_library() -> List[str]:
    generated: List[str] = []
    for length in range(1, 13):
        generated.extend(
            [
                "C" * length,
                "C" * length + "O",
                "C" * length + "N",
                "C" * length + "Cl",
                "C" * length + "F",
            ]
        )
    return generated


def _scaffold_hash(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:16]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smiles
    return hashlib.sha256(scaffold.encode("utf-8")).hexdigest()[:16]


def _record_from_smiles(smiles: str) -> Dict[str, Any] | None:
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        return None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    properties = compute_properties(mol)
    sa_score = synthetic_accessibility_score(mol)
    all_hits: Dict[Tuple[str, str], Dict[str, str]] = {}
    for alert_set in ALERT_SETS:
        for hit in alert_hits(mol, alert_set):
            key = (hit["id"], hit["family"])
            all_hits[key] = hit
    alerts = [all_hits[key] for key in sorted(all_hits)]
    counts = alert_counts_by_family(alerts)
    return {
        "canonical_smiles": canonical,
        "properties": properties,
        "sa_score": sa_score,
        "alerts": alerts,
        "alert_counts_by_family": counts,
        "scaffold_hash": _scaffold_hash(canonical),
    }


def build_corpus_records(
    *,
    seed: int = 7,
    max_molecules: int = 1500,
    reaction_depth: int = 2,
) -> List[Dict[str, Any]]:
    """Build a deterministic molecule corpus from seeds and bounded reactions."""

    rng = random.Random(seed)
    reactions = _reaction_objects()
    seen: set[str] = set()
    corpus_smiles: List[str] = []
    frontier = sorted(filter(None, (canonicalize_smiles(s) for s in SEED_SCAFFOLDS)))

    for depth in range(reaction_depth + 1):
        next_frontier: set[str] = set()
        for smiles in frontier:
            if smiles in seen:
                continue
            seen.add(smiles)
            corpus_smiles.append(smiles)
            if len(corpus_smiles) >= max_molecules:
                break
            if depth < reaction_depth:
                for candidate in _apply_reactions(smiles, reactions):
                    if candidate not in seen:
                        next_frontier.add(candidate)
        if len(corpus_smiles) >= max_molecules:
            break
        frontier = sorted(next_frontier)
        if not frontier:
            break

    if len(corpus_smiles) < max_molecules:
        fallback = []
        for raw in _fallback_linear_library():
            canonical = canonicalize_smiles(raw)
            if canonical and canonical not in seen:
                fallback.append(canonical)
        rng.shuffle(fallback)
        for smiles in fallback:
            seen.add(smiles)
            corpus_smiles.append(smiles)
            if len(corpus_smiles) >= max_molecules:
                break

    records: List[Dict[str, Any]] = []
    for smiles in corpus_smiles:
        record = _record_from_smiles(smiles)
        if record is not None:
            records.append(record)
    records.sort(key=lambda item: item["canonical_smiles"])
    return records


def _flatten_for_tabular(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        properties = record.get("properties") or {}
        row = {
            "canonical_smiles": record.get("canonical_smiles"),
            "MW": properties.get("MW"),
            "logP": properties.get("logP"),
            "TPSA": properties.get("TPSA"),
            "HBD": properties.get("HBD"),
            "HBA": properties.get("HBA"),
            "ROTB": properties.get("ROTB"),
            "sa_score": record.get("sa_score"),
            "alerts_json": json.dumps(record.get("alerts", []), sort_keys=True),
            "alert_counts_json": json.dumps(
                record.get("alert_counts_by_family", {}), sort_keys=True
            ),
            "scaffold_hash": record.get("scaffold_hash"),
        }
        rows.append(row)
    return rows


def _expand_from_tabular(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        properties = {
            "MW": float(row.get("MW", 0.0)),
            "logP": float(row.get("logP", 0.0)),
            "TPSA": float(row.get("TPSA", 0.0)),
            "HBD": float(row.get("HBD", 0.0)),
            "HBA": float(row.get("HBA", 0.0)),
            "ROTB": float(row.get("ROTB", 0.0)),
        }
        records.append(
            {
                "canonical_smiles": str(row.get("canonical_smiles")),
                "properties": properties,
                "sa_score": float(row.get("sa_score", 0.0)),
                "alerts": json.loads(str(row.get("alerts_json", "[]"))),
                "alert_counts_by_family": json.loads(
                    str(row.get("alert_counts_json", "{}"))
                ),
                "scaffold_hash": str(row.get("scaffold_hash", "")),
            }
        )
    records.sort(key=lambda item: item["canonical_smiles"])
    return records


def write_corpus_records(path: Path, records: List[Dict[str, Any]]) -> Path:
    """Write corpus records to parquet if possible, else jsonl fallback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        jsonio.write_jsonl(path, records)
        return path

    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception:
            fallback = path.with_suffix(".jsonl")
            jsonio.write_jsonl(fallback, records)
            return fallback
        rows = _flatten_for_tabular(records)
        frame = pd.DataFrame(rows)
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            fallback = path.with_suffix(".jsonl")
            jsonio.write_jsonl(fallback, records)
            return fallback

    jsonio.write_jsonl(path, records)
    return path


def load_corpus_records(path: Path) -> List[Dict[str, Any]]:
    """Load corpus from jsonl/parquet; transparently fallback from parquet to jsonl."""

    source = path
    if not source.exists() and source.suffix.lower() == ".parquet":
        fallback = source.with_suffix(".jsonl")
        if fallback.exists():
            source = fallback
    if source.suffix.lower() == ".jsonl":
        records = jsonio.read_jsonl(source)
        records.sort(key=lambda item: item["canonical_smiles"])
        return records
    if source.suffix.lower() == ".parquet":
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(source)
        rows = frame.to_dict(orient="records")
        return _expand_from_tabular(rows)
    records = jsonio.read_jsonl(source)
    records.sort(key=lambda item: item["canonical_smiles"])
    return records


def compute_corpus_sha256(records: List[Dict[str, Any]]) -> str:
    canonical = [record["canonical_smiles"] for record in records]
    payload = json.dumps(sorted(canonical), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
