from __future__ import annotations

"""Adapter registry."""

from typing import Dict, Type

from .base_adapter import BaseAdapter
from .heuristic_mutator import HeuristicMutatorAdapter
from .open_source_example import OpenSourceExampleAdapter

_ADAPTERS: Dict[str, Type[BaseAdapter]] = {
    HeuristicMutatorAdapter.name: HeuristicMutatorAdapter,
    OpenSourceExampleAdapter.name: OpenSourceExampleAdapter,
}


def get_adapter(name: str, *, seed: int = 0) -> BaseAdapter:
    try:
        adapter_cls = _ADAPTERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown adapter '{name}'") from exc
    return adapter_cls(seed=seed)
