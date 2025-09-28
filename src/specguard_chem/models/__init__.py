from __future__ import annotations

"""Adapter registry."""

from typing import Dict, Type

from .base_adapter import BaseAdapter
from .heuristic_mutator import HeuristicMutatorAdapter
from .open_source_example import OpenSourceExampleAdapter
from .abstention_guard import AbstentionGuardAdapter
from .process_adapter import ProcessAdapter
from .openai_adapter import OpenAIChatAdapter

_ADAPTERS: Dict[str, Type[BaseAdapter]] = {
    HeuristicMutatorAdapter.name: HeuristicMutatorAdapter,
    OpenSourceExampleAdapter.name: OpenSourceExampleAdapter,
    AbstentionGuardAdapter.name: AbstentionGuardAdapter,
    ProcessAdapter.name: ProcessAdapter,
    OpenAIChatAdapter.name: OpenAIChatAdapter,
}


def get_adapter(name: str, *, seed: int = 0) -> BaseAdapter:
    try:
        adapter_cls = _ADAPTERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown adapter '{name}'") from exc
    return adapter_cls(seed=seed)


def register_adapter(adapter_cls: Type[BaseAdapter]) -> None:
    """Register a new adapter class by its declared name."""

    if not getattr(adapter_cls, "name", None):  # pragma: no cover - defensive
        raise ValueError("Adapter class must define a name")
    _ADAPTERS[adapter_cls.name] = adapter_cls


def available_adapters() -> Dict[str, Type[BaseAdapter]]:
    """Return the currently registered adapter mapping."""

    return dict(_ADAPTERS)
