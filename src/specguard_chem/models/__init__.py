from __future__ import annotations

"""Adapter registry."""

from typing import Any, Dict, Type

from .base_adapter import BaseAdapter
from .heuristic_mutator import HeuristicMutatorAdapter
from .open_source_example import OpenSourceExampleAdapter
from .abstention_guard import AbstentionGuardAdapter
from .process_adapter import ProcessAdapter
from .openai_adapter import OpenAIChatAdapter
from .openai_verify_l3 import OpenAIChatVerifyL3Adapter
from .corpus_search import CorpusSearchAdapter
from .local_mutation import LocalMutationAdapter
from .verify_first import VerifyFirstAdapter

_ADAPTERS: Dict[str, Type[BaseAdapter]] = {
    HeuristicMutatorAdapter.name: HeuristicMutatorAdapter,
    OpenSourceExampleAdapter.name: OpenSourceExampleAdapter,
    AbstentionGuardAdapter.name: AbstentionGuardAdapter,
    ProcessAdapter.name: ProcessAdapter,
    OpenAIChatAdapter.name: OpenAIChatAdapter,
    OpenAIChatVerifyL3Adapter.name: OpenAIChatVerifyL3Adapter,
    CorpusSearchAdapter.name: CorpusSearchAdapter,
    LocalMutationAdapter.name: LocalMutationAdapter,
    VerifyFirstAdapter.name: VerifyFirstAdapter,
}


def get_adapter(name: str, *, seed: int = 0) -> BaseAdapter:
    return build_adapter(name=name, seed=seed)


def get_adapter_class(name: str) -> Type[BaseAdapter]:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown adapter '{name}'") from exc


def build_adapter(name: str, *, seed: int = 0, **kwargs: Any) -> BaseAdapter:
    adapter_cls = get_adapter_class(name)
    return adapter_cls(seed=seed, **kwargs)


def register_adapter(adapter_cls: Type[BaseAdapter]) -> None:
    """Register a new adapter class by its declared name."""

    if not getattr(adapter_cls, "name", None):  # pragma: no cover - defensive
        raise ValueError("Adapter class must define a name")
    _ADAPTERS[adapter_cls.name] = adapter_cls


def available_adapters() -> Dict[str, Type[BaseAdapter]]:
    """Return the currently registered adapter mapping."""

    return dict(_ADAPTERS)
