from __future__ import annotations

"""Adapter base class."""

from ..runner.adapter_api import AgentRequest, AgentResponse


class BaseAdapter:
    name: str = "base"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def step(self, req: AgentRequest) -> AgentResponse:  # pragma: no cover - interface
        """Return one step of interaction. Must be pure & deterministic for a given seed."""

        raise NotImplementedError

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
