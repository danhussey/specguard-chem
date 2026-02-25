from __future__ import annotations

"""Adapter base class."""

from typing import Any, Dict

from ..runner.adapter_api import AgentRequest, AgentResponse


class BaseAdapter:
    name: str = "base"
    track: str = "closed_book"
    is_external: bool = False

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        self._last_step_artifacts: Dict[str, Any] = {}

    def step(self, req: AgentRequest) -> AgentResponse:  # pragma: no cover - interface
        """Return one step of interaction. Must be pure & deterministic for a given seed."""

        raise NotImplementedError

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed

    def model_metadata(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.name,
            "track": self.track,
            "is_external": self.is_external,
        }

    def _record_step_artifacts(self, payload: Dict[str, Any]) -> None:
        self._last_step_artifacts = dict(payload)

    def consume_step_artifacts(self) -> Dict[str, Any]:
        payload = dict(self._last_step_artifacts)
        self._last_step_artifacts = {}
        return payload
