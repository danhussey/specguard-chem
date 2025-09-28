from __future__ import annotations

"""Adapter that delegates reasoning to an external process."""

import json
import os
import subprocess
from typing import Sequence

from .base_adapter import BaseAdapter
from ..runner.adapter_api import AgentRequest, AgentResponse

DEFAULT_ENV_VAR = "SPEC_GUARD_PROCESS_ADAPTER_CMD"


class ProcessAdapter(BaseAdapter):
    """Spawn an external command for every runner step."""

    name = "process"

    def __init__(self, *, seed: int = 0, command: Sequence[str] | None = None) -> None:
        super().__init__(seed=seed)
        if command is None:
            env_value = os.getenv(DEFAULT_ENV_VAR)
            if not env_value:
                raise RuntimeError(
                    "ProcessAdapter requires a command. Set "
                    "SPEC_GUARD_PROCESS_ADAPTER_CMD or pass a command list to the constructor."
                )
            command = env_value.split()
        self.command = list(command)

    def step(self, req: AgentRequest) -> AgentResponse:
        payload = json.dumps(req)
        proc = subprocess.run(
            self.command,
            input=payload.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            stderr_text = proc.stderr.decode("utf-8", "ignore")
            raise RuntimeError(
                "External adapter command failed with code "
                f"{proc.returncode}: {stderr_text}"
            )
        try:
            data = json.loads(proc.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError("External adapter returned invalid JSON") from exc
        if not isinstance(data, dict):  # pragma: no cover - defensive
            raise RuntimeError("External adapter must return a JSON object")
        data.setdefault("confidence", 0.5)
        return data  # type: ignore[return-value]


__all__ = ["ProcessAdapter", "DEFAULT_ENV_VAR"]
