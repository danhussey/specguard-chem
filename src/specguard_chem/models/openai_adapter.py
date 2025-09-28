from __future__ import annotations

"""Adapter that proxies runner requests to the OpenAI Chat Completions API."""

import json
import os
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from .base_adapter import BaseAdapter
from ..runner.adapter_api import AgentRequest, AgentResponse

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous medicinal chemistry assistant that must obey the provided "
    "specification. Always respond with compact JSON and no additional prose."
)


class OpenAIChatAdapter(BaseAdapter):
    """Minimal adapter that calls the OpenAI Chat Completions API per runner step."""

    name = "openai_chat"

    def __init__(
        self,
        *,
        seed: int = 0,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        client: Optional[OpenAI] = None,
    ) -> None:
        super().__init__(seed=seed)
        if client is None:
            if OpenAI is None:  # pragma: no cover - optional dependency guard
                raise RuntimeError("Install the 'openai' package to use OpenAIChatAdapter.")
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Export it before using the OpenAIChatAdapter."
                )
            client = OpenAI()
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def step(self, req: AgentRequest) -> AgentResponse:
        prompt = self._build_prompt(req)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        message = response.choices[0].message.content
        if not message:
            raise RuntimeError("OpenAI API returned empty message content")
        try:
            data = json.loads(message)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI response was not valid JSON. Configure the system prompt to force JSON output."
            ) from exc
        if not isinstance(data, dict):  # pragma: no cover - defensive
            raise RuntimeError("OpenAI response JSON must be an object")
        data.setdefault("confidence", 0.5)
        return data  # type: ignore[return-value]

    def _build_prompt(self, req: AgentRequest) -> list[Dict[str, Any]]:
        task = req.get("task", {})
        failure_vector = req.get("failure_vector")
        tools = req.get("tools") or []
        interrupt = req.get("interrupt")
        tool_instructions = (
            "Available tool: verify(smiles) returns property checks prior to submission."
            if any(tool.get("name") == "verify" for tool in tools)
            else "No tools available."
        )
        failure_summary = "None"
        if failure_vector:
            failure_summary = json.dumps(failure_vector)
        interrupt_summary = json.dumps(interrupt) if interrupt else "None"
        content = (
            "You must return JSON with keys: action, and depending on the action, either "
            "smiles (for propose), name/args (for tool_call), or reason (for abstain). "
            "You may also include confidence (float 0-1).\n\n"
            f"Task metadata: {json.dumps(task)}\n"
            f"Failure vector from the runner: {failure_summary}\n"
            f"Interrupt signal: {interrupt_summary}\n"
            f"Tools: {tool_instructions}\n"
            "Never include prose outside the JSON object."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]


__all__ = ["OpenAIChatAdapter", "DEFAULT_MODEL"]
