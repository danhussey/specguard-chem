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
    "You are an autonomous medicinal chemistry assistant operating inside an "
    "automated test harness. Every reply MUST be a single JSON object with "
    "double-quoted keys and no trailing text."
)

ALLOWED_ACTIONS = {"propose", "tool_call", "abstain"}


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
                raise RuntimeError(
                    "Install the 'openai' package to use OpenAIChatAdapter."
                )
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
                "OpenAI response was not valid JSON. Configure the system prompt "
                "to force JSON output."
            ) from exc
        if not isinstance(data, dict):  # pragma: no cover - defensive
            raise RuntimeError("OpenAI response JSON must be an object")
        return self._normalize_response(data, req)

    def _build_prompt(self, req: AgentRequest) -> list[Dict[str, Any]]:
        task = req.get("task", {})
        failure_vector = req.get("failure_vector")
        tools = req.get("tools") or []
        interrupt = req.get("interrupt")
        tool_names = [tool.get("name") for tool in tools if tool.get("name")]
        instructions = {
            "task": task,
            "failure_vector": failure_vector,
            "interrupt": interrupt,
            "available_tools": tool_names,
            "output_schema": {
                "action": "propose | tool_call | abstain (lowercase)",
                "smiles": "required if action == 'propose'",
                "name": "required if action == 'tool_call'",
                "args": "object, required if action == 'tool_call'",
                "reason": "required if action == 'abstain'",
                "confidence": "float between 0 and 1 (optional)",
            },
            "rules": [
                "Always return a single JSON object. Never include markdown or prose.",
                (
                    "If a tool is available you may choose action 'tool_call'. Otherwise "
                    "avoid tool calls."
                ),
                (
                    "If unsure or missing data, respond with action 'abstain' and a concise "
                    "reason."
                ),
                "Respect the failure vector: try to fix hard fails before finalising.",
            ],
        }
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(instructions)},
        ]

    def _normalize_response(
        self, data: Dict[str, Any], req: AgentRequest
    ) -> AgentResponse:
        action = str(data.get("action", "")).strip().lower()
        confidence = self._extract_confidence(data.get("confidence"))
        tools = {
            tool.get("name") for tool in (req.get("tools") or []) if tool.get("name")
        }

        if action not in ALLOWED_ACTIONS:
            return {
                "action": "abstain",
                "reason": "Model returned invalid action",
                "confidence": confidence,
            }

        if action == "propose":
            smiles = data.get("smiles")
            if not isinstance(smiles, str) or not smiles.strip():
                return {
                    "action": "abstain",
                    "reason": "Missing SMILES for proposal",
                    "confidence": confidence,
                }
            return {
                "action": "propose",
                "smiles": smiles.strip(),
                "confidence": confidence,
            }

        if action == "tool_call":
            name = data.get("name")
            args = data.get("args")
            if not isinstance(name, str) or name not in tools:
                return {
                    "action": "abstain",
                    "reason": "Requested unavailable tool",
                    "confidence": confidence,
                }
            if not isinstance(args, dict):
                return {
                    "action": "abstain",
                    "reason": "Tool call missing arguments object",
                    "confidence": confidence,
                }
            return {
                "action": "tool_call",
                "name": name,
                "args": args,
                "confidence": confidence,
            }

        # abstain
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "Model chose to abstain."
        return {
            "action": "abstain",
            "reason": reason.strip(),
            "confidence": confidence,
        }

    @staticmethod
    def _extract_confidence(value: Any) -> float:
        try:
            if value is None:
                raise ValueError
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.5
        return max(0.0, min(1.0, confidence))


__all__ = ["OpenAIChatAdapter", "DEFAULT_MODEL"]
