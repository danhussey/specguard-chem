from __future__ import annotations

"""Adapter that proxies runner requests to the OpenAI Chat Completions API."""

from datetime import datetime, timezone
import hashlib
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
    track = "external"
    is_external = True

    def __init__(
        self,
        *,
        seed: int = 0,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 512,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        policy: str = "default",
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
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.policy = policy

    def model_metadata(self) -> Dict[str, Any]:
        prompt_template_hash = hashlib.sha256(
            f"{self.system_prompt}|{self.policy}".encode("utf-8")
        ).hexdigest()
        return {
            "provider": "openai",
            "adapter_name": self.name,
            "model_id": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "prompt_template_hash": prompt_template_hash,
            "policy": self.policy,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "track": self.track,
            "is_external": self.is_external,
        }

    def step(self, req: AgentRequest) -> AgentResponse:
        policy_response = self._policy_pre_step(req)
        if policy_response is not None:
            self._record_step_artifacts(
                {
                    "raw_model_output": json.dumps(
                        policy_response,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "model_metadata": self.model_metadata(),
                }
            )
            return policy_response

        prompt = self._build_prompt(req)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
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
        normalized = self._normalize_response(data, req)
        self._record_step_artifacts(
            {
                "raw_model_output": message,
                "model_metadata": self.model_metadata(),
            }
        )
        return normalized

    def _policy_pre_step(self, req: AgentRequest) -> Optional[AgentResponse]:
        if self.policy != "l3_verify_tooling":
            return None
        task = req.get("task") or {}
        protocol = str(task.get("protocol") or "L1")
        if protocol != "L3":
            return None
        tools = req.get("tools") or []
        verify_available = any(
            isinstance(tool, dict) and tool.get("name") == "verify" for tool in tools
        )
        if not verify_available:
            return None
        round_id = int(req.get("round") or 1)
        if round_id != 1:
            return None
        input_smiles = (task.get("input") or {}).get("smiles")
        if not isinstance(input_smiles, str) or not input_smiles:
            return None
        return {
            "action": "tool_call",
            "name": "verify",
            "args": {"smiles": input_smiles},
            "p_hard_pass": 0.5,
        }

    def _build_prompt(self, req: AgentRequest) -> list[Dict[str, Any]]:
        task = req.get("task", {})
        spec = req.get("spec", {})
        failure_vector = req.get("failure_vector")
        tools = req.get("tools") or []
        interrupt = req.get("interrupt")
        tool_names = [tool.get("name") for tool in tools if tool.get("name")]
        instructions = {
            "task": task,
            "spec": spec,
            "failure_vector": failure_vector,
            "interrupt": interrupt,
            "available_tools": tool_names,
            "output_schema": {
                "action": "propose | tool_call | abstain (lowercase)",
                "smiles": "required if action == 'propose'",
                "name": "required if action == 'tool_call'",
                "args": "object, required if action == 'tool_call'",
                "reason": "required if action == 'abstain'",
                "p_hard_pass": (
                    "float between 0 and 1 (optional; probability the final proposal "
                    "passes hard constraints)"
                ),
                "interrupt_ack": {
                    "acknowledged": "bool (required if interrupt present)",
                    "restate_goal": "bool (required if interrupt present)",
                    "report_state": "bool (required if interrupt present)",
                    "goal": "short restatement of the goal (optional)",
                    "state": "short status update (optional)",
                },
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
                "If interrupt is present, include interrupt_ack and do not claim completion.",
            ],
        }
        if self.policy == "l3_verify_tooling":
            instructions["rules"].extend(
                [
                    (
                        "For L3 tasks with verify available, call verify(input_smiles) before "
                        "any proposal whenever possible."
                    ),
                    (
                        "Avoid finalising an L3 candidate without checking margins from "
                        "verify feedback."
                    ),
                    (
                        "Set p_hard_pass monotonically with minimum signed hard margin "
                        "(larger margin => higher confidence)."
                    ),
                ]
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(instructions)},
        ]

    def _normalize_response(
        self, data: Dict[str, Any], req: AgentRequest
    ) -> AgentResponse:
        action = str(data.get("action", "")).strip().lower()
        raw_prob = data.get("p_hard_pass", data.get("confidence"))
        p_hard_pass = self._extract_p_hard_pass(raw_prob)
        interrupt_ack = self._normalize_interrupt_ack(data.get("interrupt_ack"))
        tools = {
            tool.get("name") for tool in (req.get("tools") or []) if tool.get("name")
        }

        if action not in ALLOWED_ACTIONS:
            return {
                "action": "abstain",
                "reason": "Model returned invalid action",
                "p_hard_pass": p_hard_pass,
            }

        if action == "propose":
            smiles = data.get("smiles")
            if not isinstance(smiles, str) or not smiles.strip():
                return {
                    "action": "abstain",
                    "reason": "Missing SMILES for proposal",
                    "p_hard_pass": p_hard_pass,
                    "interrupt_ack": interrupt_ack,
                }
            return {
                "action": "propose",
                "smiles": smiles.strip(),
                "p_hard_pass": p_hard_pass,
                "interrupt_ack": interrupt_ack,
            }

        if action == "tool_call":
            name = data.get("name")
            args = data.get("args")
            if not isinstance(name, str) or name not in tools:
                return {
                    "action": "abstain",
                    "reason": "Requested unavailable tool",
                    "p_hard_pass": p_hard_pass,
                    "interrupt_ack": interrupt_ack,
                }
            if not isinstance(args, dict):
                return {
                    "action": "abstain",
                    "reason": "Tool call missing arguments object",
                    "p_hard_pass": p_hard_pass,
                    "interrupt_ack": interrupt_ack,
                }
            return {
                "action": "tool_call",
                "name": name,
                "args": args,
                "p_hard_pass": p_hard_pass,
                "interrupt_ack": interrupt_ack,
            }

        # abstain
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "Model chose to abstain."
        return {
            "action": "abstain",
            "reason": reason.strip(),
            "p_hard_pass": p_hard_pass,
            "interrupt_ack": interrupt_ack,
        }

    @staticmethod
    def _extract_p_hard_pass(value: Any) -> float:
        try:
            if value is None:
                raise ValueError
            prob = float(value)
        except (TypeError, ValueError):
            prob = 0.5
        return max(0.0, min(1.0, prob))

    @staticmethod
    def _normalize_interrupt_ack(value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        payload: Dict[str, Any] = {
            "acknowledged": bool(value.get("acknowledged")),
            "restate_goal": bool(value.get("restate_goal")),
            "report_state": bool(value.get("report_state")),
        }
        goal = value.get("goal")
        state = value.get("state")
        if goal:
            payload["goal"] = str(goal)
        if state:
            payload["state"] = str(state)
        return payload


__all__ = ["OpenAIChatAdapter", "DEFAULT_MODEL"]
