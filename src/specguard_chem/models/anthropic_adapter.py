from __future__ import annotations

"""Anthropic Messages API external adapters."""

from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Dict, Optional
from urllib import error, request

from .openai_adapter import (
    DEFAULT_SYSTEM_PROMPT,
    OUTPUT_TOOL_NAME,
    OpenAIChatAdapter,
    _coerce_structured_response,
    _parse_json_object,
)
from ..runner.adapter_api import AgentRequest, AgentResponse

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"


class _AnthropicMessagesClient:
    def __init__(
        self,
        *,
        api_key: str,
        timeout: float | None = 60.0,
        anthropic_beta: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.anthropic_beta = anthropic_beta
        self.messages = self

    def create(self, **payload: Any) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self.anthropic_beta:
            headers["anthropic-beta"] = self.anthropic_beta
        req = request.Request(ANTHROPIC_API_URL, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - exercised only live
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API error {exc.code}: {details}") from exc


class AnthropicChatAdapter(OpenAIChatAdapter):
    name = "anthropic_chat"
    track = "external"
    is_external = True

    def __init__(
        self,
        *,
        seed: int = 0,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        max_tokens: int | None = 512,
        timeout: float | None = 60.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        policy: str = "default",
        api_key_env: str = "ANTHROPIC_API_KEY",
        api_key: str | None = None,
        anthropic_beta: str | None = None,
        interface_tier: str | None = None,
        client: Optional[Any] = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
    ) -> None:
        if client is None:
            resolved_key = api_key or os.getenv(api_key_env)
            if not resolved_key:
                raise RuntimeError(
                    f"{api_key_env} is not set. Export it before using this adapter."
                )
            client = _AnthropicMessagesClient(
                api_key=resolved_key,
                timeout=timeout,
                anthropic_beta=anthropic_beta,
            )
        super().__init__(
            seed=seed,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            system_prompt=system_prompt,
            policy=policy,
            provider_name="anthropic",
            response_format_json=False,
            interface_tier=interface_tier or "strict-tool-call",
            client=client,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self.anthropic_beta = anthropic_beta

    def model_metadata(self) -> Dict[str, Any]:
        prompt_template_hash = hashlib.sha256(
            f"{self.system_prompt}|{self.policy}|{self.interface_tier}".encode("utf-8")
        ).hexdigest()
        return {
            "provider": "anthropic",
            "adapter_name": self.name,
            "model_id": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "prompt_template_hash": prompt_template_hash,
            "schema_hash": self.action_schema_hash,
            "interface_tier": self.interface_tier,
            "provider_feature": self._provider_feature(),
            "anthropic_beta": self.anthropic_beta,
            "policy": self.policy,
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff,
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
        system = str(prompt[0].get("content", self.system_prompt))
        user_content = str(prompt[1].get("content", ""))
        request_payload: Dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": self.max_tokens or 512,
        }
        if self.temperature is not None:
            request_payload["temperature"] = self.temperature
        if self.top_p is not None:
            request_payload["top_p"] = self.top_p
        if self.interface_tier == "strict-tool-call":
            request_payload["tools"] = [
                {
                    "name": OUTPUT_TOOL_NAME,
                    "description": (
                        "Return the next SpecGuard-Chem runner action. This is an "
                        "output envelope, not a chemistry verifier."
                    ),
                    "input_schema": self.action_schema,
                }
            ]
            request_payload["tool_choice"] = {
                "type": "tool",
                "name": OUTPUT_TOOL_NAME,
            }

        response = self._call_with_retries(
            lambda: self.client.messages.create(**request_payload)
        )
        if self.interface_tier == "strict-tool-call":
            data, raw = _anthropic_tool_input(response)
            if data is None:
                data = self._interface_failure_payload(
                    "missing_forced_tool_call",
                    "Anthropic response did not contain the forced action tool call.",
                )
                raw = raw or _anthropic_text(response)
        else:
            raw = _anthropic_text(response)
            data = _parse_json_object(raw) if raw else None
            if data is None:
                data = self._interface_failure_payload(
                    "empty_or_unparseable_output",
                    "Anthropic response was empty or not valid JSON.",
                )
        normalized = self._normalize_response(_coerce_structured_response(data), req)
        self._record_step_artifacts(
            {
                "raw_model_output": raw,
                "model_metadata": self.model_metadata(),
            }
        )
        return normalized

    def _provider_feature(self) -> str:
        if self.interface_tier == "strict-tool-call":
            return "messages.tools.input_schema.forced_tool_choice"
        return "messages.prompt_json"


class AnthropicChatVerifyL3Adapter(AnthropicChatAdapter):
    name = "anthropic_chat_verify_l3"

    def __init__(self, *, seed: int = 0, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs.setdefault("policy", "l3_verify_tooling")
        super().__init__(seed=seed, **kwargs)


def _anthropic_text(response: Any) -> str:
    if isinstance(response, dict):
        content = response.get("content")
    else:
        content = getattr(response, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def _anthropic_tool_input(response: Any) -> tuple[Dict[str, Any] | None, str]:
    if isinstance(response, dict):
        content = response.get("content")
    else:
        content = getattr(response, "content", None)
    if not isinstance(content, list):
        return None, ""
    raw = json.dumps(content, sort_keys=True, ensure_ascii=True, default=str)
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            name = block.get("name")
            input_payload = block.get("input")
        else:
            block_type = getattr(block, "type", None)
            name = getattr(block, "name", None)
            input_payload = getattr(block, "input", None)
        if block_type == "tool_use" and name == OUTPUT_TOOL_NAME:
            if isinstance(input_payload, dict):
                return input_payload, json.dumps(
                    input_payload,
                    sort_keys=True,
                    ensure_ascii=True,
                )
            return None, raw
    return None, raw


__all__ = ["AnthropicChatAdapter", "AnthropicChatVerifyL3Adapter"]
