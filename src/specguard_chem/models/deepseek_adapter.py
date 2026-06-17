from __future__ import annotations

"""DeepSeek OpenAI-compatible external adapters."""

from typing import Any

from .openai_adapter import OpenAIChatAdapter


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


class DeepSeekChatAdapter(OpenAIChatAdapter):
    name = "deepseek_chat"
    track = "external"
    is_external = True

    def __init__(self, *, seed: int = 0, **kwargs: Any):
        kwargs = dict(kwargs)
        interface_tier = kwargs.get("interface_tier")
        kwargs.setdefault("model", DEFAULT_DEEPSEEK_MODEL)
        kwargs.setdefault("api_key_env", "DEEPSEEK_API_KEY")
        kwargs.setdefault(
            "base_url",
            DEEPSEEK_BETA_BASE_URL
            if interface_tier == "strict-tool-call"
            else DEEPSEEK_BASE_URL,
        )
        kwargs.setdefault("provider_name", "deepseek")
        super().__init__(seed=seed, **kwargs)


class DeepSeekChatVerifyL3Adapter(DeepSeekChatAdapter):
    name = "deepseek_chat_verify_l3"

    def __init__(self, *, seed: int = 0, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs.setdefault("policy", "l3_verify_tooling")
        super().__init__(seed=seed, **kwargs)


__all__ = ["DeepSeekChatAdapter", "DeepSeekChatVerifyL3Adapter"]
