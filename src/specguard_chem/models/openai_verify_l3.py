from __future__ import annotations

"""OpenAI baseline with explicit L3 verify-first policy."""

from .openai_adapter import OpenAIChatAdapter


class OpenAIChatVerifyL3Adapter(OpenAIChatAdapter):
    name = "openai_chat_verify_l3"

    def __init__(self, *, seed: int = 0, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("policy", "l3_verify_tooling")
        super().__init__(seed=seed, **kwargs)


__all__ = ["OpenAIChatVerifyL3Adapter"]
