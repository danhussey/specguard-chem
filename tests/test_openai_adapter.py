from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from specguard_chem.models import openai_adapter as oa
from specguard_chem.models.openai_adapter import OpenAIChatAdapter
from specguard_chem.runner.adapter_api import AgentRequest


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = type("Message", (), {"content": content})


class _ChatCompletions:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def create(self, **_: Any) -> Any:
        return type("Response", (), {"choices": [self._payload["choice"]]})


class _FakeClient:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.chat = type("Chat", (), {"completions": _ChatCompletions(payload)})


@pytest.fixture(autouse=True)
def _ensure_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_openai_adapter_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choice": _Choice(content=json.dumps({"action": "propose", "smiles": "CCO", "confidence": 0.9})),
    }
    adapter = OpenAIChatAdapter(client=_FakeClient(payload))
    req: AgentRequest = {
        "task": {"prompt": "Propose"},
        "round": 1,
        "tools": [],
        "failure_vector": None,
    }
    response = adapter.step(req)
    assert response["action"] == "propose"
    assert response["smiles"] == "CCO"
    assert response["confidence"] == 0.9


def test_openai_adapter_requires_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oa, "OpenAI", None, raising=True)
    with pytest.raises(RuntimeError, match="Install the 'openai' package"):
        oa.OpenAIChatAdapter()


def test_openai_adapter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def __init__(self) -> None:
            choice = _Choice("{}")
            completions = _ChatCompletions({"choice": choice})
            self.chat = type("Chat", (), {"completions": completions})

    monkeypatch.setattr(oa, "OpenAI", lambda: _Client(), raising=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        oa.OpenAIChatAdapter()
