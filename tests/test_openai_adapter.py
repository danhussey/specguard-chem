from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from specguard_chem.models import openai_adapter as oa
from specguard_chem.models.anthropic_adapter import AnthropicChatAdapter
from specguard_chem.models.deepseek_adapter import DeepSeekChatAdapter
from specguard_chem.models.openai_adapter import OpenAIChatAdapter
from specguard_chem.runner.runner import normalize_agent_response
from specguard_chem.runner.adapter_api import AgentRequest


class _Choice:
    def __init__(self, content: str = "", tool_calls: list[Any] | None = None) -> None:
        self.message = type(
            "Message",
            (),
            {"content": content, "tool_calls": tool_calls},
        )


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
        "choice": _Choice(
            content=json.dumps(
                {"action": "propose", "smiles": "CCO", "p_hard_pass": 0.9}
            )
        ),
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
    assert response["p_hard_pass"] == 0.9


def test_openai_adapter_parses_fenced_json_response() -> None:
    payload = {
        "choice": _Choice(
            content='```json\n{"action": "propose", "smiles": "CCN"}\n```'
        ),
    }
    adapter = OpenAIChatAdapter(client=_FakeClient(payload))
    response = adapter.step({"task": {}, "round": 1, "tools": [], "failure_vector": None})
    assert response["action"] == "propose"
    assert response["smiles"] == "CCN"


def test_openai_adapter_parses_strict_tool_call() -> None:
    tool_calls = [
        {
            "function": {
                "name": "specguard_action",
                "arguments": json.dumps(
                    {
                        "action": "propose",
                        "smiles": "CCN",
                        "name": "",
                        "tool_smiles": "",
                        "reason_code": "repair_attempt",
                        "confidence": 0.8,
                        "p_hard_pass": 0.8,
                        "interrupt_acknowledged": False,
                        "interrupt_restate_goal": False,
                        "interrupt_report_state": False,
                        "interrupt_resume_token": "",
                    }
                ),
            }
        }
    ]
    payload = {"choice": _Choice(tool_calls=tool_calls)}
    adapter = OpenAIChatAdapter(
        client=_FakeClient(payload),
        interface_tier="strict-tool-call",
    )
    response = adapter.step({"task": {}, "round": 1, "tools": [], "failure_vector": None})
    assert response["action"] == "propose"
    assert response["smiles"] == "CCN"
    assert adapter.model_metadata()["interface_tier"] == "strict-tool-call"


def test_openai_adapter_maps_public_accept_to_proposal() -> None:
    payload = {
        "choice": _Choice(content=json.dumps({"action": "ACCEPT", "smiles": "CCO"})),
    }
    adapter = OpenAIChatAdapter(client=_FakeClient(payload))
    response = adapter.step({"task": {}, "round": 1, "tools": [], "failure_vector": None})
    assert response["action"] == "propose"
    assert response["smiles"] == "CCO"
    assert response["declared_public_action"] == "ACCEPT"


def test_openai_adapter_maps_public_reject_to_supplied_candidate() -> None:
    payload = {"choice": _Choice(content=json.dumps({"action": "REJECT"}))}
    adapter = OpenAIChatAdapter(client=_FakeClient(payload))
    req: AgentRequest = {
        "task": {"input": {"smiles": "CCO"}},
        "round": 1,
        "tools": [],
        "failure_vector": None,
    }
    response = adapter.step(req)
    assert response["action"] == "propose"
    assert response["smiles"] == "CCO"
    assert response["declared_public_action"] == "REJECT"


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


def test_openai_adapter_invalid_action(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choice": _Choice(content=json.dumps({"action": "INVALID", "reason": "oops"})),
    }
    adapter = OpenAIChatAdapter(client=_FakeClient(payload))
    req: AgentRequest = {"task": {}, "round": 1, "tools": [], "failure_vector": None}
    response = adapter.step(req)
    normalized = normalize_agent_response(response, allowed_tools=set())
    assert normalized["action"] == "abstain"
    assert normalized["schema_error"]
    assert normalized["schema_error_type"] == "invalid_action"


def test_openai_adapter_tool_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choice": _Choice(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "name": "verify",
                    "args": {"smiles": "CC"},
                }
            )
        ),
    }
    adapter = OpenAIChatAdapter(client=_FakeClient(payload))
    req: AgentRequest = {
        "task": {},
        "round": 2,
        "tools": [{"name": "different", "schema": {}}],
        "failure_vector": None,
    }
    response = adapter.step(req)
    normalized = normalize_agent_response(response, allowed_tools={"different"})
    assert normalized["action"] == "abstain"
    assert normalized["schema_error"]
    assert normalized["schema_error_type"] == "invalid_tool_call_name"


def test_openai_adapter_l3_verify_policy_calls_verify_first() -> None:
    payload = {
        "choice": _Choice(content=json.dumps({"action": "abstain", "reason": "unused"})),
    }
    adapter = OpenAIChatAdapter(client=_FakeClient(payload), policy="l3_verify_tooling")
    req: AgentRequest = {
        "task": {
            "protocol": "L3",
            "input": {"smiles": "CCO"},
        },
        "round": 1,
        "tools": [{"name": "verify", "schema": {"smiles": "string"}}],
        "failure_vector": None,
    }
    response = adapter.step(req)
    assert response["action"] == "tool_call"
    assert response["name"] == "verify"


def test_deepseek_adapter_uses_openai_compatible_response() -> None:
    payload = {
        "choice": _Choice(
            content=json.dumps({"action": "propose", "smiles": "CCN", "confidence": 0.8})
        ),
    }
    adapter = DeepSeekChatAdapter(client=_FakeClient(payload), model="deepseek-chat")
    response = adapter.step({"task": {}, "round": 1, "tools": [], "failure_vector": None})
    assert response["action"] == "propose"
    assert response["smiles"] == "CCN"
    assert response["p_hard_pass"] == 0.8
    assert adapter.model_metadata()["provider"] == "deepseek"


class _AnthropicMessages:
    def __init__(self, content: Any) -> None:
        self.content = content

    def create(self, **_: Any) -> Any:
        return {"content": self.content}


class _AnthropicClient:
    def __init__(self, content: Any) -> None:
        self.messages = _AnthropicMessages(content)


def test_anthropic_adapter_parses_messages_response() -> None:
    content = [
        {
            "type": "tool_use",
            "name": "specguard_action",
            "input": {
                "action": "ABSTAIN",
                "smiles": "",
                "name": "",
                "tool_smiles": "",
                "reason_code": "insufficient_information",
                "confidence": 0.0,
                "p_hard_pass": 0.0,
                "interrupt_acknowledged": False,
                "interrupt_restate_goal": False,
                "interrupt_report_state": False,
                "interrupt_resume_token": "",
            },
        }
    ]
    adapter = AnthropicChatAdapter(client=_AnthropicClient(content))
    response = adapter.step({"task": {}, "round": 1, "tools": [], "failure_vector": None})
    assert response["action"] == "abstain"
    assert response["declared_public_action"] == "ABSTAIN"
    assert adapter.model_metadata()["provider"] == "anthropic"
