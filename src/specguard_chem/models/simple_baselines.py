from __future__ import annotations

"""Simple decision-control baselines for reviewer sanity checks."""

import hashlib
from typing import Any

from .base_adapter import BaseAdapter
from ..runner.adapter_api import AgentRequest, AgentResponse

SAFE_FALLBACK = "CC(=O)NC1=CC=CC=C1O"
REJECT_SENTINEL = "not_a_smiles"


def _input_or_fallback(req: AgentRequest) -> str:
    task = req.get("task") or {}
    input_block = task.get("input") if isinstance(task.get("input"), dict) else {}
    smiles = input_block.get("smiles") or input_block.get("candidate_smiles")
    return str(smiles) if isinstance(smiles, str) and smiles else SAFE_FALLBACK


class AlwaysAcceptAdapter(BaseAdapter):
    name = "always_accept"
    track = "primary_closed_book"

    def step(self, req: AgentRequest) -> AgentResponse:
        _ = req
        return {"action": "propose", "smiles": SAFE_FALLBACK, "p_hard_pass": 0.9}


class AlwaysRejectAdapter(BaseAdapter):
    name = "always_reject"
    track = "primary_closed_book"

    def step(self, req: AgentRequest) -> AgentResponse:
        _ = req
        return {"action": "propose", "smiles": REJECT_SENTINEL, "p_hard_pass": 0.05}


class AlwaysAbstainAdapter(BaseAdapter):
    name = "always_abstain"
    track = "primary_closed_book"

    def step(self, req: AgentRequest) -> AgentResponse:
        _ = req
        return {"action": "abstain", "reason": "Baseline always abstains.", "p_hard_pass": 0.1}


class RandomActionAdapter(BaseAdapter):
    name = "random_action"
    track = "primary_closed_book"

    def step(self, req: AgentRequest) -> AgentResponse:
        task = req.get("task") or {}
        payload = str(task.get("rendered_agent_input") or "") + f"|{self.seed}"
        index = int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16) % 3
        if index == 0:
            return {"action": "propose", "smiles": _input_or_fallback(req), "p_hard_pass": 0.5}
        if index == 1:
            return {"action": "propose", "smiles": REJECT_SENTINEL, "p_hard_pass": 0.2}
        return {"action": "abstain", "reason": "Deterministic random abstention.", "p_hard_pass": 0.3}


class SchemaValidDummyAdapter(BaseAdapter):
    name = "schema_valid_dummy"
    track = "primary_closed_book"

    def step(self, req: AgentRequest) -> AgentResponse:
        _ = req
        return {"action": "propose", "smiles": SAFE_FALLBACK, "p_hard_pass": 0.5}


class VerifierGuidedGreedyAdapter(BaseAdapter):
    name = "verifier_guided_greedy"
    track = "tool_enabled"

    def step(self, req: AgentRequest) -> AgentResponse:
        round_index = int(req.get("round") or 1)
        task = req.get("task") or {}
        protocol = str(task.get("protocol") or "L1")
        tools = req.get("tools") or []
        verify_available = any(isinstance(tool, dict) and tool.get("name") == "verify" for tool in tools)
        if protocol == "L3" and round_index == 1 and verify_available:
            return {"action": "tool_call", "name": "verify", "args": {"smiles": _input_or_fallback(req)}}
        failure = req.get("failure_vector") if isinstance(req.get("failure_vector"), dict) else {}
        hard_fails: list[Any] = []
        if isinstance(failure, dict):
            hard_fails = list(failure.get("hard_fails") or failure.get("hard_fail_ids") or [])
        if hard_fails:
            return {"action": "propose", "smiles": SAFE_FALLBACK, "p_hard_pass": 0.6}
        return {"action": "propose", "smiles": _input_or_fallback(req), "p_hard_pass": 0.7}


__all__ = [
    "AlwaysAcceptAdapter",
    "AlwaysRejectAdapter",
    "AlwaysAbstainAdapter",
    "RandomActionAdapter",
    "SchemaValidDummyAdapter",
    "VerifierGuidedGreedyAdapter",
]
