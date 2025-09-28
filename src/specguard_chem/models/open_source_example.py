from __future__ import annotations

"""Lightweight deterministic adapter showcasing tool usage."""

from typing import Optional

from .base_adapter import BaseAdapter
from ..runner.adapter_api import AgentRequest, AgentResponse

_SAFE_PROPOSAL = "CC(=O)NC1=CC=CC=C1O"  # acetanilide-like scaffold


class OpenSourceExampleAdapter(BaseAdapter):
    name = "open_source_example"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)

    def step(self, req: AgentRequest) -> AgentResponse:
        task = req.get("task", {})
        protocol = task.get("protocol", "L1")
        failure_vector = req.get("failure_vector")
        round_id = req.get("round", 1)
        input_block = task.get("input") or {}
        starting_smiles: Optional[str] = input_block.get("smiles")

        if protocol == "L3" and round_id == 1 and failure_vector is None:
            return {
                "action": "tool_call",
                "name": "verify",
                "args": {"smiles": starting_smiles or _SAFE_PROPOSAL},
            }

        if failure_vector:
            hard_fails = [
                item.get("id") for item in failure_vector.get("hard_fails", [])
            ]
            proposal = starting_smiles or _SAFE_PROPOSAL
            if "pains_block" in hard_fails or not proposal:
                proposal = _SAFE_PROPOSAL
            return {
                "action": "propose",
                "smiles": proposal,
                "cited_specs": [fid for fid in hard_fails if fid],
                "confidence": 0.6,
            }

        proposal = starting_smiles or _SAFE_PROPOSAL
        return {"action": "propose", "smiles": proposal, "confidence": 0.5}
