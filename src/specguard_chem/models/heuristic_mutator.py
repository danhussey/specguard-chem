from __future__ import annotations

"""Deterministic RDKit-based mutator that attempts simple repairs."""

from typing import Any, Dict, Optional

from .base_adapter import BaseAdapter
from ..runner.adapter_api import AgentRequest, AgentResponse

_SAFE_OPTIONS = {
    "default": "CC(=O)NC1=CC=CC=C1O",
    "polar": "CCOC(=O)N(CC)CCO",
    "ring_safe": "COc1ccccc1O",
}


class HeuristicMutatorAdapter(BaseAdapter):
    name = "heuristic"

    def step(self, req: AgentRequest) -> AgentResponse:
        task = req.get("task", {})
        round_id = req.get("round", 1)
        failure_vector = req.get("failure_vector")
        input_block = task.get("input") or {}
        starting_smiles: Optional[str] = input_block.get("smiles")

        if round_id == 1 and not failure_vector:
            proposal = starting_smiles or _SAFE_OPTIONS["default"]
            return {"action": "propose", "smiles": proposal, "confidence": 0.5}

        proposal = self._select_fix(failure_vector, starting_smiles)
        cited = [item.get("id") for item in failure_vector.get("hard_fails", [])] if failure_vector else []
        return {
            "action": "propose",
            "smiles": proposal,
            "cited_specs": [cid for cid in cited if cid],
            "confidence": 0.7,
        }

    def _select_fix(
        self, failure_vector: Optional[Dict[str, Any]], starting_smiles: Optional[str]
    ) -> str:
        if not failure_vector:
            return starting_smiles or _SAFE_OPTIONS["default"]

        margins = {item.get("id"): item.get("distance_to_bound") for item in failure_vector.get("margins", [])}
        hard_ids = [item.get("id") for item in failure_vector.get("hard_fails", [])]

        if "pains_block" in hard_ids:
            return _SAFE_OPTIONS["ring_safe"]
        if margins.get("logP", 1) < 0:
            return _SAFE_OPTIONS["polar"]
        if margins.get("MW", 1) < 0:
            return _SAFE_OPTIONS["polar"]
        if starting_smiles:
            return starting_smiles
        return _SAFE_OPTIONS["default"]


__all__ = ["HeuristicMutatorAdapter"]
