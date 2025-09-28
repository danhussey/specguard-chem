from __future__ import annotations

"""Adapter that prefers abstaining near safety margins."""

from typing import Any, Dict, Optional

from .base_adapter import BaseAdapter
from ..runner.adapter_api import AgentRequest, AgentResponse

_SAFE_CANONICAL = "CC(=O)NC1=CC=CC=C1O"


class AbstentionGuardAdapter(BaseAdapter):
    name = "abstention_guard"

    def __init__(self, *, seed: int = 0, margin_threshold: float = 0.25) -> None:
        super().__init__(seed=seed)
        self.margin_threshold = margin_threshold

    def step(self, req: AgentRequest) -> AgentResponse:
        round_id = req.get("round", 1)
        failure_vector = req.get("failure_vector") or {}
        task = req.get("task", {})
        starting_smiles = (task.get("input") or {}).get("smiles")

        if round_id == 1 and not failure_vector:
            proposal = starting_smiles or _deterministic_choice(self.seed)
            return {
                "action": "propose",
                "smiles": proposal,
                "confidence": 0.55,
                "cited_specs": ["initial"],
            }

        hard_fails = failure_vector.get("hard_fails", [])
        if hard_fails:
            # Hard gating still violated; fall back to conservative scaffold.
            return {
                "action": "propose",
                "smiles": _deterministic_choice(self.seed + len(hard_fails)),
                "confidence": 0.45,
                "cited_specs": [item.get("id") for item in hard_fails if item.get("id")],
            }

        margins = [
            abs(item.get("distance_to_bound", 1.0) or 0.0)
            for item in failure_vector.get("margins", [])
        ]
        soft_deltas = [
            abs(item.get("delta", 0.0) or 0.0)
            for item in failure_vector.get("soft_misses", [])
        ]
        near_margin = any(value <= self.margin_threshold for value in margins if value is not None)
        near_soft = any(value <= self.margin_threshold for value in soft_deltas if value is not None)

        if near_margin or near_soft:
            return {
                "action": "abstain",
                "reason": "Uncertain compliance; margins too tight",
                "confidence": 0.35,
            }

        proposal = starting_smiles or _deterministic_choice(self.seed + round_id)
        return {
            "action": "propose",
            "smiles": proposal,
            "confidence": 0.6,
        }


def _deterministic_choice(seed: int) -> str:
    # Three deterministic safe scaffolds rotated by seed.
    candidates = (
        "CC(=O)NC1=CC=CC=C1O",
        "COc1ccc(N)cc1O",
        "CCOC(=O)N(CC)CCO",
    )
    index = abs(seed) % len(candidates)
    return candidates[index]


__all__ = ["AbstentionGuardAdapter"]
