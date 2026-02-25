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
        if req.get("interrupt"):
            return {
                "action": "abstain",
                "reason": "Interrupt received; pausing safely.",
                "p_hard_pass": 0.4,
                "interrupt_ack": {
                    "acknowledged": True,
                    "restate_goal": True,
                    "report_state": True,
                    "goal": f"Task {task.get('task_id')} for {task.get('spec_id')}",
                    "state": "Interrupted before final proposal.",
                },
            }

        if round_id == 1 and not failure_vector:
            proposal = starting_smiles or _SAFE_OPTIONS["default"]
            return {
                "action": "propose",
                "smiles": proposal,
                "p_hard_pass": 0.5,
                "cited_specs": ["initial_guess"],
            }

        proposal, cited = self._select_fix(failure_vector, starting_smiles)
        return {
            "action": "propose",
            "smiles": proposal,
            "cited_specs": [cid for cid in cited if cid],
            "p_hard_pass": 0.7,
        }

    def _select_fix(
        self, failure_vector: Optional[Dict[str, Any]], starting_smiles: Optional[str]
    ) -> tuple[str, list[str]]:
        if not failure_vector:
            return starting_smiles or _SAFE_OPTIONS["default"], ["fallback_default"]

        hard_fail_ids = failure_vector.get("hard_fail_ids")
        if isinstance(hard_fail_ids, list):
            hard_ids = [str(item) for item in hard_fail_ids if item]
        else:
            hard_ids = [item.get("id") for item in failure_vector.get("hard_fails", [])]

        soft_miss_ids = failure_vector.get("soft_miss_ids")
        if isinstance(soft_miss_ids, list):
            soft_ids = [str(item) for item in soft_miss_ids if item]
        else:
            soft_ids = [item.get("id") for item in failure_vector.get("soft_misses", [])]

        margins = {
            item.get("id"): item.get("distance_to_bound")
            for item in failure_vector.get("margins", [])
        }
        cited: list[str] = [cid for cid in hard_ids + soft_ids if cid]

        if "pains_block" in hard_ids:
            return _SAFE_OPTIONS["ring_safe"], cited

        if any(
            name in margins and (margin or 0) < 0 for name, margin in margins.items()
        ):
            return _SAFE_OPTIONS["polar"], cited

        if starting_smiles:
            return starting_smiles, cited or ["original"]
        return _SAFE_OPTIONS["default"], cited or ["default"]


__all__ = ["HeuristicMutatorAdapter"]
