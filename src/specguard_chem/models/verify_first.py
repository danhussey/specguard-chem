from __future__ import annotations

"""L3 baseline that explicitly calls verify() before proposing."""

import math
from typing import Any, Dict, List, Optional

from ..config import SpecModel
from ..runner.adapter_api import AgentRequest, AgentResponse
from ..runner.protocols import ConstraintEvaluator
from ..verifiers import canonicalize_smiles
from .base_adapter import BaseAdapter
from .local_mutation import LocalMutationAdapter

_SAFE_PROPOSALS: tuple[str, ...] = (
    "CC(=O)NC1=CC=CC=C1O",
    "CCOC(=O)N(CC)CCO",
    "COc1ccccc1O",
    "CCN(CC)CC",
)


class VerifyFirstAdapter(BaseAdapter):
    name = "verify_first"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._local_search = LocalMutationAdapter(seed=seed)

    def step(self, req: AgentRequest) -> AgentResponse:
        task = req.get("task") or {}
        protocol = str(task.get("protocol") or "L1")
        round_index = int(req.get("round") or 1)
        failure_vector = req.get("failure_vector")
        interrupt = req.get("interrupt") or {}
        input_smiles = (task.get("input") or {}).get("smiles")
        start_smiles = (
            canonicalize_smiles(str(input_smiles))
            if isinstance(input_smiles, str) and input_smiles
            else _SAFE_PROPOSALS[0]
        )

        if protocol == "L3" and failure_vector is None and self._has_verify_tool(req):
            response: AgentResponse = {
                "action": "tool_call",
                "name": "verify",
                "args": {"smiles": start_smiles},
            }
            if interrupt:
                response["interrupt_ack"] = _interrupt_ack(interrupt)
            return response

        proposal = self._select_proposal(
            start_smiles=start_smiles,
            failure_vector=failure_vector,
            round_index=round_index,
            task=task,
            spec_payload=req.get("spec"),
        )
        response = {
            "action": "propose",
            "smiles": proposal,
            "p_hard_pass": self._probability_from_feedback(failure_vector),
        }
        if interrupt:
            response["interrupt_ack"] = _interrupt_ack(interrupt)
        return response

    @staticmethod
    def _has_verify_tool(req: AgentRequest) -> bool:
        tools = req.get("tools") or []
        if not isinstance(tools, list):
            return False
        for tool in tools:
            if isinstance(tool, dict) and tool.get("name") == "verify":
                return True
        return False

    def _select_proposal(
        self,
        *,
        start_smiles: str,
        failure_vector: Optional[Dict[str, Any]],
        round_index: int,
        task: Dict[str, Any],
        spec_payload: Any,
    ) -> str:
        if isinstance(spec_payload, dict):
            try:
                spec = SpecModel.model_validate(spec_payload)
                input_smiles = (task.get("input") or {}).get("smiles")
                evaluator = ConstraintEvaluator(spec, input_smiles=input_smiles)
                # Search with one extra step after the initial verify round.
                return self._local_search._search(
                    task=task,
                    evaluator=evaluator,
                    round_index=max(round_index + 1, 2),
                )
            except Exception:
                pass

        candidates: List[str] = []

        def _add(smiles: str) -> None:
            canonical = canonicalize_smiles(smiles)
            if canonical and canonical not in candidates:
                candidates.append(canonical)

        _add(start_smiles)

        hard_fail_ids = set()
        if isinstance(failure_vector, dict):
            for fail in failure_vector.get("hard_fails", []):
                if isinstance(fail, dict) and fail.get("id"):
                    hard_fail_ids.add(str(fail["id"]))
            for fail_id in failure_vector.get("hard_fail_ids", []):
                if isinstance(fail_id, str):
                    hard_fail_ids.add(fail_id)

        directions = _extract_directional_signals(failure_vector)
        if "similarity" in " ".join(sorted(hard_fail_ids)).lower():
            _add(start_smiles)

        if directions.get("too_high", {}).get("logP") or directions.get("too_high", {}).get("MW"):
            _add(_SAFE_PROPOSALS[1])
        if directions.get("too_high", {}).get("TPSA") or directions.get("too_high", {}).get("HBD"):
            _add(_SAFE_PROPOSALS[2])
        if directions.get("too_high", {}).get("ROTB"):
            _add(_SAFE_PROPOSALS[2])
        if directions.get("too_low", {}).get("logP"):
            _add(_SAFE_PROPOSALS[3])

        for fallback in _SAFE_PROPOSALS:
            _add(fallback)
        if not candidates:
            return _SAFE_PROPOSALS[0]
        return candidates[(max(round_index, 1) - 1) % len(candidates)]

    @staticmethod
    def _probability_from_feedback(failure_vector: Optional[Dict[str, Any]]) -> float:
        if not isinstance(failure_vector, dict):
            return 0.55
        margins = failure_vector.get("margins")
        if not isinstance(margins, list) or not margins:
            return 0.55
        min_margin: Optional[float] = None
        for item in margins:
            if not isinstance(item, dict):
                continue
            value = item.get("distance_to_bound")
            try:
                margin = float(value)
            except (TypeError, ValueError):
                continue
            if min_margin is None or margin < min_margin:
                min_margin = margin
        if min_margin is None:
            return 0.55
        score = 1.0 / (1.0 + math.exp(-4.0 * min_margin))
        return max(0.01, min(0.99, float(score)))


def _extract_directional_signals(
    failure_vector: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, bool]]:
    signals: Dict[str, Dict[str, bool]] = {"too_high": {}, "too_low": {}}
    if not isinstance(failure_vector, dict):
        return signals
    results = failure_vector.get("constraint_results")
    if not isinstance(results, list):
        return signals
    for result in results:
        if not isinstance(result, dict):
            continue
        details = result.get("property_details")
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict):
                continue
            prop = detail.get("property")
            bounds = detail.get("bounds")
            value = detail.get("value")
            if not isinstance(prop, str) or not isinstance(bounds, dict):
                continue
            try:
                lower = float(bounds.get("min"))
                upper = float(bounds.get("max"))
                observed = float(value)
            except (TypeError, ValueError):
                continue
            if observed > upper:
                signals["too_high"][prop] = True
            elif observed < lower:
                signals["too_low"][prop] = True
    return signals


def _interrupt_ack(interrupt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "acknowledged": True,
        "restate_goal": True,
        "report_state": True,
        "resume_token": interrupt.get("resume_token"),
    }
