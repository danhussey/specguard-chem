from __future__ import annotations

"""Deterministic verifier-wrapper baseline for reality-check experiments."""

from typing import Any, Dict, Optional

from ..config import SpecModel
from ..runner.adapter_api import AgentRequest, AgentResponse
from ..runner.protocols import ConstraintEvaluator
from ..verifiers import canonicalize_smiles
from .base_adapter import BaseAdapter
from .corpus_search import CorpusSearchAdapter
from .local_mutation import LocalMutationAdapter

REJECT_SENTINEL = "not_a_smiles"


class WellEngineeredWrapperAdapter(BaseAdapter):
    """Strong non-LLM wrapper using public specs and verifier semantics.

    The adapter intentionally consumes only the public task/spec request. It is a
    reality-check baseline: if it solves most tasks, the release should be
    framed as an evaluation contract and audit harness rather than as evidence
    that raw LLM agents need this exact benchmark.
    """

    name = "well_engineered_wrapper"
    track = "wrapper_guarded"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._corpus = CorpusSearchAdapter(seed=seed)
        self._local = LocalMutationAdapter(seed=seed)

    def step(self, req: AgentRequest) -> AgentResponse:
        task = req.get("task") or {}
        visible_name = str(task.get("visible_task_name") or "evaluation_task")
        protocol = str(task.get("protocol") or "L1")
        round_index = int(req.get("round") or 1)
        interrupt = req.get("interrupt") or {}
        candidate = _public_input_smiles(task)
        failure = req.get("failure_vector") if isinstance(req.get("failure_vector"), dict) else None

        if protocol == "L3" and round_index == 1 and candidate and _has_verify_tool(req):
            response: AgentResponse = {
                "action": "tool_call",
                "name": "verify",
                "args": {"smiles": candidate},
                "p_hard_pass": 0.5,
            }
            if interrupt:
                response["interrupt_ack"] = _interrupt_ack(interrupt)
            return response

        if visible_name in {"candidate_audit", "boundary_audit", "representation_invariance"}:
            response = self._audit_candidate(req=req, candidate=candidate, failure=failure)
        elif visible_name == "feasibility_check":
            response = self._feasibility_check(req=req)
        else:
            response = self._construct_or_repair(req=req, candidate=candidate, failure=failure)

        if interrupt:
            response["interrupt_ack"] = _interrupt_ack(interrupt)
        return response

    def _audit_candidate(
        self,
        *,
        req: AgentRequest,
        candidate: Optional[str],
        failure: Optional[Dict[str, Any]],
    ) -> AgentResponse:
        if not candidate:
            return {"action": "propose", "smiles": REJECT_SENTINEL, "p_hard_pass": 0.01}
        feedback_pass = _feedback_hard_pass(failure)
        if feedback_pass is False:
            return {"action": "propose", "smiles": candidate, "p_hard_pass": 0.01}
        if feedback_pass is True:
            return {"action": "propose", "smiles": candidate, "p_hard_pass": 0.99}

        result = _evaluate_public(req, candidate)
        if result is None:
            return {"action": "propose", "smiles": candidate, "p_hard_pass": 0.5}
        return {
            "action": "propose",
            "smiles": candidate if result.hard_pass else candidate,
            "p_hard_pass": 0.99 if result.hard_pass else 0.01,
        }

    def _feasibility_check(self, *, req: AgentRequest) -> AgentResponse:
        spec_payload = req.get("spec")
        if isinstance(spec_payload, dict) and _has_property_bounds_contradiction(spec_payload):
            return {
                "action": "abstain",
                "reason": "Visible hard property bounds are contradictory.",
                "p_hard_pass": 0.01,
            }
        # If no explicit contradiction is visible, try normal construction.
        return self._construct_or_repair(req=req, candidate=None, failure=None)

    def _construct_or_repair(
        self,
        *,
        req: AgentRequest,
        candidate: Optional[str],
        failure: Optional[Dict[str, Any]],
    ) -> AgentResponse:
        if candidate and _feedback_hard_pass(failure) is False:
            # Do not accept the same candidate after verifier feedback says it
            # fails; switch to search/retrieval.
            candidate = None
        response = self._corpus.step(req)
        smiles = response.get("smiles")
        if isinstance(smiles, str) and _passes_public(req, smiles):
            response["p_hard_pass"] = 0.95
            return response
        local_response = self._local.step(req)
        local_smiles = local_response.get("smiles")
        if isinstance(local_smiles, str) and _passes_public(req, local_smiles):
            local_response["p_hard_pass"] = 0.9
            return local_response
        if candidate and _passes_public(req, candidate):
            return {"action": "propose", "smiles": candidate, "p_hard_pass": 0.9}
        return {"action": "abstain", "reason": "No passing candidate found by wrapper search.", "p_hard_pass": 0.05}


def _has_verify_tool(req: AgentRequest) -> bool:
    tools = req.get("tools") or []
    return any(isinstance(tool, dict) and tool.get("name") == "verify" for tool in tools)


def _public_input_smiles(task: Dict[str, Any]) -> Optional[str]:
    input_payload = task.get("input") if isinstance(task.get("input"), dict) else {}
    for key in ("smiles", "candidate_smiles"):
        value = input_payload.get(key)
        if isinstance(value, str) and value.strip():
            canonical = canonicalize_smiles(value.strip())
            return canonical or value.strip()
    return None


def _feedback_hard_pass(failure: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not isinstance(failure, dict):
        return None
    hard_fails = failure.get("hard_fails")
    if isinstance(hard_fails, list):
        return len(hard_fails) == 0
    hard_fail_ids = failure.get("hard_fail_ids")
    if isinstance(hard_fail_ids, list):
        return len(hard_fail_ids) == 0
    return None


def _evaluate_public(req: AgentRequest, smiles: str):
    spec_payload = req.get("spec")
    if not isinstance(spec_payload, dict):
        return None
    try:
        spec = SpecModel.model_validate(spec_payload)
        task = req.get("task") or {}
        input_smiles = _public_input_smiles(task)
        evaluator = ConstraintEvaluator(spec, input_smiles=input_smiles)
        return evaluator.evaluate(smiles)
    except Exception:
        return None


def _passes_public(req: AgentRequest, smiles: str) -> bool:
    result = _evaluate_public(req, smiles)
    return bool(result and result.hard_pass)


def _has_property_bounds_contradiction(spec_payload: Dict[str, Any]) -> bool:
    bounds: dict[str, dict[str, float]] = {}
    for constraint in spec_payload.get("constraints", []):
        if not isinstance(constraint, dict):
            continue
        if constraint.get("severity") not in {None, "hard"}:
            continue
        if constraint.get("check") != "property_bounds":
            continue
        params = constraint.get("params")
        if not isinstance(params, dict):
            continue
        for prop, raw in params.items():
            if not isinstance(raw, dict):
                continue
            lower = raw.get("min")
            upper = raw.get("max")
            entry = bounds.setdefault(str(prop), {"min": float("-inf"), "max": float("inf")})
            try:
                if lower is not None:
                    entry["min"] = max(entry["min"], float(lower))
                if upper is not None:
                    entry["max"] = min(entry["max"], float(upper))
            except (TypeError, ValueError):
                continue
            if entry["min"] > entry["max"]:
                return True
    return False


def _interrupt_ack(interrupt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "acknowledged": True,
        "restate_goal": True,
        "report_state": True,
        "resume_token": interrupt.get("resume_token"),
    }


__all__ = ["WellEngineeredWrapperAdapter"]
