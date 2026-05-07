from __future__ import annotations

"""Deterministic corpus-search baseline adapter."""

import json
from typing import Dict, List, Optional

from ..config import SpecModel
from ..dataset.corpus import build_corpus_records
from ..runner.adapter_api import AgentRequest, AgentResponse
from ..runner.protocols import ConstraintEvaluator
from ..verifiers import canonicalize_smiles, morgan_tanimoto
from .base_adapter import BaseAdapter


class CorpusSearchAdapter(BaseAdapter):
    name = "corpus_search"
    track = "retrieval"

    def __init__(self, *, seed: int = 0, corpus_size: int = 1200) -> None:
        super().__init__(seed=seed)
        self.corpus_size = max(1, int(corpus_size))
        self._corpus = [
            str(item["canonical_smiles"])
            for item in build_corpus_records(
                seed=max(seed, 1) + 101,
                max_molecules=self.corpus_size,
                reaction_depth=2,
            )
        ]
        self._pass_cache: Dict[str, List[str]] = {}

    def step(self, req: AgentRequest) -> AgentResponse:
        task = req.get("task") or {}
        spec_payload = req.get("spec") or {}
        if not isinstance(spec_payload, dict):
            return {"action": "abstain", "reason": "Missing structured spec payload."}
        if req.get("interrupt"):
            interrupt = req.get("interrupt") or {}
            return {
                "action": "propose",
                "smiles": self._select_candidate(task=task, spec_payload=spec_payload),
                "p_hard_pass": 0.85,
                "interrupt_ack": {
                    "acknowledged": True,
                    "restate_goal": True,
                    "report_state": True,
                    "resume_token": interrupt.get("resume_token"),
                },
            }
        return {
            "action": "propose",
            "smiles": self._select_candidate(task=task, spec_payload=spec_payload),
            "p_hard_pass": 0.85,
        }

    def _select_candidate(self, *, task: dict, spec_payload: dict) -> str:
        spec = SpecModel.model_validate(spec_payload)
        input_smiles = (task.get("input") or {}).get("smiles")
        input_canonical = (
            canonicalize_smiles(input_smiles)
            if isinstance(input_smiles, str) and input_smiles
            else None
        )
        evaluator = ConstraintEvaluator(spec, input_smiles=input_smiles)
        family = str(task.get("visible_task_name") or task.get("task_family") or "")
        requires_input = _requires_input_context(spec_payload) or family.startswith("repair")
        spec_key = json.dumps(
            {
                "spec": spec_payload,
                "input_canonical": input_canonical if requires_input else None,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        passers = self._pass_cache.get(spec_key)
        if passers is None:
            passers = []
            for smiles in self._corpus:
                if evaluator.evaluate(smiles).hard_pass:
                    passers.append(smiles)
            passers.sort()
            self._pass_cache[spec_key] = passers
        if not passers:
            return "CC(=O)NC1=CC=CC=C1O"

        candidate_pool = list(passers)
        if isinstance(input_smiles, str) and input_smiles and family.startswith("repair"):
            if input_canonical:
                best_smiles = candidate_pool[0]
                best_score = -1.0
                for candidate in candidate_pool:
                    sim = morgan_tanimoto(input_canonical, candidate)
                    score = float(sim) if sim is not None else -1.0
                    if score > best_score:
                        best_score = score
                        best_smiles = candidate
                return best_smiles
        return candidate_pool[0]


def _requires_input_context(spec_payload: dict) -> bool:
    constraints = spec_payload.get("constraints")
    if not isinstance(constraints, list):
        return False
    input_dependent = {"similarity_min_to_input", "equivalent_to_input"}
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        if constraint.get("check") in input_dependent:
            return True
    return False
