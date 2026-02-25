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

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._corpus = [
            str(item["canonical_smiles"])
            for item in build_corpus_records(
                seed=max(seed, 1) + 101,
                max_molecules=1200,
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
        spec_key = json.dumps(
            {
                "spec": spec_payload,
                "input_canonical": input_canonical,
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

        evidence = task.get("evidence") or {}
        witness_smiles = evidence.get("feasible_witness_smiles")
        witness_canonical = (
            canonicalize_smiles(witness_smiles)
            if isinstance(witness_smiles, str) and witness_smiles
            else None
        )
        candidate_pool = [
            smiles for smiles in passers if not witness_canonical or smiles != witness_canonical
        ]
        if not candidate_pool:
            candidate_pool = list(passers)
        family = str(task.get("task_family") or "")
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
