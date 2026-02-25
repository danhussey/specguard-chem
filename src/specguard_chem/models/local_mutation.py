from __future__ import annotations

"""Deterministic local-mutation hill-climb baseline adapter."""

from dataclasses import dataclass
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from ..config import SpecModel
from ..runner.adapter_api import AgentRequest, AgentResponse
from ..runner.protocols import ConstraintEvaluator, EvaluationResult
from ..utils.edit_distance import levenshtein
from ..verifiers import canonicalize_smiles
from .base_adapter import BaseAdapter

REACTION_SMARTS: tuple[str, ...] = (
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]Cl",
    "[cH:1]>>[c:1]N",
    "[cH:1]>>[c:1]C(=O)N",
    "[CH3:1]>>[CH2:1]F",
    "[CH3:1]>>[CH2:1]Cl",
    "[CH3:1]>>[CH2:1]N",
    "[CH2:1]>>[CH:1](C)",
)

FALLBACK_PROPOSALS: tuple[str, ...] = (
    "CC(=O)NC1=CC=CC=C1O",
    "COc1ccccc1O",
    "CCOC(=O)N(CC)CCO",
)


@dataclass(frozen=True)
class CandidateScore:
    objective: float
    edit_cost: int
    smiles: str


def _reaction_objects() -> List[rdChemReactions.ChemicalReaction]:
    return [rdChemReactions.ReactionFromSmarts(smarts) for smarts in REACTION_SMARTS]


def _neighbors(smiles: str, reactions: List[rdChemReactions.ChemicalReaction]) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    candidates: set[str] = set()
    for reaction in reactions:
        try:
            product_sets = reaction.RunReactants((mol,))
        except Exception:
            continue
        for product_tuple in product_sets:
            if not product_tuple:
                continue
            candidate = canonicalize_smiles(Chem.MolToSmiles(product_tuple[0]))
            if candidate:
                candidates.add(candidate)
    return sorted(candidates)


def _objective(result: EvaluationResult) -> float:
    if result.hard_pass:
        return 0.0
    margin_penalty = sum(value for value in result.property_margins.values() if value < 0.0)
    hard_penalty = 0.0
    for outcome in result.hard_outcomes:
        if outcome.passed:
            continue
        if outcome.constraint.check != "property_bounds":
            hard_penalty -= 1.0
    return margin_penalty + hard_penalty


class LocalMutationAdapter(BaseAdapter):
    name = "local_mutation"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._reactions = _reaction_objects()

    def step(self, req: AgentRequest) -> AgentResponse:
        task = req.get("task") or {}
        spec_payload = req.get("spec") or {}
        if not isinstance(spec_payload, dict):
            return {"action": "abstain", "reason": "Missing structured spec payload."}
        spec = SpecModel.model_validate(spec_payload)
        input_smiles = (task.get("input") or {}).get("smiles")
        evaluator = ConstraintEvaluator(spec, input_smiles=input_smiles)

        interrupt = req.get("interrupt") or {}
        proposal = self._search(task=task, evaluator=evaluator, round_index=int(req.get("round", 1)))
        response: AgentResponse = {
            "action": "propose",
            "smiles": proposal,
            "p_hard_pass": 0.75,
        }
        if interrupt:
            response["interrupt_ack"] = {
                "acknowledged": True,
                "restate_goal": True,
                "report_state": True,
                "resume_token": interrupt.get("resume_token"),
            }
        return response

    def _search(self, *, task: dict, evaluator: ConstraintEvaluator, round_index: int) -> str:
        input_smiles = (task.get("input") or {}).get("smiles")
        start = (
            canonicalize_smiles(str(input_smiles))
            if isinstance(input_smiles, str) and input_smiles
            else None
        )
        current = start or FALLBACK_PROPOSALS[self.seed % len(FALLBACK_PROPOSALS)]
        current_eval = evaluator.evaluate(current)
        if current_eval.hard_pass:
            return current

        search_steps = max(1, min(round_index, 4))
        reference = start or current
        for _ in range(search_steps):
            if current_eval.hard_pass:
                break
            best = CandidateScore(
                objective=_objective(current_eval),
                edit_cost=levenshtein(reference, current),
                smiles=current,
            )
            best_eval = current_eval

            neighbors = _neighbors(current, self._reactions)
            for fallback in FALLBACK_PROPOSALS:
                fallback_canonical = canonicalize_smiles(fallback)
                if fallback_canonical:
                    neighbors.append(fallback_canonical)
            for candidate in sorted(set(neighbors)):
                evaluation = evaluator.evaluate(candidate)
                score = CandidateScore(
                    objective=_objective(evaluation),
                    edit_cost=levenshtein(reference, candidate),
                    smiles=candidate,
                )
                if self._better(score, best):
                    best = score
                    best_eval = evaluation

            if best.smiles == current:
                break
            current = best.smiles
            current_eval = best_eval

        return current

    @staticmethod
    def _better(candidate: CandidateScore, incumbent: CandidateScore) -> bool:
        if candidate.objective != incumbent.objective:
            return candidate.objective > incumbent.objective
        if candidate.edit_cost != incumbent.edit_cost:
            return candidate.edit_cost < incumbent.edit_cost
        return candidate.smiles < incumbent.smiles
