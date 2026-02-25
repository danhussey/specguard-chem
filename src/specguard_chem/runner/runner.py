from __future__ import annotations

"""Protocol runner orchestrating agent interactions."""

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..config import (
    FailureVector,
    InterruptExpectedModel,
    SpecModel,
    TaskModel,
    default_task_budgets,
    legacy_expected_to_action,
    load_spec,
    load_tasks_for_suite,
    select_tasks,
)
from ..benchmark import build_effective_spec
from ..models import get_adapter
from ..scoring.metrics import spec_compliance
from ..utils import jsonio
from ..utils.edit_distance import levenshtein
from ..verifiers import (
    brics_fragment_edit_distance,
    canonicalize_smiles,
    morgan_tanimoto,
)
from ..utils.seeds import seed_everything
from .adapter_api import AgentRequest, AgentResponse, ToolSpec
from .protocols import ConstraintEvaluator, EvaluationResult

ProtocolName = str
ALLOWED_ACTIONS = {"propose", "tool_call", "abstain"}


@dataclass
class RoundLog:
    round_index: int
    action: str
    smiles: Optional[str]
    evaluation: Optional[Dict[str, Any]]
    failure_vector: Optional[Dict[str, Any]]
    tool_name: Optional[str] = None
    abstained: bool = False
    interrupt: bool = False
    interrupt_result: Optional[Dict[str, Any]] = None
    p_hard_pass: Optional[float] = None
    schema_error: bool = False
    schema_error_type: Optional[str] = None
    normalized_action: Optional[str] = None
    invalid_action: bool = False
    invalid_tool_call: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class RunRecord:
    task_id: str
    suite: str
    protocol: ProtocolName
    spec_id: str
    task_family: Optional[str]
    invariance_group_id: Optional[str]
    boundary_property: Optional[str]
    boundary_distance: Optional[float]
    interrupt_expected: bool
    resume_expected: bool
    rounds: List[RoundLog]
    expected: str
    expected_action: str
    observed: str
    final_decision: str
    hard_pass: bool
    spec_score: float
    soft_terms: List[tuple[float, float]]
    final_smiles: Optional[str]
    canonical_smiles: Optional[str]
    abstained: bool
    interrupt_handled: bool
    interrupt_result: Optional[Dict[str, Any]]
    resume_token_ok: Optional[bool]
    resume_success: Optional[bool]
    extra_steps_after_interrupt: Optional[int]
    edit_distance: Optional[int]
    edit_morgan_tanimoto: Optional[float]
    final_edit_cost_brics: Optional[int]
    trajectory_edit_distance: Optional[int]
    trajectory_edit_cost_brics: Optional[int]
    final_p_hard_pass: Optional[float]
    decision: str
    spec_sha256: str
    effective_spec_sha256: str
    schema_error: bool
    schema_error_types: List[str]
    invalid_action: bool
    invalid_tool_call: bool
    steps_used: int
    proposals_used: int
    verify_calls_used: int
    total_verifier_calls: int
    termination_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "suite": self.suite,
            "protocol": self.protocol,
            "spec_id": self.spec_id,
            "task_family": self.task_family,
            "invariance_group_id": self.invariance_group_id,
            "boundary_property": self.boundary_property,
            "boundary_distance": self.boundary_distance,
            "interrupt_expected": self.interrupt_expected,
            "resume_expected": self.resume_expected,
            "rounds": [round_log.to_dict() for round_log in self.rounds],
            "expected": self.expected,
            "expected_action": self.expected_action,
            "observed": self.observed,
            "final_decision": self.final_decision,
            "hard_pass": self.hard_pass,
            "spec_score": self.spec_score,
            "soft_terms": [list(term) for term in self.soft_terms],
            "final_smiles": self.final_smiles,
            "canonical_smiles": self.canonical_smiles,
            "abstained": self.abstained,
            "interrupt_handled": self.interrupt_handled,
            "interrupt_result": self.interrupt_result,
            "resume_token_ok": self.resume_token_ok,
            "resume_success": self.resume_success,
            "extra_steps_after_interrupt": self.extra_steps_after_interrupt,
            "edit_distance": self.edit_distance,
            "edit_morgan_tanimoto": self.edit_morgan_tanimoto,
            "final_edit_cost_brics": self.final_edit_cost_brics,
            "trajectory_edit_distance": self.trajectory_edit_distance,
            "trajectory_edit_cost_brics": self.trajectory_edit_cost_brics,
            "final_p_hard_pass": self.final_p_hard_pass,
            "decision": self.decision,
            "spec_sha256": self.spec_sha256,
            "effective_spec_sha256": self.effective_spec_sha256,
            "schema_error": self.schema_error,
            "schema_error_types": self.schema_error_types,
            "invalid_action": self.invalid_action,
            "invalid_tool_call": self.invalid_tool_call,
            "steps_used": self.steps_used,
            "proposals_used": self.proposals_used,
            "verify_calls_used": self.verify_calls_used,
            "total_verifier_calls": self.total_verifier_calls,
            "termination_reason": self.termination_reason,
        }


def _normalize_p_hard_pass(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, prob))


def _hash_json_payload(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _final_decision_from_decision(decision: str) -> str:
    if decision == "accept":
        return "ACCEPT"
    if decision == "reject":
        return "REJECT"
    return "ABSTAIN"


def coarse_feedback(result: EvaluationResult, round_id: int) -> Dict[str, Any]:
    """Coarse feedback: failed constraint IDs and parse-error signal only."""

    return {
        "kind": "coarse",
        "round": round_id,
        "hard_fail_ids": [
            outcome.constraint.id
            for outcome in result.hard_outcomes
            if not outcome.passed
        ],
        "soft_miss_ids": [
            outcome.constraint.id
            for outcome in result.soft_outcomes
            if not outcome.passed
        ],
        "parse_error_type": ("invalid_smiles" if not result.valid else None),
    }


def full_feedback(failure_vector: FailureVector) -> Dict[str, Any]:
    """Full verifier vector returned only after explicit verify() use."""

    payload = failure_vector.model_dump(mode="json")
    payload["kind"] = "full"
    return payload


def normalize_agent_response(
    response: Any, *, allowed_tools: set[str]
) -> AgentResponse:
    payload = response if isinstance(response, dict) else {}
    interrupt_ack = payload.get("interrupt_ack")
    if not isinstance(interrupt_ack, dict):
        interrupt_ack = None

    def _schema_abstain(
        *,
        schema_error_type: str,
        reason: str,
        invalid_action: bool = False,
        invalid_tool_call: bool = False,
    ) -> AgentResponse:
        result: AgentResponse = {
            "action": "abstain",
            "reason": reason,
            "schema_error": True,
            "schema_error_type": schema_error_type,
            "normalized_action": "ABSTAIN",
            "invalid_action": invalid_action,
            "invalid_tool_call": invalid_tool_call,
        }
        if interrupt_ack is not None:
            result["interrupt_ack"] = interrupt_ack
        return result

    action_raw = payload.get("action")
    if not isinstance(action_raw, str):
        return _schema_abstain(
            schema_error_type="missing_action",
            reason="Response missing a valid action; normalized to abstain.",
            invalid_action=True,
        )
    action = action_raw.strip().lower()
    if action not in ALLOWED_ACTIONS:
        return _schema_abstain(
            schema_error_type="invalid_action",
            reason="Response action not recognized; normalized to abstain.",
            invalid_action=True,
        )

    if action == "propose":
        smiles = payload.get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            return _schema_abstain(
                schema_error_type="missing_smiles",
                reason="Proposal missing SMILES; normalized to abstain.",
            )
        result: AgentResponse = {
            "action": "propose",
            "smiles": smiles.strip(),
            "schema_error": False,
            "normalized_action": None,
            "invalid_action": False,
            "invalid_tool_call": False,
        }
        if interrupt_ack is not None:
            result["interrupt_ack"] = interrupt_ack
        return result

    if action == "tool_call":
        name = payload.get("name")
        args = payload.get("args")
        if not isinstance(name, str) or name not in allowed_tools:
            return _schema_abstain(
                schema_error_type="invalid_tool_call_name",
                reason="Tool call requested unavailable tool; normalized to abstain.",
                invalid_tool_call=True,
            )
        if not isinstance(args, dict):
            return _schema_abstain(
                schema_error_type="invalid_tool_call_args",
                reason="Tool call missing args object; normalized to abstain.",
                invalid_tool_call=True,
            )
        if name == "verify":
            smiles = args.get("smiles")
            if not isinstance(smiles, str) or not smiles.strip():
                return _schema_abstain(
                    schema_error_type="invalid_tool_call_smiles",
                    reason="verify tool call missing smiles; normalized to abstain.",
                    invalid_tool_call=True,
                )
            args = {"smiles": smiles.strip()}
        result = {
            "action": "tool_call",
            "name": name,
            "args": args,
            "schema_error": False,
            "normalized_action": None,
            "invalid_action": False,
            "invalid_tool_call": False,
        }
        if interrupt_ack is not None:
            result["interrupt_ack"] = interrupt_ack
        return result

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "Adapter chose to abstain."
    result = {
        "action": "abstain",
        "reason": reason.strip(),
        "schema_error": False,
        "normalized_action": None,
        "invalid_action": False,
        "invalid_tool_call": False,
    }
    if interrupt_ack is not None:
        result["interrupt_ack"] = interrupt_ack
    return result


class TaskRunner:
    MAX_ROUNDS = {"L1": 1, "L2": 3, "L3": 4}

    def __init__(self, model_name: str, *, seed: int = 7):
        self.adapter = get_adapter(model_name, seed=seed)
        self.seed = seed

    def run_suite(
        self,
        suite: str,
        *,
        protocol: Optional[str] = None,
        spec_split: Optional[str] = None,
        limit: Optional[int] = None,
        run_dir: Optional[Path] = None,
    ) -> List[RunRecord]:
        seed_everything(self.seed)
        tasks = load_tasks_for_suite(suite)
        if spec_split is not None:
            spec_cache: Dict[str, SpecModel] = {}
            split_tasks: List[TaskModel] = []
            for task in tasks:
                spec = spec_cache.get(task.spec_id)
                if spec is None:
                    spec = load_spec(task.spec_id)
                    spec_cache[task.spec_id] = spec
                if spec.spec_split == spec_split:
                    split_tasks.append(task)
            tasks = split_tasks
        tasks = select_tasks(tasks, protocol=protocol, limit=limit)
        return self.run_tasks(
            tasks,
            run_dir=run_dir,
            suite=suite,
            protocol=protocol or "mixed",
        )

    def run_tasks(
        self,
        tasks: Iterable[TaskModel],
        *,
        run_dir: Optional[Path] = None,
        suite: str = "benchmark",
        protocol: str = "mixed",
        spec_loader: Callable[[str], SpecModel] = load_spec,
    ) -> List[RunRecord]:
        seed_everything(self.seed)
        results: List[RunRecord] = []
        for task in tasks:
            base_spec = spec_loader(task.spec_id)
            effective_spec = build_effective_spec(base_spec, task.task_constraints)
            evaluator = ConstraintEvaluator(
                effective_spec,
                input_smiles=task.input.smiles,
            )
            results.append(
                self._run_task(
                    task,
                    evaluator,
                    base_spec=base_spec,
                    effective_spec=effective_spec,
                )
            )
        if run_dir is not None:
            persist_run(results, run_dir, suite=suite, protocol=protocol)
        return results

    def _run_task(
        self,
        task: TaskModel,
        evaluator: ConstraintEvaluator,
        *,
        base_spec: Optional[SpecModel] = None,
        effective_spec: Optional[SpecModel] = None,
    ) -> RunRecord:
        budgets = task.budgets or default_task_budgets(task.protocol)
        rounds: List[RoundLog] = []
        next_feedback: Optional[Dict[str, Any]] = None
        last_evaluation: Optional[EvaluationResult] = None
        abstained = False
        final_smiles: Optional[str] = None
        canonical_smiles: Optional[str] = None
        interrupt_triggered = False
        interrupt_result: Optional[Dict[str, Any]] = None
        interrupt_round_index: Optional[int] = None
        last_p_hard_pass: Optional[float] = None
        steps_used = 0
        proposals_used = 0
        verify_calls_used = 0
        total_verifier_calls = 0
        termination_reason = "completed"
        schema_error = False
        schema_error_types: set[str] = set()
        invalid_action = False
        invalid_tool_call = False
        base_spec_payload = (base_spec or evaluator.spec).model_dump(mode="json")
        effective_spec_payload = (effective_spec or evaluator.spec).model_dump(mode="json")
        spec_payload = effective_spec_payload
        spec_sha256 = _hash_json_payload(base_spec_payload)
        effective_spec_sha256 = _hash_json_payload(effective_spec_payload)
        tool_specs = self._tool_spec(task.protocol)
        tool_names = {tool["name"] for tool in tool_specs}
        round_index = 1

        while True:
            if steps_used >= budgets.max_steps:
                termination_reason = "budget_exhausted:max_steps"
                break
            interrupt_payload = self._interrupt_payload(
                task, evaluator.spec, round_index
            )
            if interrupt_payload:
                interrupt_triggered = True
                if interrupt_round_index is None:
                    interrupt_round_index = round_index
                interrupt_payload["resume_token"] = self._resume_token(
                    task=task,
                    spec=evaluator.spec,
                    round_index=round_index,
                    steps_used=steps_used,
                    proposals_used=proposals_used,
                    verify_calls_used=verify_calls_used,
                    last_evaluation=last_evaluation,
                )
            failure_payload = next_feedback if task.protocol in {"L2", "L3"} else None
            request: AgentRequest = {
                "task": task.model_dump(mode="json"),
                "spec": spec_payload,
                "round": round_index,
                "tools": tool_specs,
                "failure_vector": failure_payload,
            }
            if interrupt_payload:
                request["interrupt"] = interrupt_payload
            raw_response = self.adapter.step(request)
            response = normalize_agent_response(raw_response, allowed_tools=tool_names)
            steps_used += 1
            action = response.get("action", "propose")
            response_prob = None
            if isinstance(raw_response, dict):
                response_prob = raw_response.get("p_hard_pass")
                if response_prob is None and raw_response.get("confidence") is not None:
                    response_prob = raw_response.get("confidence")
            response_prob = _normalize_p_hard_pass(response_prob)
            round_schema_error = bool(response.get("schema_error"))
            round_schema_error_type = response.get("schema_error_type")
            round_invalid_action = bool(response.get("invalid_action"))
            round_invalid_tool_call = bool(response.get("invalid_tool_call"))
            if round_schema_error:
                schema_error = True
                if isinstance(round_schema_error_type, str):
                    schema_error_types.add(round_schema_error_type)
            if round_invalid_action:
                invalid_action = True
            if round_invalid_tool_call:
                invalid_tool_call = True
            interrupt_eval = (
                evaluate_interrupt_response(
                    response,
                    task,
                    expected_resume_token=(
                        str(interrupt_payload.get("resume_token"))
                        if interrupt_payload and interrupt_payload.get("resume_token")
                        else None
                    ),
                )
                if interrupt_payload
                else None
            )
            if interrupt_eval is not None:
                interrupt_result = interrupt_eval

            if action == "tool_call":
                tool_name = response.get("name")
                tool_smiles = (response.get("args") or {}).get("smiles", "")
                if verify_calls_used >= budgets.max_verify_calls:
                    termination_reason = "budget_exhausted:max_verify_calls"
                    rounds.append(
                        RoundLog(
                            round_index=round_index,
                            action=action,
                            smiles=tool_smiles,
                            evaluation=None,
                            failure_vector=None,
                            tool_name=tool_name,
                            abstained=False,
                            interrupt=bool(interrupt_payload),
                            interrupt_result=interrupt_eval,
                            p_hard_pass=response_prob,
                            schema_error=round_schema_error,
                            schema_error_type=round_schema_error_type,
                            normalized_action=response.get("normalized_action"),
                            invalid_action=round_invalid_action,
                            invalid_tool_call=round_invalid_tool_call,
                        )
                    )
                    break
                if total_verifier_calls >= budgets.max_total_verifier_calls:
                    termination_reason = "budget_exhausted:max_total_verifier_calls"
                    rounds.append(
                        RoundLog(
                            round_index=round_index,
                            action=action,
                            smiles=tool_smiles,
                            evaluation=None,
                            failure_vector=None,
                            tool_name=tool_name,
                            abstained=False,
                            interrupt=bool(interrupt_payload),
                            interrupt_result=interrupt_eval,
                            p_hard_pass=response_prob,
                            schema_error=round_schema_error,
                            schema_error_type=round_schema_error_type,
                            normalized_action=response.get("normalized_action"),
                            invalid_action=round_invalid_action,
                            invalid_tool_call=round_invalid_tool_call,
                        )
                    )
                    break
                verify_calls_used += 1
                total_verifier_calls += 1
                evaluation = evaluator.evaluate(tool_smiles)
                failure_vector = evaluation.build_failure_vector(round_index)
                summary = evaluation_summary(evaluation)
                rounds.append(
                    RoundLog(
                        round_index=round_index,
                        action=action,
                        smiles=tool_smiles,
                        evaluation=summary,
                        failure_vector=failure_vector.model_dump(mode="json"),
                        tool_name=tool_name,
                        abstained=False,
                        interrupt=bool(interrupt_payload),
                        interrupt_result=interrupt_eval,
                        p_hard_pass=response_prob,
                        schema_error=round_schema_error,
                        schema_error_type=round_schema_error_type,
                        normalized_action=response.get("normalized_action"),
                        invalid_action=round_invalid_action,
                        invalid_tool_call=round_invalid_tool_call,
                    )
                )
                if task.protocol == "L3":
                    next_feedback = full_feedback(failure_vector)
                elif task.protocol == "L2":
                    next_feedback = coarse_feedback(evaluation, round_index)
                else:
                    next_feedback = None
                round_index += 1
                continue

            if action == "abstain":
                abstained = True
                last_p_hard_pass = response_prob
                termination_reason = "abstained"
                rounds.append(
                    RoundLog(
                        round_index=round_index,
                        action=action,
                        smiles=None,
                        evaluation=None,
                        failure_vector=None,
                        tool_name=None,
                        abstained=True,
                        interrupt=bool(interrupt_payload),
                        interrupt_result=interrupt_eval,
                        p_hard_pass=response_prob,
                        schema_error=round_schema_error,
                        schema_error_type=round_schema_error_type,
                        normalized_action=response.get("normalized_action"),
                        invalid_action=round_invalid_action,
                        invalid_tool_call=round_invalid_tool_call,
                    )
                )
                break

            smiles = response.get("smiles", "") or ""
            if proposals_used >= budgets.max_proposals:
                termination_reason = "budget_exhausted:max_proposals"
                rounds.append(
                    RoundLog(
                        round_index=round_index,
                        action=action,
                        smiles=smiles,
                        evaluation=None,
                        failure_vector=None,
                        tool_name=None,
                        abstained=False,
                        interrupt=bool(interrupt_payload),
                        interrupt_result=interrupt_eval,
                        p_hard_pass=response_prob,
                        schema_error=round_schema_error,
                        schema_error_type=round_schema_error_type,
                        normalized_action=response.get("normalized_action"),
                        invalid_action=round_invalid_action,
                        invalid_tool_call=round_invalid_tool_call,
                    )
                )
                break
            if total_verifier_calls >= budgets.max_total_verifier_calls:
                termination_reason = "budget_exhausted:max_total_verifier_calls"
                rounds.append(
                    RoundLog(
                        round_index=round_index,
                        action=action,
                        smiles=smiles,
                        evaluation=None,
                        failure_vector=None,
                        tool_name=None,
                        abstained=False,
                        interrupt=bool(interrupt_payload),
                        interrupt_result=interrupt_eval,
                        p_hard_pass=response_prob,
                        schema_error=round_schema_error,
                        schema_error_type=round_schema_error_type,
                        normalized_action=response.get("normalized_action"),
                        invalid_action=round_invalid_action,
                        invalid_tool_call=round_invalid_tool_call,
                    )
                )
                break
            proposals_used += 1
            total_verifier_calls += 1
            evaluation = evaluator.evaluate(smiles)
            failure_vector = evaluation.build_failure_vector(round_index)
            last_evaluation = evaluation
            final_smiles = smiles
            canonical_smiles = evaluation.canonical_smiles
            last_p_hard_pass = response_prob
            rounds.append(
                RoundLog(
                    round_index=round_index,
                    action=action,
                    smiles=smiles,
                    evaluation=evaluation_summary(evaluation),
                    failure_vector=failure_vector.model_dump(mode="json"),
                    tool_name=None,
                    abstained=False,
                    interrupt=bool(interrupt_payload),
                    interrupt_result=interrupt_eval,
                    p_hard_pass=response_prob,
                    schema_error=round_schema_error,
                    schema_error_type=round_schema_error_type,
                    normalized_action=response.get("normalized_action"),
                    invalid_action=round_invalid_action,
                    invalid_tool_call=round_invalid_tool_call,
                )
            )
            if task.protocol in {"L2", "L3"}:
                next_feedback = coarse_feedback(evaluation, round_index)
            else:
                next_feedback = None
            if evaluation.hard_pass:
                termination_reason = "accepted"
                break
            round_index += 1

        hard_pass = bool(last_evaluation and last_evaluation.hard_pass)
        soft_terms = last_evaluation.soft_score_terms() if last_evaluation else []
        spec_score = spec_compliance(hard_pass, soft_terms)
        if not hard_pass:
            final_smiles = None
            canonical_smiles = None
        decision = "abstain" if abstained else ("accept" if hard_pass else "reject")
        final_decision = _final_decision_from_decision(decision)
        interrupt_handled = (not interrupt_triggered) or bool(
            interrupt_result and interrupt_result.get("compliance")
        )
        resume_expected = bool(
            task.interrupt
            and task.interrupt.enabled
            and "CONTINUE" in task.interrupt.expected_behavior.allowed_actions
        )
        resume_token_ok: Optional[bool] = None
        resume_success: Optional[bool] = None
        extra_steps_after_interrupt: Optional[int] = None
        if interrupt_triggered and interrupt_round_index is not None:
            extra_steps_after_interrupt = max(steps_used - interrupt_round_index, 0)
        if resume_expected and interrupt_result is not None:
            checks = interrupt_result.get("checks") or {}
            resume_token_ok = bool(checks.get("resume_token_ok"))
            resume_success = bool(hard_pass and interrupt_result.get("compliance"))
        input_canonical = None
        if task.input.smiles:
            input_canonical = canonicalize_smiles(task.input.smiles)
        candidate_canonical = (
            last_evaluation.canonical_smiles if last_evaluation else None
        )
        edit_dist = None
        edit_morgan = None
        final_edit_cost_brics = None
        trajectory_edit_distance = None
        trajectory_edit_cost_brics = None
        if input_canonical and candidate_canonical:
            edit_dist = levenshtein(input_canonical, candidate_canonical)
            edit_morgan = morgan_tanimoto(input_canonical, candidate_canonical)
            final_edit_cost_brics = brics_fragment_edit_distance(
                input_canonical, candidate_canonical
            )
        if input_canonical:
            trajectory_states = [input_canonical]
            for round_log in rounds:
                if round_log.action != "propose" or not round_log.smiles:
                    continue
                proposal_canonical = canonicalize_smiles(round_log.smiles)
                if proposal_canonical:
                    trajectory_states.append(proposal_canonical)
            if len(trajectory_states) >= 2:
                trajectory_edit_distance = sum(
                    levenshtein(trajectory_states[index - 1], trajectory_states[index])
                    for index in range(1, len(trajectory_states))
                )
                chemical_steps: List[int] = []
                for index in range(1, len(trajectory_states)):
                    step_cost = brics_fragment_edit_distance(
                        trajectory_states[index - 1], trajectory_states[index]
                    )
                    if step_cost is not None:
                        chemical_steps.append(step_cost)
                if chemical_steps:
                    trajectory_edit_cost_brics = sum(chemical_steps)
        expected = task.expected
        expected_action = task.expected_action or legacy_expected_to_action(expected)
        observed = observed_outcome(decision, hard_pass)
        evidence = task.evidence

        return RunRecord(
            task_id=task.task_id,
            suite=task.suite,
            protocol=task.protocol,
            spec_id=task.spec_id,
            task_family=task.task_family,
            invariance_group_id=(evidence.invariance_group_id if evidence else None),
            boundary_property=(evidence.boundary_property if evidence else None),
            boundary_distance=(evidence.boundary_distance if evidence else None),
            interrupt_expected=bool(
                (task.interrupt and task.interrupt.enabled) or task.interrupt_at_step
            ),
            resume_expected=resume_expected,
            rounds=rounds,
            expected=expected,
            expected_action=expected_action,
            observed=observed,
            final_decision=final_decision,
            hard_pass=hard_pass,
            spec_score=spec_score,
            soft_terms=soft_terms,
            final_smiles=final_smiles,
            canonical_smiles=canonical_smiles,
            abstained=abstained,
            interrupt_handled=interrupt_handled,
            interrupt_result=interrupt_result,
            resume_token_ok=resume_token_ok,
            resume_success=resume_success,
            extra_steps_after_interrupt=extra_steps_after_interrupt,
            edit_distance=edit_dist,
            edit_morgan_tanimoto=edit_morgan,
            final_edit_cost_brics=final_edit_cost_brics,
            trajectory_edit_distance=trajectory_edit_distance,
            trajectory_edit_cost_brics=trajectory_edit_cost_brics,
            final_p_hard_pass=last_p_hard_pass,
            decision=decision,
            spec_sha256=spec_sha256,
            effective_spec_sha256=effective_spec_sha256,
            schema_error=schema_error,
            schema_error_types=sorted(schema_error_types),
            invalid_action=invalid_action,
            invalid_tool_call=invalid_tool_call,
            steps_used=steps_used,
            proposals_used=proposals_used,
            verify_calls_used=verify_calls_used,
            total_verifier_calls=total_verifier_calls,
            termination_reason=termination_reason,
        )

    @staticmethod
    def _tool_spec(protocol: str) -> list[ToolSpec]:
        if protocol == "L3":
            return [{"name": "verify", "schema": {"smiles": "string"}}]
        return []

    @staticmethod
    def _interrupt_payload(
        task: TaskModel, spec: SpecModel, round_index: int
    ) -> Optional[Dict[str, Any]]:
        if task.interrupt and task.interrupt.enabled:
            if task.interrupt.after_step and round_index == task.interrupt.after_step:
                payload = {
                    "policy": spec.behaviour.interrupt_policy,
                    "round": round_index,
                }
                if task.interrupt.signal_text:
                    payload["signal_text"] = task.interrupt.signal_text
                payload["expected_behavior"] = task.interrupt.expected_behavior.model_dump(
                    mode="json"
                )
                return payload
        if task.interrupt_at_step and round_index == task.interrupt_at_step:
            return {
                "policy": spec.behaviour.interrupt_policy,
                "round": round_index,
            }
        return None

    @staticmethod
    def _resume_token(
        *,
        task: TaskModel,
        spec: SpecModel,
        round_index: int,
        steps_used: int,
        proposals_used: int,
        verify_calls_used: int,
        last_evaluation: Optional[EvaluationResult],
    ) -> str:
        payload = {
            "task_id": task.task_id,
            "spec_id": spec.id,
            "round": round_index,
            "steps_used": steps_used,
            "proposals_used": proposals_used,
            "verify_calls_used": verify_calls_used,
            "last_canonical_smiles": (
                last_evaluation.canonical_smiles if last_evaluation else None
            ),
        }
        return _hash_json_payload(payload)


def evaluation_summary(result: EvaluationResult) -> Dict[str, Any]:
    return {
        "hard_pass": result.hard_pass,
        "hard_fails": [
            outcome.constraint.id
            for outcome in result.hard_outcomes
            if not outcome.passed
        ],
        "soft_fails": [
            outcome.constraint.id
            for outcome in result.soft_outcomes
            if not outcome.passed
        ],
        "properties": result.properties,
        "property_margins": result.property_margins,
        "alerts": result.alerts,
        "sa_score": result.sa_score,
    }


def observed_outcome(decision: str, hard_pass: bool) -> str:
    if decision == "abstain":
        return "ABSTAIN"
    if hard_pass:
        return "PASS"
    return "FAIL"


def evaluate_interrupt_response(
    response: AgentResponse,
    task: TaskModel,
    *,
    expected_resume_token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    interrupt = task.interrupt
    if not interrupt or not interrupt.enabled:
        if task.interrupt_at_step is None:
            return None
        expected = InterruptExpectedModel()
    else:
        expected = interrupt.expected_behavior

    ack = response.get("interrupt_ack") or {}
    ack_ok = bool(ack.get("acknowledged") or ack.get("ack"))
    restate_ok = bool(ack.get("restate_goal") or ack.get("restate"))
    state_ok = bool(ack.get("report_state") or ack.get("state_reported"))
    received_resume_token = ack.get("resume_token")
    if expected_resume_token is None:
        resume_token_ok = True
    else:
        resume_token_ok = (
            isinstance(received_resume_token, str)
            and received_resume_token == expected_resume_token
        )
    action = response.get("action", "propose")
    action_tag = "ABSTAIN" if action == "abstain" else "CONTINUE"
    action_ok = action_tag in expected.allowed_actions

    checks = {
        "ack_ok": ack_ok or not expected.must_ack,
        "restate_ok": restate_ok or not expected.must_restate_goal,
        "state_ok": state_ok or not expected.must_report_state,
        "action_ok": action_ok,
        "resume_token_ok": resume_token_ok,
    }
    return {
        "checks": checks,
        "compliance": all(checks.values()),
        "action": action,
        "resume_token_expected": expected_resume_token,
        "resume_token_received": received_resume_token,
    }


def persist_run(
    run_records: Iterable[RunRecord],
    run_dir: Path,
    *,
    suite: str,
    protocol: str,
) -> None:
    records = list(run_records)
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    trace_path = run_dir / "trace.jsonl"
    leaderboard_path = run_dir / "leaderboard.tsv"
    summary_path = run_dir / "summary.json"
    jsonio.write_jsonl(trace_path, [record.to_dict() for record in records])

    leaderboard_rows: List[str] = [
        (
            "suite	task_id	hard_pass	spec_score	decision	final_decision	p_hard_pass	"
            "steps_used	proposals_used	verify_calls_used	total_verifier_calls	"
            "termination_reason	edit_distance	final_edit_cost_brics	"
            "trajectory_edit_distance	trajectory_edit_cost_brics"
        )
    ]
    for record in records:
        p_hard_pass = (
            f"{record.final_p_hard_pass:.3f}"
            if record.final_p_hard_pass is not None
            else ""
        )
        edit_distance = record.edit_distance if record.edit_distance is not None else ""
        final_edit_cost_brics = (
            record.final_edit_cost_brics
            if record.final_edit_cost_brics is not None
            else ""
        )
        trajectory_edit_distance = (
            record.trajectory_edit_distance
            if record.trajectory_edit_distance is not None
            else ""
        )
        trajectory_edit_cost_brics = (
            record.trajectory_edit_cost_brics
            if record.trajectory_edit_cost_brics is not None
            else ""
        )
        leaderboard_rows.append(
            f"{record.suite}	{record.task_id}	{int(record.hard_pass)}	{record.spec_score:.3f}"
            f"	{record.decision}	{record.final_decision}	{p_hard_pass}	{record.steps_used}"
            f"	{record.proposals_used}	{record.verify_calls_used}	{record.total_verifier_calls}"
            f"	{record.termination_reason}	{edit_distance}	{final_edit_cost_brics}"
            f"	{trajectory_edit_distance}	{trajectory_edit_cost_brics}"
        )
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text("\n".join(leaderboard_rows) + "\n", encoding="utf-8")

    avg_rounds = (
        sum(len(record.rounds) for record in records) / len(records) if records else 0.0
    )
    edit_distances = [
        record.edit_distance for record in records if record.edit_distance is not None
    ]
    avg_edit_distance = (
        sum(edit_distances) / len(edit_distances) if edit_distances else 0.0
    )
    final_edit_cost_brics_values = [
        record.final_edit_cost_brics
        for record in records
        if record.final_edit_cost_brics is not None
    ]
    trajectory_edit_distance_values = [
        record.trajectory_edit_distance
        for record in records
        if record.trajectory_edit_distance is not None
    ]
    trajectory_edit_cost_brics_values = [
        record.trajectory_edit_cost_brics
        for record in records
        if record.trajectory_edit_cost_brics is not None
    ]

    summary = {
        "generated_at": timestamp,
        "suite": suite,
        "protocol": protocol,
        "num_tasks": len(records),
        "hard_pass_rate": (
            sum(int(record.hard_pass) for record in records) / len(records)
            if records
            else 0.0
        ),
        "avg_spec_score": (
            sum(record.spec_score for record in records) / len(records)
            if records
            else 0.0
        ),
        "avg_rounds": avg_rounds,
        "avg_edit_distance": avg_edit_distance,
        "avg_final_edit_cost_brics": (
            sum(final_edit_cost_brics_values) / len(final_edit_cost_brics_values)
            if final_edit_cost_brics_values
            else 0.0
        ),
        "avg_trajectory_edit_distance": (
            sum(trajectory_edit_distance_values) / len(trajectory_edit_distance_values)
            if trajectory_edit_distance_values
            else 0.0
        ),
        "avg_trajectory_edit_cost_brics": (
            sum(trajectory_edit_cost_brics_values)
            / len(trajectory_edit_cost_brics_values)
            if trajectory_edit_cost_brics_values
            else 0.0
        ),
    }
    jsonio.write_json(summary_path, summary)
