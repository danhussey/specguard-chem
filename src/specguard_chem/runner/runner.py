from __future__ import annotations

"""Protocol runner orchestrating agent interactions."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import (
    FailureVector,
    InterruptExpectedModel,
    SpecModel,
    TaskModel,
    load_spec,
    load_tasks_for_suite,
    select_tasks,
)
from ..models import get_adapter
from ..scoring.metrics import spec_compliance
from ..utils import jsonio
from ..utils.edit_distance import levenshtein
from ..verifiers import canonicalize_smiles, morgan_tanimoto
from ..utils.seeds import seed_everything
from .adapter_api import AgentRequest, AgentResponse, ToolSpec
from .protocols import ConstraintEvaluator, EvaluationResult

ProtocolName = str


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

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class RunRecord:
    task_id: str
    suite: str
    protocol: ProtocolName
    spec_id: str
    interrupt_expected: bool
    rounds: List[RoundLog]
    expected: str
    observed: str
    hard_pass: bool
    spec_score: float
    soft_terms: List[tuple[float, float]]
    final_smiles: Optional[str]
    canonical_smiles: Optional[str]
    abstained: bool
    interrupt_handled: bool
    interrupt_result: Optional[Dict[str, Any]]
    edit_distance: Optional[int]
    edit_morgan_tanimoto: Optional[float]
    final_p_hard_pass: Optional[float]
    decision: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "suite": self.suite,
            "protocol": self.protocol,
            "spec_id": self.spec_id,
            "interrupt_expected": self.interrupt_expected,
            "rounds": [round_log.to_dict() for round_log in self.rounds],
            "expected": self.expected,
            "observed": self.observed,
            "hard_pass": self.hard_pass,
            "spec_score": self.spec_score,
            "soft_terms": [list(term) for term in self.soft_terms],
            "final_smiles": self.final_smiles,
            "canonical_smiles": self.canonical_smiles,
            "abstained": self.abstained,
            "interrupt_handled": self.interrupt_handled,
            "interrupt_result": self.interrupt_result,
            "edit_distance": self.edit_distance,
            "edit_morgan_tanimoto": self.edit_morgan_tanimoto,
            "final_p_hard_pass": self.final_p_hard_pass,
            "decision": self.decision,
        }


def _normalize_p_hard_pass(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, prob))


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
        limit: Optional[int] = None,
        run_dir: Optional[Path] = None,
    ) -> List[RunRecord]:
        seed_everything(self.seed)
        tasks = load_tasks_for_suite(suite)
        tasks = select_tasks(tasks, protocol=protocol, limit=limit)
        results: List[RunRecord] = []
        for task in tasks:
            spec = load_spec(task.spec_id)
            evaluator = ConstraintEvaluator(spec)
            results.append(self._run_task(task, evaluator))
        if run_dir is not None:
            persist_run(results, run_dir, suite=suite, protocol=protocol or "mixed")
        return results

    def _run_task(self, task: TaskModel, evaluator: ConstraintEvaluator) -> RunRecord:
        max_rounds = self.MAX_ROUNDS.get(task.protocol, 1)
        rounds: List[RoundLog] = []
        last_failure_vector: Optional[FailureVector] = None
        last_evaluation: Optional[EvaluationResult] = None
        abstained = False
        final_smiles: Optional[str] = None
        canonical_smiles: Optional[str] = None
        interrupt_triggered = False
        interrupt_result: Optional[Dict[str, Any]] = None
        last_p_hard_pass: Optional[float] = None

        for round_index in range(1, max_rounds + 1):
            interrupt_payload = self._interrupt_payload(
                task, evaluator.spec, round_index
            )
            if interrupt_payload:
                interrupt_triggered = True
            failure_payload = None
            if (
                task.protocol in {"L2", "L3"}
                and last_failure_vector is not None
            ):
                failure_payload = last_failure_vector.model_dump(mode="json")
            request: AgentRequest = {
                "task": task.model_dump(mode="json"),
                "round": round_index,
                "tools": self._tool_spec(task.protocol),
                "failure_vector": failure_payload,
            }
            if interrupt_payload:
                request["interrupt"] = interrupt_payload
            response = self.adapter.step(request)
            action = response.get("action", "propose")
            response_prob = response.get("p_hard_pass")
            if response_prob is None and response.get("confidence") is not None:
                response_prob = response.get("confidence")
            response_prob = _normalize_p_hard_pass(response_prob)
            interrupt_eval = (
                evaluate_interrupt_response(response, task)
                if interrupt_payload
                else None
            )
            if interrupt_eval is not None:
                interrupt_result = interrupt_eval

            if action == "tool_call":
                tool_name = response.get("name")
                tool_smiles = (response.get("args") or {}).get("smiles", "")
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
                    )
                )
                if task.protocol in {"L2", "L3"}:
                    last_failure_vector = failure_vector
                continue

            if action == "abstain":
                abstained = True
                last_p_hard_pass = response_prob
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
                    )
                )
                break

            smiles = response.get("smiles", "") or ""
            evaluation = evaluator.evaluate(smiles)
            failure_vector = evaluation.build_failure_vector(round_index)
            if task.protocol in {"L2", "L3"}:
                last_failure_vector = failure_vector
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
                )
            )
            if evaluation.hard_pass:
                break

        hard_pass = bool(last_evaluation and last_evaluation.hard_pass)
        soft_terms = last_evaluation.soft_score_terms() if last_evaluation else []
        spec_score = spec_compliance(hard_pass, soft_terms)
        if not hard_pass:
            final_smiles = None
            canonical_smiles = None
        decision = "abstain" if abstained else ("accept" if hard_pass else "reject")
        interrupt_handled = (not interrupt_triggered) or bool(
            interrupt_result and interrupt_result.get("compliance")
        )
        input_canonical = None
        if task.input.smiles:
            input_canonical = canonicalize_smiles(task.input.smiles)
        candidate_canonical = (
            last_evaluation.canonical_smiles if last_evaluation else None
        )
        edit_dist = None
        edit_morgan = None
        if input_canonical and candidate_canonical:
            edit_dist = levenshtein(input_canonical, candidate_canonical)
            edit_morgan = morgan_tanimoto(input_canonical, candidate_canonical)
        expected = task.expected
        observed = observed_outcome(decision, hard_pass)

        return RunRecord(
            task_id=task.task_id,
            suite=task.suite,
            protocol=task.protocol,
            spec_id=task.spec_id,
            interrupt_expected=bool(
                (task.interrupt and task.interrupt.enabled) or task.interrupt_at_step
            ),
            rounds=rounds,
            expected=expected,
            observed=observed,
            hard_pass=hard_pass,
            spec_score=spec_score,
            soft_terms=soft_terms,
            final_smiles=final_smiles,
            canonical_smiles=canonical_smiles,
            abstained=abstained,
            interrupt_handled=interrupt_handled,
            interrupt_result=interrupt_result,
            edit_distance=edit_dist,
            edit_morgan_tanimoto=edit_morgan,
            final_p_hard_pass=last_p_hard_pass,
            decision=decision,
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
    response: AgentResponse, task: TaskModel
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
    action = response.get("action", "propose")
    action_tag = "ABSTAIN" if action == "abstain" else "CONTINUE"
    action_ok = action_tag in expected.allowed_actions

    checks = {
        "ack_ok": ack_ok or not expected.must_ack,
        "restate_ok": restate_ok or not expected.must_restate_goal,
        "state_ok": state_ok or not expected.must_report_state,
        "action_ok": action_ok,
    }
    return {
        "checks": checks,
        "compliance": all(checks.values()),
        "action": action,
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
        "task_id	hard_pass	spec_score	decision	p_hard_pass	rounds	edit_distance"
    ]
    for record in records:
        p_hard_pass = (
            f"{record.final_p_hard_pass:.3f}"
            if record.final_p_hard_pass is not None
            else ""
        )
        edit_distance = record.edit_distance if record.edit_distance is not None else ""
        leaderboard_rows.append(
            f"{record.task_id}	{int(record.hard_pass)}	{record.spec_score:.3f}"
            f"	{record.decision}	{p_hard_pass}	{len(record.rounds)}	{edit_distance}"
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
    }
    jsonio.write_json(summary_path, summary)
