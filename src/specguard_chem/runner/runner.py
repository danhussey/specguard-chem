from __future__ import annotations

"""Protocol runner orchestrating agent interactions."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import (
    FailureVector,
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
from ..utils.seeds import seed_everything
from .adapter_api import AgentRequest
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
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class RunRecord:
    task_id: str
    suite: str
    protocol: ProtocolName
    rounds: List[RoundLog]
    hard_pass: bool
    spec_score: float
    soft_terms: List[tuple[float, float]]
    final_smiles: Optional[str]
    canonical_smiles: Optional[str]
    abstained: bool
    interrupt_handled: bool
    edit_distance: Optional[int]
    final_confidence: Optional[float]
    decision: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "suite": self.suite,
            "protocol": self.protocol,
            "rounds": [round_log.to_dict() for round_log in self.rounds],
            "hard_pass": self.hard_pass,
            "spec_score": self.spec_score,
            "soft_terms": [list(term) for term in self.soft_terms],
            "final_smiles": self.final_smiles,
            "canonical_smiles": self.canonical_smiles,
            "abstained": self.abstained,
            "interrupt_handled": self.interrupt_handled,
            "edit_distance": self.edit_distance,
            "final_confidence": self.final_confidence,
            "decision": self.decision,
        }


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

    def _run_task(
        self, task: TaskModel, evaluator: ConstraintEvaluator
    ) -> RunRecord:
        max_rounds = self.MAX_ROUNDS.get(task.protocol, 1)
        rounds: List[RoundLog] = []
        last_failure_vector: Optional[FailureVector] = None
        last_evaluation: Optional[EvaluationResult] = None
        abstained = False
        final_smiles: Optional[str] = None
        canonical_smiles: Optional[str] = None
        interrupt_triggered = False
        interrupt_resolved = False
        last_confidence: Optional[float] = None

        for round_index in range(1, max_rounds + 1):
            interrupt_payload = self._interrupt_payload(
                task, evaluator.spec, round_index
            )
            if interrupt_payload:
                interrupt_triggered = True
            request = AgentRequest(
                task=task.model_dump(mode="json"),
                round=round_index,
                tools=self._tool_spec(task.protocol),
                failure_vector=last_failure_vector.model_dump(mode="json")
                if last_failure_vector
                else None,
            )
            if interrupt_payload:
                request["interrupt"] = interrupt_payload
            response = self.adapter.step(request)
            action = response.get("action", "propose")
            response_conf = response.get("confidence")

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
                        confidence=response_conf,
                    )
                )
                last_failure_vector = failure_vector
                continue

            if action == "abstain":
                abstained = True
                last_confidence = response_conf
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
                        confidence=response_conf,
                    )
                )
                break

            smiles = response.get("smiles", "") or ""
            evaluation = evaluator.evaluate(smiles)
            failure_vector = evaluation.build_failure_vector(round_index)
            last_failure_vector = failure_vector
            last_evaluation = evaluation
            final_smiles = smiles
            canonical_smiles = evaluation.canonical_smiles
            last_confidence = response_conf
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
                    confidence=response_conf,
                )
            )
            if evaluation.hard_pass:
                if interrupt_triggered:
                    interrupt_resolved = True
                break

        hard_pass = bool(last_evaluation and last_evaluation.hard_pass)
        soft_terms = last_evaluation.soft_score_terms() if last_evaluation else []
        spec_score = spec_compliance(hard_pass, soft_terms)
        if not hard_pass:
            final_smiles = None
            canonical_smiles = None
        decision = (
            "abstain"
            if abstained
            else ("accept" if hard_pass else "reject")
        )
        interrupt_handled = (
            task.interrupt_at_step is None
            or (not interrupt_triggered)
            or (interrupt_resolved and hard_pass and not abstained)
        )
        edit_dist = None
        if task.input.smiles and final_smiles:
            edit_dist = levenshtein(task.input.smiles, final_smiles)

        return RunRecord(
            task_id=task.task_id,
            suite=task.suite,
            protocol=task.protocol,
            rounds=rounds,
            hard_pass=hard_pass,
            spec_score=spec_score,
            soft_terms=soft_terms,
            final_smiles=final_smiles,
            canonical_smiles=canonical_smiles,
            abstained=abstained,
            interrupt_handled=interrupt_handled,
            edit_distance=edit_dist,
            final_confidence=last_confidence,
            decision=decision,
        )

    @staticmethod
    def _tool_spec(protocol: str) -> List[Dict[str, Any]]:
        if protocol == "L3":
            return [{"name": "verify", "schema": {"smiles": "string"}}]
        return []

    @staticmethod
    def _interrupt_payload(
        task: TaskModel, spec: SpecModel, round_index: int
    ) -> Optional[Dict[str, Any]]:
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
            outcome.constraint.id for outcome in result.hard_outcomes if not outcome.passed
        ],
        "soft_fails": [
            outcome.constraint.id for outcome in result.soft_outcomes if not outcome.passed
        ],
        "properties": result.properties,
        "property_margins": result.property_margins,
        "alerts": result.alerts,
        "sa_score": result.sa_score,
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
        "task_id	hard_pass	spec_score	decision	confidence	rounds	edit_distance"
    ]
    for record in records:
        confidence = (
            f"{record.final_confidence:.3f}" if record.final_confidence is not None else ""
        )
        edit_distance = record.edit_distance if record.edit_distance is not None else ""
        leaderboard_rows.append(
            f"{record.task_id}	{int(record.hard_pass)}	{record.spec_score:.3f}"
            f"	{record.decision}	{confidence}	{len(record.rounds)}	{edit_distance}"
        )
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text("\n".join(leaderboard_rows) + "\n", encoding="utf-8")

    avg_rounds = (
        sum(len(record.rounds) for record in records) / len(records)
        if records
        else 0.0
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
