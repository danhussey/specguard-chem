from __future__ import annotations

"""Protocol runner orchestrating agent interactions."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import FailureVector, TaskModel, load_spec, load_tasks_for_suite, select_tasks
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

        for round_index in range(1, max_rounds + 1):
            request = AgentRequest(
                task=task.model_dump(mode="json"),
                round=round_index,
                tools=self._tool_spec(task.protocol),
                failure_vector=last_failure_vector.model_dump(mode="json")
                if last_failure_vector
                else None,
            )
            response = self.adapter.step(request)
            action = response.get("action", "propose")

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
                    )
                )
                last_failure_vector = failure_vector
                continue

            if action == "abstain":
                abstained = True
                rounds.append(
                    RoundLog(
                        round_index=round_index,
                        action=action,
                        smiles=None,
                        evaluation=None,
                        failure_vector=None,
                        tool_name=None,
                        abstained=True,
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
            rounds.append(
                RoundLog(
                    round_index=round_index,
                    action=action,
                    smiles=smiles,
                    evaluation=evaluation_summary(evaluation),
                    failure_vector=failure_vector.model_dump(mode="json"),
                    tool_name=None,
                    abstained=False,
                )
            )
            if evaluation.hard_pass:
                break

        hard_pass = bool(last_evaluation and last_evaluation.hard_pass)
        soft_terms = last_evaluation.soft_score_terms() if last_evaluation else []
        spec_score = spec_compliance(hard_pass, soft_terms)
        interrupt_handled = task.interrupt_at_step is None or hard_pass
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
        )

    @staticmethod
    def _tool_spec(protocol: str) -> List[Dict[str, Any]]:
        if protocol == "L3":
            return [{"name": "verify", "schema": {"smiles": "string"}}]
        return []


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
        "task_id	hard_pass	spec_score	abstained	rounds"
    ]
    for record in records:
        leaderboard_rows.append(
            f"{record.task_id}	{int(record.hard_pass)}	{record.spec_score:.3f}	{int(record.abstained)}	{len(record.rounds)}"
        )
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text("\n".join(leaderboard_rows) + "\n", encoding="utf-8")

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
    }
    jsonio.write_json(summary_path, summary)
