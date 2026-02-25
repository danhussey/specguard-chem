from __future__ import annotations

from pathlib import Path

import pytest

from specguard_chem.runner.runner import TaskRunner


def test_runner_writes_cache_and_replays_identically(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    runner_a = TaskRunner("heuristic", seed=7, cache_dir=cache_dir)
    record_a = runner_a.run_suite(
        "basic_plain", protocol="L1", limit=1, run_dir=run_a
    )[0]

    cache_file = cache_dir / "cache.jsonl"
    assert cache_file.exists()
    cache_lines = cache_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(cache_lines) >= 1
    assert '"adapter_request"' in cache_lines[0]
    assert '"raw_model_output"' in cache_lines[0]
    assert '"parsed_adapter_response"' in cache_lines[0]
    assert '"model_metadata"' in cache_lines[0]

    # Replay must not depend on live adapter outputs.
    runner_b = TaskRunner("heuristic", seed=999, replay_cache=cache_dir)
    record_b = runner_b.run_suite(
        "basic_plain", protocol="L1", limit=1, run_dir=run_b
    )[0]

    assert record_b.hard_pass == record_a.hard_pass
    assert record_b.final_decision == record_a.final_decision
    assert record_b.final_smiles == record_a.final_smiles
    assert record_b.steps_used == record_a.steps_used


def test_external_adapter_requires_explicit_allow_external() -> None:
    with pytest.raises(RuntimeError, match="allow-external"):
        TaskRunner("process", seed=0)


def test_external_adapter_can_replay_without_live_process(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    TaskRunner("heuristic", seed=7, cache_dir=cache_dir).run_suite(
        "basic_plain", protocol="L1", limit=1
    )

    replay_runner = TaskRunner("process", seed=0, replay_cache=cache_dir)
    record = replay_runner.run_suite("basic_plain", protocol="L1", limit=1)[0]
    assert record.steps_used >= 1
