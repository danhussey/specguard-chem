from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from specguard_chem.models import register_adapter
from specguard_chem.models.process_adapter import DEFAULT_ENV_VAR, ProcessAdapter
from specguard_chem.runner.runner import TaskRunner

SCRIPT = """
import json, sys
request = json.load(sys.stdin)
if request["round"] == 1:
    sys.stdout.write(json.dumps({"action": "tool_call", "name": "verify", "args": {"smiles": "CC"}}))
else:
    sys.stdout.write(json.dumps({"action": "propose", "smiles": "CC(=O)NC1=CC=CC=C1O", "confidence": 0.6}))
"""


@pytest.fixture(autouse=True)
def _register(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "adapter.py"
    script.write_text(SCRIPT)
    command = [sys.executable, str(script)]
    monkeypatch.setenv(DEFAULT_ENV_VAR, " ".join(command))
    register_adapter(ProcessAdapter)
    yield


@pytest.mark.usefixtures("_register")
def test_process_adapter_handles_tool_and_proposal():
    runner = TaskRunner(ProcessAdapter.name, seed=0)
    record = runner.run_suite("basic", protocol="L3", limit=1)[0]
    assert record.rounds[0].action == "tool_call"
    assert record.rounds[1].action == "propose"
    assert record.rounds[1].smiles == "CC(=O)NC1=CC=CC=C1O"

