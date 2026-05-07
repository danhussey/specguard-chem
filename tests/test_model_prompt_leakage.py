from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_audit():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_model_prompt_leakage.py"
    spec = importlib.util.spec_from_file_location("audit_model_prompt_leakage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_model_prompt_leakage_audit_passes(v1_release: Path, v1_tasks: list[dict]) -> None:
    summary = _load_audit().audit_release_model_prompts(v1_release)
    assert summary["actual_model_prompts_checked"] == len(v1_tasks)
    assert summary["oracle_field_leaks"] == 0
    assert summary["literal_witness_leaks"] == 0
    assert summary["label_leaks"] == 0
    assert summary["audit_accept_reject_name_leaks"] == 0
    assert summary["valid"] is True
