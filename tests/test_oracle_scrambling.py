from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_audit():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_oracle_scrambling.py"
    spec = importlib.util.spec_from_file_location("audit_oracle_scrambling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_scrambling_does_not_change_public_views_or_baseline_outputs(v1_release: Path) -> None:
    summary = _load_audit().audit_oracle_scrambling(v1_release)
    assert summary["public_views_identical_under_oracle_scrambling"] is True
    assert summary["non_oracle_baseline_outputs_identical"] is True
    assert summary["valid"] is True
