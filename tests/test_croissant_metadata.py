from __future__ import annotations

import json

from specguard_chem.dataset.validate_v1 import validate_croissant_metadata


def test_croissant_metadata_structure(v1_release) -> None:
    path = v1_release / "croissant.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"]
    assert payload["version"] == "sgchem_v1.0"
    assert payload["license"]
    rendered = json.dumps(payload, sort_keys=True)
    assert "tasks/train.jsonl" in rendered
    assert "tasks/dev.jsonl" in rendered
    assert "tasks/test.jsonl" in rendered
    assert "specs/spec_catalog.json" in rendered
    assert "responsibleAI" in payload
    assert validate_croissant_metadata(path, anonymous=True)["valid"] is True
