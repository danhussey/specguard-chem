from __future__ import annotations

"""Preflight configured external adapters with one tiny live request each."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from specguard_chem.models import build_adapter
from specguard_chem.utils import jsonio


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload.get("baselines") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("baseline config must contain a list")
    return [row for row in rows if isinstance(row, dict)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write the preflight report and return zero even when rows fail.",
    )
    args = parser.parse_args(argv)

    results: list[dict[str, Any]] = []
    for row in _load_rows(args.baselines):
        name = str(row.get("name") or "")
        model = str(row.get("model") or "")
        kwargs = row.get("adapter_kwargs") if isinstance(row.get("adapter_kwargs"), dict) else {}
        try:
            adapter = build_adapter(model, seed=7, **kwargs)
            response = adapter.step(
                {
                    "task": {
                        "task_id": "preflight",
                        "protocol": "L1",
                        "input": {},
                        "prompt": "Return a valid abstention action for adapter preflight.",
                    },
                    "spec": {},
                    "round": 1,
                    "tools": [],
                    "failure_vector": None,
                    "interrupt": None,
                }
            )
            metadata = adapter.model_metadata()
            artifacts = (
                adapter.consume_step_artifacts()
                if hasattr(adapter, "consume_step_artifacts")
                else {}
            )
            response_action = response.get("action") if isinstance(response, dict) else None
            interface_error_type = (
                response.get("interface_error_type")
                if isinstance(response, dict)
                else None
            )
            status = "fail" if response_action == "interface_error" else "pass"
            error = None
            if response_action == "interface_error":
                error = f"interface_error:{interface_error_type or 'unknown'}"
            results.append(
                {
                    "name": name,
                    "model": model,
                    "provider": metadata.get("provider"),
                    "model_id": metadata.get("model_id"),
                    "interface_tier": metadata.get("interface_tier"),
                    "schema_hash": metadata.get("schema_hash"),
                    "provider_feature": metadata.get("provider_feature"),
                    "status": status,
                    "error": error,
                    "response_action": response_action,
                    "raw_output_empty": not bool(str((artifacts or {}).get("raw_model_output") or "").strip()),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "model": model,
                    "status": "fail",
                    "error": str(exc),
                }
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_rows": len(results),
        "num_failed": sum(1 for row in results if row.get("status") != "pass"),
        "rows": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonio.write_json(args.out, payload)
    return 0 if args.allow_failures or payload["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
