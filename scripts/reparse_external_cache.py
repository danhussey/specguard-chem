from __future__ import annotations

"""Rebuild parsed external-cache responses from saved raw model output."""

import argparse
import json
from pathlib import Path
from typing import Any

from specguard_chem.models.anthropic_adapter import AnthropicChatAdapter
from specguard_chem.models.deepseek_adapter import DeepSeekChatAdapter
from specguard_chem.models.openai_adapter import OpenAIChatAdapter, _parse_json_object


class _UnusedOpenAIClient:
    chat = object()


class _UnusedAnthropicClient:
    messages = object()


def _adapter(provider: str) -> OpenAIChatAdapter:
    if provider == "anthropic":
        return AnthropicChatAdapter(client=_UnusedAnthropicClient())
    if provider == "deepseek":
        return DeepSeekChatAdapter(client=_UnusedOpenAIClient())
    return OpenAIChatAdapter(client=_UnusedOpenAIClient())


def _fallback(provider: str, raw: str) -> dict[str, Any]:
    error_type = "empty_output" if not raw.strip() else "malformed_json"
    return {
        "action": "interface_error",
        "reason": (
            f"{provider.capitalize()} API returned empty message content"
            if not raw.strip()
            else f"{provider.capitalize()} response was not valid JSON"
        ),
        "p_hard_pass": 0.0,
        "interface_error_type": error_type,
    }


def reparse_cache_file(path: Path, *, providers: set[str] | None) -> int:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    adapters: dict[str, OpenAIChatAdapter] = {}
    changed = 0
    for row in rows:
        metadata = row.get("model_metadata")
        provider = str((metadata or {}).get("provider") or "").strip().lower()
        if providers is not None and provider not in providers:
            continue
        raw = row.get("raw_model_output")
        if not isinstance(raw, str):
            continue
        cache_request = row.get("adapter_request")
        request_payload = (cache_request or {}).get("request")
        if not isinstance(request_payload, dict):
            continue
        data = _parse_json_object(raw) or _fallback(provider or "external", raw)
        adapter = adapters.setdefault(provider, _adapter(provider))
        reparsed = adapter._normalize_response(data, request_payload)
        if row.get("parsed_adapter_response") != reparsed:
            row["parsed_adapter_response"] = reparsed
            changed += 1
    if changed:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--providers",
        default="",
        help="Optional comma-separated provider allow-list.",
    )
    args = parser.parse_args(argv)
    providers = {part.strip().lower() for part in args.providers.split(",") if part.strip()} or None
    total_changed = 0
    for path in sorted(args.cache_root.glob("*/cache.jsonl")):
        total_changed += reparse_cache_file(path, providers=providers)
    print(json.dumps({"cache_root": str(args.cache_root), "changed": total_changed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
