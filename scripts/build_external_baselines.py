from __future__ import annotations

"""Build external LLM baseline matrices from a provider/model config."""

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

ALLOWED_KWARGS = [
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "timeout",
    "base_url",
    "interface_tier",
    "response_format_json",
    "anthropic_beta",
]


def _clean_kwargs(raw: dict[str, Any], *, interface_tier: str) -> dict[str, Any]:
    kwargs = {key: raw.get(key) for key in ALLOWED_KWARGS if key in raw}
    kwargs["interface_tier"] = interface_tier
    if interface_tier == "prompt-json":
        kwargs["response_format_json"] = False
    if interface_tier == "json-mode":
        kwargs["response_format_json"] = True
    return kwargs


def _slug(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _provider_available(provider_cfg: dict[str, Any]) -> bool:
    env_name = provider_cfg.get("env")
    return isinstance(env_name, str) and bool(os.getenv(env_name))


def _interfaces_for(
    *,
    provider_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    interface_filter: set[str] | None,
) -> list[str]:
    raw = model_cfg.get("interface_tiers", provider_cfg.get("interface_tiers"))
    if raw is None:
        raw = ["strict-tool-call"]
    if not isinstance(raw, list):
        raise ValueError("interface_tiers must be a list")
    tiers = [str(item).strip() for item in raw if str(item).strip()]
    if interface_filter is not None:
        tiers = [tier for tier in tiers if tier in interface_filter]
    return tiers


def build_baselines(
    config_path: Path,
    *,
    providers_filter: set[str] | None = None,
    interface_filter: set[str] | None = None,
    skip_names: set[str] | None = None,
    only_available_env: bool = False,
) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    providers = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(providers, dict):
        raise ValueError("external model config must contain a providers object")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for provider, provider_cfg in providers.items():
        provider = str(provider)
        if providers_filter is not None and provider not in providers_filter:
            continue
        if not isinstance(provider_cfg, dict):
            continue
        if only_available_env and not _provider_available(provider_cfg):
            skipped.append({"provider": provider, "reason": "missing_env"})
            continue
        adapter = str(provider_cfg.get("adapter") or "").strip()
        verify_adapter = str(provider_cfg.get("verify_adapter") or "").strip()
        models = provider_cfg.get("models")
        if not adapter or not verify_adapter or not isinstance(models, dict):
            raise ValueError(f"provider {provider!r} is missing adapter/model config")
        for model_tier, model_cfg in models.items():
            model_tier = str(model_tier)
            if not isinstance(model_cfg, dict):
                raise ValueError(f"provider {provider!r} model tier {model_tier!r} is invalid")
            for interface_tier in _interfaces_for(
                provider_cfg=provider_cfg,
                model_cfg=model_cfg,
                interface_filter=interface_filter,
            ):
                kwargs = _clean_kwargs(model_cfg, interface_tier=interface_tier)
                interface_slug = _slug(interface_tier)
                closed_name = f"{provider}_{model_tier}_{interface_slug}_closed"
                verify_name = f"{provider}_{model_tier}_{interface_slug}_verify_l3"
                if skip_names is None or closed_name not in skip_names:
                    rows.append(
                        {
                            "name": closed_name,
                            "model": adapter,
                            "track": "external",
                            "optional": False,
                            "adapter_kwargs": dict(kwargs),
                        }
                    )
                if skip_names is None or verify_name not in skip_names:
                    rows.append(
                        {
                            "name": verify_name,
                            "model": verify_adapter,
                            "track": "external",
                            "optional": False,
                            "adapter_kwargs": dict(kwargs),
                        }
                    )
    return {"baselines": rows, "skipped_providers": skipped}


def _split_csv(value: str) -> set[str] | None:
    parsed = {part.strip() for part in value.split(",") if part.strip()}
    return parsed or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--providers",
        type=str,
        default="",
        help="Optional comma-separated provider allow-list.",
    )
    parser.add_argument(
        "--interface-tiers",
        type=str,
        default="",
        help="Optional comma-separated interface-tier allow-list.",
    )
    parser.add_argument(
        "--skip-names",
        type=str,
        default="",
        help="Optional comma-separated baseline names to omit.",
    )
    parser.add_argument(
        "--only-available-env",
        action="store_true",
        help="Include only providers whose configured API-key env var is set.",
    )
    args = parser.parse_args(argv)

    payload = build_baselines(
        args.config,
        providers_filter=_split_csv(args.providers),
        interface_filter=_split_csv(args.interface_tiers),
        skip_names=_split_csv(args.skip_names),
        only_available_env=args.only_available_env,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
