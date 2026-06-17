from __future__ import annotations

"""Generate external-baseline subset manifests and legacy comparison outputs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_md_research_outputs import write_outputs, write_subset_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("benchmarks/releases/sgchem_v1.0"))
    parser.add_argument("--results", type=Path, default=Path("external_baselines/results"))
    parser.add_argument("--subset-manifest", type=Path, default=Path("external_baselines/config/diagnostic_subset.json"))
    parser.add_argument("--generate-subset", action="store_true")
    parser.add_argument("--subset-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-per-family", type=int, default=1)
    parser.add_argument("--offline-aggregate", type=Path, default=Path("external_baselines/results/raw_runs/subset_offline/aggregate.json"))
    parser.add_argument("--external-aggregate", type=Path, default=Path("external_baselines/results/raw_runs/external_replay/aggregate.json"))
    parser.add_argument("--live-aggregate", type=Path, default=Path("external_baselines/results/raw_runs/external_live/aggregate.json"))
    parser.add_argument("--external-cache-root", type=Path, default=Path("external_baselines/results/cache/external_live"))
    args = parser.parse_args(argv)

    if args.generate_subset or not args.subset_manifest.exists():
        write_subset_manifest(
            release=args.release,
            subset_manifest=args.subset_manifest,
            seed=args.seed,
            max_per_family=args.max_per_family,
        )
    if args.subset_only:
        return 0
    write_outputs(
        release=args.release,
        results=args.results,
        subset_manifest=args.subset_manifest,
        offline_aggregate=args.offline_aggregate,
        external_aggregate=args.external_aggregate if args.external_aggregate.exists() else None,
        external_cache_root=args.external_cache_root if args.external_cache_root.exists() else None,
        live_aggregate=args.live_aggregate if args.live_aggregate.exists() else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
