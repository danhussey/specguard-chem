from __future__ import annotations

"""Compatibility wrapper for the renamed external-baseline builder."""

from build_external_baselines import main


if __name__ == "__main__":
    raise SystemExit(main())
