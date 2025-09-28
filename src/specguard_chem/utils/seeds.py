from __future__ import annotations

"""Seed helpers to keep runs deterministic."""

import os
import random
from typing import Optional

import numpy as np
from rdkit import rdBase


def seed_everything(seed: int, *, numpy_gaussian_seed: Optional[int] = None) -> None:
    """Seed Python, NumPy, and RDKit PRNGs."""

    random.seed(seed)
    np.random.seed(numpy_gaussian_seed or seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.error")
    rdBase.rdkitVersion  # touch to ensure RDKit is initialised
