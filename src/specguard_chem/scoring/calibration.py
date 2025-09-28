from __future__ import annotations

"""Calibration metrics (Brier, ECE)."""

import math
from typing import Iterable, Sequence


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    for truth, prob in zip(y_true, y_prob):
        prob = min(max(prob, 0.0), 1.0)
        total += (prob - truth) ** 2
    return total / len(y_true)


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    if not y_true:
        return 0.0
    bin_totals = [0] * n_bins
    bin_acc = [0.0] * n_bins
    bin_conf = [0.0] * n_bins
    for truth, prob in zip(y_true, y_prob):
        clamped = min(max(prob, 0.0), 1.0)
        index = min(int(clamped * n_bins), n_bins - 1)
        bin_totals[index] += 1
        bin_acc[index] += truth
        bin_conf[index] += clamped
    ece = 0.0
    total_samples = len(y_true)
    for total, acc_sum, conf_sum in zip(bin_totals, bin_acc, bin_conf):
        if not total:
            continue
        acc = acc_sum / total
        conf = conf_sum / total
        ece += abs(acc - conf) * (total / total_samples)
    return ece
