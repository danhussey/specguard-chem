from __future__ import annotations

import pytest

from specguard_chem.scoring.calibration import brier_score, expected_calibration_error


def test_brier_score_basic_case() -> None:
    truths = [1, 0, 1, 0]
    probs = [0.9, 0.2, 0.7, 0.4]
    # Manual computation
    expected = sum((p - t) ** 2 for t, p in zip(truths, probs)) / len(truths)
    assert brier_score(truths, probs) == expected


def test_expected_calibration_error_two_bins() -> None:
    truths = [1, 0]
    probs = [0.9, 0.1]
    # Bin 9: |1 - 0.9| * 1/2 = 0.05 ; Bin1: |0 - 0.1| * 1/2 = 0.05
    assert expected_calibration_error(truths, probs) == pytest.approx(0.1)


def test_expected_calibration_error_custom_bins() -> None:
    truths = [1, 0, 1]
    probs = [0.8, 0.3, 0.2]
    ece = expected_calibration_error(truths, probs, n_bins=5)
    assert 0 <= ece <= 1
