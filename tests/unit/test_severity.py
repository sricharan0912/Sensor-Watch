from __future__ import annotations

import pytest

from app.detection.severity import classify_severity
from app.models.anomaly import AlertSeverity

_WARNING = 0.005
_CRITICAL = 0.010


@pytest.mark.parametrize("error,expected", [
    (0.000, AlertSeverity.NORMAL),
    (0.004, AlertSeverity.NORMAL),
    (0.004999, AlertSeverity.NORMAL),
    (0.005, AlertSeverity.WARNING),        # exactly at warning threshold
    (0.007, AlertSeverity.WARNING),
    (0.009999, AlertSeverity.WARNING),
    (0.010, AlertSeverity.CRITICAL),       # exactly at critical threshold
    (0.020, AlertSeverity.CRITICAL),
    (1.000, AlertSeverity.CRITICAL),
])
def test_classify_severity(error: float, expected: AlertSeverity) -> None:
    result = classify_severity(error, _WARNING, _CRITICAL)
    assert result == expected


def test_normal_when_zero_error() -> None:
    assert classify_severity(0.0, _WARNING, _CRITICAL) == AlertSeverity.NORMAL


def test_warning_threshold_is_inclusive() -> None:
    assert classify_severity(_WARNING, _WARNING, _CRITICAL) == AlertSeverity.WARNING


def test_critical_threshold_is_inclusive() -> None:
    assert classify_severity(_CRITICAL, _WARNING, _CRITICAL) == AlertSeverity.CRITICAL
