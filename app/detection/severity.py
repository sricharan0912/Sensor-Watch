from __future__ import annotations

from app.models.anomaly import AlertSeverity


def classify_severity(
    reconstruction_error: float,
    threshold_warning: float,
    threshold_critical: float,
) -> AlertSeverity:
    """Map a reconstruction error to a severity level."""
    if reconstruction_error >= threshold_critical:
        return AlertSeverity.CRITICAL
    if reconstruction_error >= threshold_warning:
        return AlertSeverity.WARNING
    return AlertSeverity.NORMAL
