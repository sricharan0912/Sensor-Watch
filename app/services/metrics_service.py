from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.models.anomaly import AlertSeverity


@dataclass
class MetricsService:
    """In-process metrics counters.  Stored on app.state.metrics."""

    _start_time: float = field(default_factory=time.monotonic)
    total_readings: int = 0
    total_windows: int = 0
    total_anomalies: int = 0
    anomalies_by_severity: dict[str, int] = field(
        default_factory=lambda: {s.value: 0 for s in AlertSeverity}
    )
    _total_inference_ms: float = 0.0
    _inference_count: int = 0

    def record_readings(self, count: int) -> None:
        self.total_readings += count

    def record_window(self, inference_ms: float) -> None:
        self.total_windows += 1
        self._total_inference_ms += inference_ms
        self._inference_count += 1

    def record_anomaly(self, severity: AlertSeverity) -> None:
        self.total_anomalies += 1
        self.anomalies_by_severity[severity.value] += 1

    @property
    def avg_inference_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_ms / self._inference_count

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time
