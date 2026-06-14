from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from app.detection.inference import AnomalyDetector, AnomalyScore, ModelMetadata
from app.models.anomaly import AlertSeverity


def _make_metadata(**overrides: object) -> ModelMetadata:
    defaults = dict(
        subset="FD001",
        window_size=5,
        latent_dim=8,
        hidden_size=16,
        n_features=3,
        sensor_cols=["s2", "s3", "s4"],
        threshold_warning=0.005,
        threshold_critical=0.010,
        val_loss=0.002,
        n_train_windows=100,
        trained_at="2026-05-16T00:00:00Z",
        model_version="test-v1",
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)  # type: ignore[arg-type]


class TestAnomalyDetectorPredict:
    def _make_detector(self, error: float) -> AnomalyDetector:
        from sklearn.preprocessing import MinMaxScaler

        from ml.model.lstm_autoencoder import LSTMAutoencoder

        metadata = _make_metadata()

        mock_model = MagicMock(spec=LSTMAutoencoder)
        mock_model.reconstruction_error.return_value = torch.tensor([error])

        mock_scaler = MagicMock(spec=MinMaxScaler)

        detector = AnomalyDetector(
            model=mock_model,
            scaler=mock_scaler,
            metadata=metadata,
            device=torch.device("cpu"),
        )
        return detector

    def test_normal_when_error_below_warning(self) -> None:
        detector = self._make_detector(error=0.001)
        window = np.zeros((1, 5, 3), dtype=np.float32)
        score = detector.predict(window)
        assert score.severity == AlertSeverity.NORMAL
        assert abs(score.reconstruction_error - 0.001) < 1e-6

    def test_warning_when_error_at_warning_threshold(self) -> None:
        # Use 0.0051 — float32 representation of exactly 0.005 can be <0.005
        detector = self._make_detector(error=0.0051)
        window = np.zeros((1, 5, 3), dtype=np.float32)
        score = detector.predict(window)
        assert score.severity == AlertSeverity.WARNING

    def test_critical_when_error_exceeds_critical(self) -> None:
        detector = self._make_detector(error=0.020)
        window = np.zeros((1, 5, 3), dtype=np.float32)
        score = detector.predict(window)
        assert score.severity == AlertSeverity.CRITICAL

    def test_returns_anomaly_score_type(self) -> None:
        detector = self._make_detector(error=0.003)
        window = np.zeros((1, 5, 3), dtype=np.float32)
        score = detector.predict(window)
        assert isinstance(score, AnomalyScore)


class TestAnomalyDetectorNormalize:
    def test_normalize_calls_scaler(self) -> None:
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler

        metadata = _make_metadata()
        mock_model = MagicMock()
        mock_scaler = MagicMock(spec=MinMaxScaler)
        mock_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])

        detector = AnomalyDetector(
            model=mock_model,
            scaler=mock_scaler,
            metadata=metadata,
            device=torch.device("cpu"),
        )

        result = detector.normalize([100.0, 200.0, 300.0])
        mock_scaler.transform.assert_called_once()
        assert len(result) == 3
