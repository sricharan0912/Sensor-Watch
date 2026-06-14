"""Online anomaly detection — loads the trained LSTM Autoencoder at startup."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]

from app.detection.severity import classify_severity
from app.models.anomaly import AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    subset: str
    window_size: int
    latent_dim: int
    hidden_size: int
    n_features: int
    sensor_cols: list[str]
    threshold_warning: float
    threshold_critical: float
    val_loss: float
    n_train_windows: int
    trained_at: str
    model_version: str


@dataclass
class AnomalyScore:
    reconstruction_error: float
    severity: AlertSeverity


class AnomalyDetector:
    """Loaded once at app startup via lifespan; held in app.state.detector.

    Inference is synchronous and takes ~2ms for a (1, 30, 14) tensor on CPU.
    For high-throughput deployments, wrap predict() in run_in_executor.
    """

    def __init__(
        self,
        model: object,
        scaler: MinMaxScaler,
        metadata: ModelMetadata,
        device: torch.device,
    ) -> None:
        self._model = model
        self._scaler = scaler
        self.metadata = metadata
        self._device = device

    @classmethod
    def load(cls, artifacts_dir: Path) -> AnomalyDetector:
        """Load model weights, scaler, and metadata from *artifacts_dir*."""
        metadata_path = artifacts_dir / "model_metadata.json"
        model_path = artifacts_dir / "model.pt"
        scaler_path = artifacts_dir / "scaler.joblib"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"model_metadata.json not found in {artifacts_dir}. "
                "Run 'make train' first."
            )

        with open(metadata_path) as f:
            raw = json.load(f)

        metadata = ModelMetadata(**{k: raw[k] for k in ModelMetadata.__dataclass_fields__})

        # Import here to avoid circular deps at module level
        from ml.model.lstm_autoencoder import LSTMAutoencoder

        model = LSTMAutoencoder(
            input_size=metadata.n_features,
            hidden_size=metadata.hidden_size,
            latent_dim=metadata.latent_dim,
            seq_len=metadata.window_size,
        )
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()

        scaler: MinMaxScaler = joblib.load(scaler_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "Anomaly detector loaded",
            extra={
                "model_version": metadata.model_version,
                "threshold_warning": metadata.threshold_warning,
                "threshold_critical": metadata.threshold_critical,
                "device": str(device),
            },
        )

        return cls(model=model, scaler=scaler, metadata=metadata, device=device)

    def normalize(self, values: list[float]) -> list[float]:
        """Normalise a single reading's sensor values using the fitted scaler."""
        arr = np.array(values, dtype=np.float32).reshape(1, -1)
        return self._scaler.transform(arr).flatten().tolist()

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> AnomalyScore:
        """Run inference on a single (1, W, F) normalised window.

        Returns the reconstruction error (MSE) and severity classification.
        """
        tensor = torch.from_numpy(window.astype(np.float32)).to(self._device)
        errors = self._model.reconstruction_error(tensor)  # type: ignore[union-attr]
        error = float(errors[0].item())

        severity = classify_severity(
            error,
            self.metadata.threshold_warning,
            self.metadata.threshold_critical,
        )
        return AnomalyScore(reconstruction_error=error, severity=severity)
