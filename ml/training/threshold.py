from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.model.lstm_autoencoder import LSTMAutoencoder


def calibrate_threshold(
    model: LSTMAutoencoder,
    loader: DataLoader,  # type: ignore[type-arg]
    percentile: float = 95.0,
    device: str = "cpu",
) -> float:
    """Compute the *percentile* reconstruction error on normal-data windows.

    The returned value is used as the anomaly detection threshold:
    windows with error > threshold are flagged as anomalies.
    """
    model.eval()
    errors: list[float] = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            per_sample_errors = model.reconstruction_error(batch_x)
            errors.extend(per_sample_errors.cpu().numpy().tolist())

    return float(np.percentile(errors, percentile))
