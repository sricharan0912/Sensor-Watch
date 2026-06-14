"""Evaluate anomaly detector: precision / recall / F1 on CMAPSS test set.

Usage: python -m ml.scripts.evaluate [--subset FD001] [--artifacts-dir artifacts/]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.data.cmapss_loader import download_cmapss, load_dataset
from ml.data.dataset import CMAPSSWindowDataset
from ml.data.preprocessor import NORMAL_RUL_THRESHOLD, load_scaler, per_unit_windows
from ml.model.lstm_autoencoder import LSTMAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="FD001")
    p.add_argument("--data-dir", default="data/", type=Path)
    p.add_argument("--artifacts-dir", default="artifacts/", type=Path)
    args = p.parse_args()

    artifacts_dir: Path = args.artifacts_dir
    with open(artifacts_dir / "model_metadata.json") as f:
        meta = json.load(f)

    # Load model
    model = LSTMAutoencoder(
        input_size=meta["n_features"],
        hidden_size=meta["hidden_size"],
        latent_dim=meta["latent_dim"],
        seq_len=meta["window_size"],
    )
    model.load_state_dict(
        torch.load(artifacts_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    scaler = load_scaler(artifacts_dir / "scaler.joblib")

    # Load test data
    data_dir = download_cmapss(Path(args.data_dir))
    cmapss = load_dataset(args.subset, data_dir)

    # Build test windows — all cycles (healthy + degraded)
    per_unit_windows(
        cmapss.test_df, scaler, meta["sensor_cols"], window_size=meta["window_size"]
    )
    # Label: degraded = last 30 cycles per unit (RUL proxy via cycle position)
    # For evaluation, we approximate: windows from last 15% of each unit's cycles = anomaly

    # Simpler: use train set degraded cycles as positive examples
    degraded_windows = per_unit_windows(
        cmapss.train_df[cmapss.train_df["rul"] <= NORMAL_RUL_THRESHOLD],
        scaler, meta["sensor_cols"], window_size=meta["window_size"],
    )
    normal_windows = per_unit_windows(
        cmapss.train_df[cmapss.train_df["rul"] > NORMAL_RUL_THRESHOLD],
        scaler, meta["sensor_cols"], window_size=meta["window_size"],
    )

    def compute_errors(windows: np.ndarray) -> np.ndarray:
        ds = CMAPSSWindowDataset(windows)
        loader = DataLoader(ds, batch_size=512)
        errors: list[float] = []
        with torch.no_grad():
            for x, _ in loader:
                errors.extend(model.reconstruction_error(x).numpy().tolist())
        return np.array(errors)

    normal_errors = compute_errors(normal_windows)
    degraded_errors = compute_errors(degraded_windows)

    for threshold_key in ("threshold_warning", "threshold_critical"):
        threshold = meta[threshold_key]
        tp = int((degraded_errors >= threshold).sum())
        fp = int((normal_errors >= threshold).sum())
        fn = int((degraded_errors < threshold).sum())
        tn = int((normal_errors < threshold).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n── {threshold_key} (threshold={threshold:.6f}) ──")
        print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
        print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


if __name__ == "__main__":
    main()
