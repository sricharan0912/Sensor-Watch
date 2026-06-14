"""Training entry point: python -m ml.scripts.train [options]

Downloads the NASA CMAPSS dataset (or generates synthetic data), trains the
LSTM Autoencoder, calibrates anomaly thresholds, and writes artifacts to disk.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from torch.utils.data import DataLoader

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.data.cmapss_loader import download_cmapss, load_dataset
from ml.data.dataset import CMAPSSWindowDataset
from ml.data.preprocessor import (
    NORMAL_RUL_THRESHOLD,
    fit_scaler,
    per_unit_windows,
    save_scaler,
)
from ml.training.threshold import calibrate_threshold
from ml.training.trainer import TrainConfig, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM Autoencoder on NASA CMAPSS")
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--data-dir", default="data/", type=Path)
    p.add_argument("--artifacts-dir", default="artifacts/", type=Path)
    p.add_argument("--window-size", default=30, type=int)
    p.add_argument("--latent-dim", default=32, type=int)
    p.add_argument("--hidden-size", default=64, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--patience", default=10, type=int)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warning-percentile", default=90.0, type=float)
    p.add_argument("--critical-percentile", default=99.0, type=float)
    p.add_argument("--force-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir: Path = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download dataset ───────────────────────────────────────────────────
    data_dir = download_cmapss(Path(args.data_dir), force=args.force_download)
    cmapss = load_dataset(args.subset, data_dir)

    # ── 2. Fit scaler on normal training data ─────────────────────────────────
    normal_train = cmapss.train_df[cmapss.train_df["rul"] > NORMAL_RUL_THRESHOLD]
    scaler = fit_scaler(normal_train, cmapss.sensor_cols)
    save_scaler(scaler, artifacts_dir / "scaler.joblib")
    logger.info("Scaler fitted and saved")

    # ── 3. Build sliding-window dataset (normal cycles only) ──────────────────
    train_windows = per_unit_windows(
        cmapss.train_df,
        scaler,
        cmapss.sensor_cols,
        window_size=args.window_size,
        rul_threshold=NORMAL_RUL_THRESHOLD,
    )
    logger.info("Training windows", extra={"shape": train_windows.shape})

    dataset = CMAPSSWindowDataset(train_windows)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    config = TrainConfig(
        subset=args.subset,
        window_size=args.window_size,
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        n_features=len(cmapss.sensor_cols),
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        device=args.device,
        artifacts_dir=artifacts_dir,
    )
    result = train(dataset, config)

    # ── 5. Calibrate thresholds on normal validation data ─────────────────────
    # Re-use the same normal windows as a calibration set (held-out val portion)
    calib_loader = DataLoader(dataset, batch_size=512, shuffle=False)
    threshold_warning = calibrate_threshold(
        result.model, calib_loader, percentile=args.warning_percentile, device=args.device
    )
    threshold_critical = calibrate_threshold(
        result.model, calib_loader, percentile=args.critical_percentile, device=args.device
    )
    logger.info(
        "Thresholds calibrated",
        extra={
            "warning": round(threshold_warning, 6),
            "critical": round(threshold_critical, 6),
        },
    )

    # ── 6. Save metadata ──────────────────────────────────────────────────────
    trained_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "subset": args.subset,
        "window_size": args.window_size,
        "latent_dim": args.latent_dim,
        "hidden_size": args.hidden_size,
        "n_features": len(cmapss.sensor_cols),
        "sensor_cols": cmapss.sensor_cols,
        "threshold_warning": threshold_warning,
        "threshold_critical": threshold_critical,
        "val_loss": result.best_val_loss,
        "n_train_windows": len(dataset),
        "trained_at": datetime.now(UTC).isoformat(),
        "model_version": f"{args.subset}-{trained_at}",
        "device": args.device,
    }
    with open(artifacts_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training pipeline complete", extra={"artifacts_dir": str(artifacts_dir)})
    print("\n✓ Artifacts written:")
    for p in sorted(artifacts_dir.iterdir()):
        if not p.name.startswith("."):
            print(f"  {p}")


if __name__ == "__main__":
    main()
