from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ml.data.dataset import CMAPSSWindowDataset
from ml.model.lstm_autoencoder import LSTMAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    subset: str = "FD001"
    window_size: int = 30
    latent_dim: int = 32
    hidden_size: int = 64
    n_features: int = 14
    batch_size: int = 256
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 1e-5
    val_split: float = 0.2
    num_workers: int = 0          # 0 = main process (safer on macOS / Windows)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    artifacts_dir: Path = Path("artifacts/")


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_trained: int
    train_time_seconds: float
    model: LSTMAutoencoder


def train(dataset: CMAPSSWindowDataset, config: TrainConfig) -> TrainResult:
    """Train the LSTM Autoencoder on *dataset* (normal-data windows only).

    Saves best model weights to *config.artifacts_dir / model.pt*.
    Returns a TrainResult with the loaded best model.
    """
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    # ── Train / validation split ──────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * config.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LSTMAutoencoder(
        input_size=config.n_features,
        hidden_size=config.hidden_size,
        latent_dim=config.latent_dim,
        seq_len=config.window_size,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = config.artifacts_dir / "model.pt"
    start_time = time.monotonic()

    logger.info(
        "Starting training",
        extra={
            "n_train": n_train,
            "n_val": n_val,
            "max_epochs": config.max_epochs,
            "device": config.device,
        },
    )

    for epoch in range(1, config.max_epochs + 1):
        # ── Training epoch ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= n_train

        # ── Validation epoch ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                x_hat, _ = model(batch_x)
                val_loss += criterion(x_hat, batch_y).item() * len(batch_x)
        val_loss /= n_val

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch complete",
            extra={
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": lr,
            },
        )

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping", extra={"epoch": epoch, "best_val_loss": best_val_loss})
                break

    elapsed = time.monotonic() - start_time

    # Load best weights before returning
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()

    logger.info(
        "Training complete",
        extra={
            "best_val_loss": round(best_val_loss, 6),
            "epochs": epoch,
            "duration_seconds": round(elapsed, 1),
        },
    )

    return TrainResult(
        best_val_loss=best_val_loss,
        epochs_trained=epoch,
        train_time_seconds=elapsed,
        model=model,
    )
