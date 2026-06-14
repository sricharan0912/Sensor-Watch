"""LSTM Autoencoder for time series anomaly detection.

Architecture:
  Encoder: LSTM(14→64) → Dropout(0.2) → LSTM(64→32) → last hidden state (latent z)
  Decoder: repeat z × seq_len → LSTM(32→32) → Dropout(0.2) → LSTM(32→64) → Linear(64→14)

Reconstruction error (MSE) is the anomaly score; high error = anomaly.
Total parameters ≈ 64k → ~256KB float32 → deployable to Raspberry Pi / Jetson Nano.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_dim: int) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(hidden_size, latent_dim, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        _, (hidden, _) = self.lstm2(out)
        # hidden: (1, batch, latent_dim) → (batch, latent_dim)
        return hidden.squeeze(0)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, output_size: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(latent_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, z: Tensor) -> Tensor:
        # z: (batch, latent_dim) → repeat → (batch, seq_len, latent_dim)
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm1(z_repeated)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        # out: (batch, seq_len, hidden_size) → (batch, seq_len, output_size)
        return self.linear(out)


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder.

    Args:
        input_size:  Number of sensor features (default 14 for CMAPSS).
        hidden_size: LSTM hidden dimension (default 64).
        latent_dim:  Bottleneck dimension (default 32).
        seq_len:     Sequence / window length (default 30).
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        latent_dim: int = 32,
        seq_len: int = 30,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, input_size, seq_len)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_error(self, x: Tensor) -> Tensor:
        """Per-sample MSE between input and reconstruction.

        Returns shape (batch,) — the anomaly score for each window.
        """
        x_hat, _ = self(x)
        # Mean over time and feature dims → scalar per sample
        return ((x - x_hat) ** 2).mean(dim=(1, 2))
