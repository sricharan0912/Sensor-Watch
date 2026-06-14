"""Export trained LSTM Autoencoder to ONNX for edge deployment.

Exports two ONNX graphs:
  - model.onnx          full autoencoder (input → reconstruction)
  - model_encoder.onnx  encoder only (input → latent vector z)

Usage:
    python -m ml.scripts.export_onnx [--artifacts-dir artifacts/] [--opset 17] [--verify]

The --verify flag runs a round-trip check with onnxruntime to confirm output
matches PyTorch within 1e-5 tolerance. Requires: pip install onnxruntime
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import warnings

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Suppress verbose PyTorch legacy-exporter deprecation notices
warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")
warnings.filterwarnings("ignore", message=".*batch_size other than 1.*LSTM.*")

from ml.model.lstm_autoencoder import LSTMAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_model(artifacts_dir: Path) -> tuple[LSTMAutoencoder, dict]:  # type: ignore[type-arg]
    meta_path = artifacts_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model_metadata.json not found in {artifacts_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

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
    return model, meta


def _export_full(
    model: LSTMAutoencoder,
    dummy: torch.Tensor,
    out_path: Path,
    opset: int,
) -> None:
    # dynamo=False forces the legacy TorchScript exporter (no onnxscript dependency)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["sensor_window"],
        output_names=["reconstruction", "latent_z"],
        dynamic_axes={
            "sensor_window": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
            "latent_z": {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Exported full autoencoder → %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


def _export_encoder(
    model: LSTMAutoencoder,
    dummy: torch.Tensor,
    out_path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        model.encoder,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["sensor_window"],
        output_names=["latent_z"],
        dynamic_axes={
            "sensor_window": {0: "batch_size"},
            "latent_z": {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Exported encoder only → %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


def _verify(
    model: LSTMAutoencoder,
    dummy: torch.Tensor,
    onnx_path: Path,
    output_index: int = 0,
    atol: float = 1e-5,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping verification (pip install onnxruntime)")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: dummy.numpy()})

    with torch.no_grad():
        pt_out = model(dummy)

    pt_arr = pt_out[output_index].numpy()
    ort_arr = ort_out[output_index]
    max_diff = float(np.abs(pt_arr - ort_arr).max())

    if max_diff <= atol:
        logger.info("Verification passed — max abs diff: %.2e (atol=%.2e)", max_diff, atol)
    else:
        logger.error("Verification FAILED — max abs diff: %.2e exceeds atol=%.2e", max_diff, atol)
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Export LSTM Autoencoder to ONNX")
    p.add_argument("--artifacts-dir", default="artifacts/", type=Path)
    p.add_argument("--opset", default=17, type=int, help="ONNX opset version (default 17)")
    p.add_argument("--verify", action="store_true", help="Verify ONNX output against PyTorch")
    p.add_argument("--batch-size", default=1, type=int, help="Dummy batch size for tracing")
    args = p.parse_args()

    artifacts_dir: Path = args.artifacts_dir
    model, meta = _load_model(artifacts_dir)

    window_size: int = meta["window_size"]
    n_features: int = meta["n_features"]
    dummy = torch.randn(args.batch_size, window_size, n_features)

    full_path = artifacts_dir / "model.onnx"
    encoder_path = artifacts_dir / "model_encoder.onnx"

    _export_full(model, dummy, full_path, args.opset)
    _export_encoder(model, dummy, encoder_path, args.opset)

    if args.verify:
        logger.info("Verifying full autoencoder…")
        _verify(model, dummy, full_path, output_index=0)
        logger.info("Verifying encoder…")

        import onnxruntime as ort  # noqa: PLC0415
        sess = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
        ort_z = sess.run(None, {"sensor_window": dummy.numpy()})[0]
        with torch.no_grad():
            pt_z = model.encoder(dummy).numpy()
        max_diff = float(np.abs(pt_z - ort_z).max())
        if max_diff <= 1e-5:
            logger.info("Encoder verification passed — max abs diff: %.2e", max_diff)
        else:
            logger.error("Encoder verification FAILED — max abs diff: %.2e", max_diff)
            sys.exit(1)

    print("\n✓ ONNX artifacts written:")
    for path in (full_path, encoder_path):
        print(f"  {path}  ({path.stat().st_size / 1024:.1f} KB)")
    print(
        f"\nDeploy with: onnxruntime (CPU), TensorRT (Jetson), or CoreML Tools (Apple Silicon)"
    )


if __name__ == "__main__":
    main()
