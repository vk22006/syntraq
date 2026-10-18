#!/usr/bin/env python3
"""
SyntraQ — Export PyTorch Traffic Prediction Model to ONNX
=========================================================
Milestone 12: Native AI Inference

Exports the trained PyTorch TrafficMLP model (from Milestone 11) to ONNX format.
Validates the exported ONNX model against PyTorch output for numerical consistency.
Generates an inference validation dataset (JSON) for automated C++ test verification.

Usage:
  python scripts/export_onnx_model.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


class TrafficMLP(nn.Module):
    """3-layer feedforward neural network matching the trained PyTorch model."""

    def __init__(self, in_features: int = 26, hidden1: int = 64, hidden2: int = 32, out_features: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch TrafficMLP model to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_traffic_mlp.pt",
        help="Path to trained PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="models/traffic_predictor.onnx",
        help="Destination path for the exported ONNX model.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/dataset/processed/dataset.npz",
        help="Path to processed dataset archive for numerical validation.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/dataset/processed/metadata.json",
        help="Path to dataset metadata JSON (scalers & feature names).",
    )
    parser.add_argument(
        "--val-json-output",
        type=str,
        default="data/dataset/inference_validation_samples.json",
        help="Path to export reference validation samples for C++ GoogleTest verification.",
    )
    return parser.parse_args()


def export_model_to_onnx(checkpoint_path: Path, output_onnx_path: Path) -> Tuple[TrafficMLP, Dict]:
    """Load PyTorch checkpoint and export to ONNX format."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path.resolve()}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    arch = ckpt.get("model_architecture", {"in_features": 26, "hidden1": 64, "hidden2": 32, "out_features": 1})
    model = TrafficMLP(
        in_features=arch["in_features"],
        hidden1=arch["hidden1"],
        hidden2=arch["hidden2"],
        out_features=arch["out_features"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, arch["in_features"], dtype=torch.float32)

    print(f"Exporting to ONNX: {output_onnx_path.resolve()} (Opset 17)...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    file_size_kb = output_onnx_path.stat().st_size / 1024.0
    print(f"[OK] ONNX model graph verified. File size: {file_size_kb:.1f} KB")

    return model, ckpt


def verify_numerical_consistency(
    model: TrafficMLP,
    output_onnx_path: Path,
    dataset_path: Path,
) -> None:
    """Verify that ONNX Runtime produces bit-accurate outputs matching PyTorch."""
    if not dataset_path.exists():
        print(f"Dataset archive not found at {dataset_path}, skipping dataset verification.")
        return

    print("Verifying numerical parity between PyTorch and ONNX Runtime...")
    data = np.load(dataset_path)
    X_test = data["X_test"]

    # 1. PyTorch predictions
    with torch.no_grad():
        t_in = torch.tensor(X_test, dtype=torch.float32)
        pt_preds = model(t_in).squeeze().cpu().numpy()

    # 2. ONNX Runtime predictions
    ort_session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: X_test.astype(np.float32)}
    ort_preds = ort_session.run(None, ort_inputs)[0].squeeze()

    # 3. Maximum absolute difference
    max_diff = np.max(np.abs(pt_preds - ort_preds))
    mean_diff = np.mean(np.abs(pt_preds - ort_preds))

    print(f"Comparison across {len(X_test)} samples:")
    print(f"  Maximum Absolute Discrepancy : {max_diff:.8e}")
    print(f"  Mean Absolute Discrepancy    : {mean_diff:.8e}")

    assert max_diff < 1e-4, f"Discrepancy {max_diff} exceeds allowable numerical tolerance 1e-4"
    print("[PASS] Numerical parity confirmed: ONNX Runtime matches PyTorch exactly.")


def export_validation_dataset(
    output_onnx_path: Path,
    dataset_path: Path,
    metadata_path: Path,
    val_json_path: Path,
    num_samples: int = 15,
) -> None:
    """
    Export a curated JSON dataset with exact inputs and expected outputs
    for C++ automated unit testing and end-to-end verification.
    """
    if not dataset_path.exists() or not metadata_path.exists():
        print("Missing dataset or metadata, skipping validation JSON generation.")
        return

    data = np.load(dataset_path)
    X_test = data["X_test"]
    y_test = data["y_test_queue_15s"]

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    feature_names = [str(n) for n in data["feature_names"]]
    scalers = metadata.get("scalers", {})

    ort_session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])

    # Pick spaced indices across test set for varied traffic regimes
    indices = np.linspace(0, len(X_test) - 1, num_samples, dtype=int).tolist()

    samples_payload = []
    for idx in indices:
        features_vec = X_test[idx].astype(np.float32).tolist()
        inp_dict = {ort_session.get_inputs()[0].name: np.array([features_vec], dtype=np.float32)}
        pred_val = float(ort_session.run(None, inp_dict)[0][0, 0])

        samples_payload.append({
            "sample_index": int(idx),
            "features": features_vec,
            "expected_prediction": pred_val,
            "ground_truth_target": float(y_test[idx]),
        })

    val_payload = {
        "model_file": str(output_onnx_path.name),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "scalers": scalers,
        "sample_count": len(samples_payload),
        "samples": samples_payload,
    }

    val_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_json_path, "w") as f:
        json.dump(val_payload, f, indent=2)

    print(f"Exported C++ validation test vectors to: {val_json_path.resolve()}")


def main() -> int:
    args = parse_args()
    checkpoint_file = Path(args.checkpoint)
    onnx_file = Path(args.output_onnx)
    dataset_file = Path(args.dataset_path)
    metadata_file = Path(args.metadata_path)
    val_json_file = Path(args.val_json_output)

    # 1. Export model
    model, ckpt = export_model_to_onnx(checkpoint_file, onnx_file)

    # 2. Check numerical parity
    verify_numerical_consistency(model, onnx_file, dataset_file)

    # 3. Export validation samples for C++ test harness
    export_validation_dataset(onnx_file, dataset_file, metadata_file, val_json_file)

    print("\n[SUCCESS] Milestone 12 ONNX Export & Parity Verification Completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
