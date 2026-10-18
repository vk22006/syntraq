#!/usr/bin/env python3
"""
Unit tests verifying the PyTorch TrafficMLP model:
  1. Checkpoint integrity and metadata schema in best_traffic_mlp.pt
  2. Neural network parameter count and architecture topology
  3. Feedforward inference shape and numerical stability (no NaNs or Infs)
  4. Holdout test set performance verification (R^2 > 0.85, RMSE < 3.0 vehicles)
"""

import os
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score


class TrafficMLP(nn.Module):
    """3-layer feedforward neural network matching training specification."""

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


def test_checkpoint_file_exists_and_contains_expected_keys():
    checkpoint_path = Path("models/best_traffic_mlp.pt")
    assert checkpoint_path.exists(), f"Model checkpoint not found at: {checkpoint_path}"

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    assert isinstance(ckpt, dict), "Checkpoint must be a dictionary"

    required_keys = [
        "model_architecture",
        "state_dict",
        "best_epoch",
        "best_val_loss",
        "target_name",
        "feature_names",
        "feature_count",
        "scaler_stats",
        "seed",
        "metrics",
        "latency_us_per_sample",
        "timestamp",
    ]
    for key in required_keys:
        assert key in ckpt, f"Missing required key in checkpoint: {key}"

    assert ckpt["feature_count"] == 26
    assert ckpt["model_architecture"]["in_features"] == 26
    assert ckpt["model_architecture"]["hidden1"] == 64
    assert ckpt["model_architecture"]["hidden2"] == 32
    assert ckpt["model_architecture"]["out_features"] == 1
    print(f"Checkpoint verified: best epoch {ckpt['best_epoch']} with val loss {ckpt['best_val_loss']:.4f}")


def test_model_architecture_parameter_count():
    model = TrafficMLP(in_features=26, hidden1=64, hidden2=32, out_features=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Layer 1: 26*64 + 64 = 1728
    # Layer 2: 64*32 + 32 = 2080
    # Layer 3: 32*1 + 1   = 33
    # Total: 3841
    assert total_params == 3841, f"Expected 3,841 parameters, got {total_params}"


def test_forward_pass_output_shape_and_finite():
    checkpoint_path = Path("models/best_traffic_mlp.pt")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = TrafficMLP(in_features=26, hidden1=64, hidden2=32, out_features=1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Test single-sample and batch inference
    batch_sizes = [1, 5, 32, 100]
    for b in batch_sizes:
        dummy_input = torch.randn(b, 26, dtype=torch.float32)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (b, 1), f"Expected shape ({b}, 1), got {output.shape}"
        assert not torch.isnan(output).any(), f"Output contains NaN for batch size {b}"
        assert not torch.isinf(output).any(), f"Output contains Inf for batch size {b}"


def test_evaluation_performance_meets_benchmarks():
    checkpoint_path = Path("models/best_traffic_mlp.pt")
    dataset_path = Path("data/dataset/processed/dataset.npz")
    assert dataset_path.exists(), f"Processed dataset not found at: {dataset_path}"

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = TrafficMLP(in_features=26, hidden1=64, hidden2=32, out_features=1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = np.load(dataset_path)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = data["y_test_queue_15s"]

    with torch.no_grad():
        preds = model(X_test).squeeze().cpu().numpy()

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Verified test set performance: R^2 = {r2:.4f}, RMSE = {rmse:.3f} vehicles")
    assert r2 > 0.85, f"R^2 must exceed 0.85 threshold, got {r2:.4f}"
    assert rmse < 3.0, f"RMSE must be below 3.0 vehicles, got {rmse:.3f}"
    print("[PASSED] All PyTorch model tests passed cleanly.")


if __name__ == "__main__":
    test_checkpoint_file_exists_and_contains_expected_keys()
    test_model_architecture_parameter_count()
    test_forward_pass_output_shape_and_finite()
    test_evaluation_performance_meets_benchmarks()
