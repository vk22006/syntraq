#!/usr/bin/env python3
"""
Unit test verifying that the saved traffic prediction model bundle:
  1. Loads cleanly with joblib
  2. Contains all required fields (estimator, metadata, features, scaler stats)
  3. Predicts on test batch with matching dimensions and valid numerical outputs
"""

import os
from pathlib import Path
import joblib
import numpy as np


def test_saved_model_bundle_integrity():
    model_path = Path("models/best_traffic_predictor.joblib")
    assert model_path.exists(), f"Model bundle not found at: {model_path}"

    bundle = joblib.load(model_path)
    assert isinstance(bundle, dict), "Model bundle is not a dictionary"

    required_keys = [
        "model_name",
        "estimator",
        "target_name",
        "feature_names",
        "feature_count",
        "scaler_stats",
        "seed",
        "metrics",
        "timestamp",
    ]
    for key in required_keys:
        assert key in bundle, f"Missing key in model bundle: {key}"

    print(f"Verified model bundle: {bundle['model_name']} for {bundle['target_name']}")
    assert bundle["feature_count"] == 26, f"Expected 26 features, got {bundle['feature_count']}"

    # Verify inference on dummy batch
    dummy_input = np.zeros((5, 26), dtype=np.float32)
    estimator = bundle["estimator"]
    predictions = estimator.predict(dummy_input)

    assert predictions.shape == (5,), f"Expected shape (5,), got {predictions.shape}"
    assert not np.isnan(predictions).any(), "Predictions contain NaN"
    assert not np.isinf(predictions).any(), "Predictions contain Inf"
    print(f"Sample predictions on dummy zero-vector: {predictions}")

    # Verify inference on real test sample
    dataset_path = Path("data/dataset/processed/dataset.npz")
    if dataset_path.exists():
        data = np.load(dataset_path)
        X_test = data["X_test"][:10]
        test_preds = estimator.predict(X_test)
        assert test_preds.shape == (10,)
        print(f"Sample predictions on 10 test samples: {test_preds}")

    print("[PASSED] Model bundle integrity and inference tests passed cleanly.")


if __name__ == "__main__":
    test_saved_model_bundle_integrity()
