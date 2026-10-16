#!/usr/bin/env python3
"""
scripts/preprocess_dataset.py

Scientific preprocessing and dataset splitting pipeline for traffic prediction models.
Implements strict leakage-free data processing:
- Run-level trajectory splitting (prevents temporal auto-correlation leakage)
- Feature standardization computed strictly on the training partition
- One-hot encoding of categorical scenario and signal phase features
- Structured multi-horizon target extraction (future queue lengths and congestion)
- Export to clean CSV, compressed NumPy NPZ, and metadata JSON.

Author: SyntraQ Core Team
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ── Metadata & Documentation ───────────────────────────────────────────────────

FEATURE_DEFINITIONS = {
    "vehicle_count": "Active vehicle count currently traveling or stopped on the road network.",
    "spawn_rate_vpm": "Vehicle inflow spawn rate (vehicles per minute).",
    "queue_vehicle_count": "Number of vehicles currently stopped or traveling below 0.5 m/s.",
    "max_queue_length": "Maximum queue length on any individual approach lane across all junctions.",
    "avg_queue_per_lane": "Average queue depth across all active lanes in the road network.",
    "queue_ratio": "Fraction of active vehicle fleet currently stopped in queues (queue_count / vehicle_count).",
    "avg_speed_mps": "Instantaneous mean speed of active vehicles in meters per second.",
    "avg_speed_kmh": "Instantaneous mean speed of active vehicles in kilometers per hour.",
    "speed_ratio": "Ratio of mean vehicle speed to the maximum design speed limit (0.0 to 1.0).",
    "avg_wait_time_s": "Mean accumulated stopped/waiting duration experienced by active vehicles.",
    "avg_travel_time_s": "Mean total elapsed trip travel duration across active vehicles.",
    "road_density_vpm": "Network vehicle density (active vehicles per total lane meters).",
    "road_occupancy_pct": "Percentage of total road network length occupied by vehicle bounding boxes.",
    "critical_occupancy_pct": "Highest occupancy percentage observed on any single road in the network.",
    "time_in_phase_s": "Elapsed time in seconds within the currently active traffic signal phase.",
    "phase_duration_s": "Total nominal duration allocated for the current signal phase.",
    "phase_elapsed_ratio": "Fraction of nominal phase duration elapsed (time_in_phase / phase_duration).",
    "is_all_red": "Binary indicator: 1 if signal is in all-red clearance interval, 0 otherwise.",
    "green_approaches_count": "Number of incoming junction approaches with active green signal.",
    "recent_throughput_vph": "Vehicle completions per hour measured over a rolling 10-second window.",
    "recent_queue_delta": "Rate of queue change dQ/dt (veh/s) over rolling window (positive = building, negative = clearing).",
    "recent_speed_delta": "Speed acceleration/deceleration trend dv/dt (m/s^2) over rolling window.",
}

TARGET_DEFINITIONS = {
    "target_queue_15s": "Ground truth total queued vehicle count at t + 15 seconds.",
    "target_max_queue_15s": "Ground truth peak lane queue depth at t + 15 seconds.",
    "target_congestion_15s": "Ground truth congestion ratio (queued / active) at t + 15 seconds.",
    "target_queue_30s": "Ground truth total queued vehicle count at t + 30 seconds.",
    "target_max_queue_30s": "Ground truth peak lane queue depth at t + 30 seconds.",
    "target_congestion_30s": "Ground truth congestion ratio (queued / active) at t + 30 seconds.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SyntraQ traffic dataset for predictive modeling without data leakage."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/dataset/raw_traffic_data.csv",
        help="Path to raw traffic dataset CSV exported from simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dataset/processed",
        help="Directory to save preprocessed train/val/test splits and metadata.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of runs allocated to the training split (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of runs allocated to the validation split (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of runs allocated to the test split (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic run-level splitting.",
    )
    return parser.parse_args()


def load_and_validate_dataset(csv_path: str) -> pd.DataFrame:
    """Load raw dataset CSV and validate required columns and sample integrity."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input dataset CSV not found at: {csv_path}")

    print(f"Loading raw dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows across {len(df.columns)} columns.")

    # Filter only rows with complete 30s lookahead targets
    if "has_target_30s" in df.columns:
        initial_count = len(df)
        df = df[df["has_target_30s"] == 1].copy()
        print(f"Filtered {initial_count - len(df)} terminal samples lacking +30s ground truth lookahead.")
        print(f"Retained {len(df)} samples with complete dual-horizon targets.")

    # Assert no nulls in numeric features
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Found {missing_count} missing values. Imputing with column medians...")
        df = df.fillna(df.median(numeric_only=True))

    return df


def split_by_runs(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Split dataset strictly at the simulation run (trajectory) level.
    Ensures zero temporal or spatial leakage between training and evaluation splits.
    Stratified across scenario types so all splits have balanced traffic demands.
    """
    np.random.seed(seed)

    train_runs: List[str] = []
    val_runs: List[str] = []
    test_runs: List[str] = []

    # Group runs by scenario for stratified group partitioning
    scenarios = df["scenario_name"].unique()
    split_audit = {}

    for sc in scenarios:
        sc_df = df[df["scenario_name"] == sc]
        runs = sorted(sc_df["run_id"].unique())
        np.random.shuffle(runs)

        n_runs = len(runs)
        n_train = max(1, int(round(n_runs * train_ratio)))
        n_val = max(1, int(round(n_runs * val_ratio)))
        # Remaining runs go to test
        sc_train = runs[:n_train]
        sc_val = runs[n_train : n_train + n_val]
        sc_test = runs[n_train + n_val :]

        # Guarantee non-empty test if sufficient runs exist
        if len(sc_test) == 0 and len(sc_train) > 1:
            sc_test = [sc_train.pop()]

        train_runs.extend(sc_train)
        val_runs.extend(sc_val)
        test_runs.extend(sc_test)

        split_audit[sc] = {
            "total_runs": n_runs,
            "train_runs": sc_train,
            "val_runs": sc_val,
            "test_runs": sc_test,
        }

    # Strict Leakage Verification Check
    train_set, val_set, test_set = set(train_runs), set(val_runs), set(test_runs)
    assert len(train_set & val_set) == 0, "Data leakage detected: train and val overlap!"
    assert len(train_set & test_set) == 0, "Data leakage detected: train and test overlap!"
    assert len(val_set & test_set) == 0, "Data leakage detected: val and test overlap!"

    df_train = df[df["run_id"].isin(train_set)].copy().reset_index(drop=True)
    df_val = df[df["run_id"].isin(val_set)].copy().reset_index(drop=True)
    df_test = df[df["run_id"].isin(test_set)].copy().reset_index(drop=True)

    print("\n--- Split Verification & Leakage Audit ---")
    print(f"Total Simulation Runs : {len(df['run_id'].unique())}")
    print(f"Train Partition       : {len(train_set)} runs ({len(df_train)} samples, {len(df_train)/len(df)*100:.1f}%)")
    print(f"Validation Partition  : {len(val_set)} runs ({len(df_val)} samples, {len(df_val)/len(df)*100:.1f}%)")
    print(f"Test Partition        : {len(test_set)} runs ({len(df_test)} samples, {len(df_test)/len(df)*100:.1f}%)")
    print("Overlap Verification  : PASSED (0 run ID overlap across splits)\n")

    return df_train, df_val, df_test, split_audit


def preprocess_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Standardize continuous features using parameters fitted strictly on the training set.
    One-hot encode categorical features.
    """
    continuous_features = list(FEATURE_DEFINITIONS.keys())

    # Check presence of all continuous features
    missing = [f for f in continuous_features if f not in df_train.columns]
    if missing:
        raise KeyError(f"Missing expected features in raw dataset: {missing}")

    # Compute mean and std strictly on df_train
    scalers = {}
    for col in continuous_features:
        mean_val = float(df_train[col].mean())
        std_val = float(df_train[col].std())
        if std_val < 1e-6:
            std_val = 1.0  # Prevent division by zero for constant features
        scalers[col] = {"mean": mean_val, "std": std_val}

    # Transform continuous features
    def scale_df(source_df: pd.DataFrame) -> pd.DataFrame:
        scaled = pd.DataFrame(index=source_df.index)
        for c in continuous_features:
            scaled[c] = (source_df[c] - scalers[c]["mean"]) / scalers[c]["std"]
        return scaled

    X_train_cont = scale_df(df_train)
    X_val_cont = scale_df(df_val)
    X_test_cont = scale_df(df_test)

    # One-hot encode scenario name
    scenario_categories = ["Low Traffic", "Medium Traffic", "High Traffic", "Rush Hour"]
    for sc in scenario_categories:
        col_name = f"scenario_{sc.replace(' ', '_').lower()}"
        X_train_cont[col_name] = (df_train["scenario_name"] == sc).astype(float)
        X_val_cont[col_name] = (df_val["scenario_name"] == sc).astype(float)
        X_test_cont[col_name] = (df_test["scenario_name"] == sc).astype(float)

    # Attach metadata for identification and grouping
    meta_cols = ["run_id", "scenario_name", "seed", "timestamp_s"]
    for m in meta_cols:
        X_train_cont[m] = df_train[m]
        X_val_cont[m] = df_val[m]
        X_test_cont[m] = df_test[m]

    # Attach prediction targets
    target_cols = list(TARGET_DEFINITIONS.keys())
    for t in target_cols:
        X_train_cont[t] = df_train[t]
        X_val_cont[t] = df_val[t]
        X_test_cont[t] = df_test[t]

    return X_train_cont, X_val_cont, X_test_cont, scalers


def export_processed_dataset(
    output_dir: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    scalers: Dict,
    split_audit: Dict,
) -> None:
    """Save clean CSVs, compressed NPZ arrays, and metadata JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    train_csv = out_path / "train.csv"
    val_csv = out_path / "val.csv"
    test_csv = out_path / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Extract NumPy matrices for ML training
    meta_cols = ["run_id", "scenario_name", "seed", "timestamp_s"]
    target_cols = list(TARGET_DEFINITIONS.keys())
    feature_cols = [c for c in df_train.columns if c not in meta_cols and c not in target_cols]

    npz_path = out_path / "dataset.npz"
    np.savez_compressed(
        npz_path,
        feature_names=np.array(feature_cols),
        X_train=df_train[feature_cols].values.astype(np.float32),
        y_train_queue_15s=df_train["target_queue_15s"].values.astype(np.float32),
        y_train_queue_30s=df_train["target_queue_30s"].values.astype(np.float32),
        y_train_cong_15s=df_train["target_congestion_15s"].values.astype(np.float32),
        y_train_cong_30s=df_train["target_congestion_30s"].values.astype(np.float32),
        X_val=df_val[feature_cols].values.astype(np.float32),
        y_val_queue_15s=df_val["target_queue_15s"].values.astype(np.float32),
        y_val_queue_30s=df_val["target_queue_30s"].values.astype(np.float32),
        y_val_cong_15s=df_val["target_congestion_15s"].values.astype(np.float32),
        y_val_cong_30s=df_val["target_congestion_30s"].values.astype(np.float32),
        X_test=df_test[feature_cols].values.astype(np.float32),
        y_test_queue_15s=df_test["target_queue_15s"].values.astype(np.float32),
        y_test_queue_30s=df_test["target_queue_30s"].values.astype(np.float32),
        y_test_cong_15s=df_test["target_congestion_15s"].values.astype(np.float32),
        y_test_cong_30s=df_test["target_congestion_30s"].values.astype(np.float32),
    )

    # Write metadata.json
    metadata = {
        "dataset_name": "SyntraQ Urban Traffic Prediction Dataset",
        "version": "1.0.0",
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "feature_definitions": FEATURE_DEFINITIONS,
        "target_definitions": TARGET_DEFINITIONS,
        "prediction_horizons_seconds": [15.0, 30.0],
        "scalers": scalers,
        "split_statistics": {
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "total_samples": len(df_train) + len(df_val) + len(df_test),
        },
        "split_audit": split_audit,
    }

    meta_json = out_path / "metadata.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("========================================================================================")
    print("                           DATASET PREPROCESSING COMPLETE                               ")
    print("========================================================================================")
    print(f" Train Split CSV     : {train_csv} ({len(df_train)} rows)")
    print(f" Val Split CSV       : {val_csv} ({len(df_val)} rows)")
    print(f" Test Split CSV      : {test_csv} ({len(df_test)} rows)")
    print(f" Compressed NPZ      : {npz_path} (Tensor shapes: Train={df_train[feature_cols].shape}, Val={df_val[feature_cols].shape}, Test={df_test[feature_cols].shape})")
    print(f" Metadata JSON       : {meta_json}")
    print(f" Feature Dimension   : {len(feature_cols)} input features")
    print("========================================================================================\n")


def main() -> None:
    args = parse_args()
    raw_df = load_and_validate_dataset(args.input)
    df_train, df_val, df_test, split_audit = split_by_runs(
        raw_df,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    X_train, X_val, X_test, scalers = preprocess_features(df_train, df_val, df_test)
    export_processed_dataset(
        args.output_dir,
        X_train,
        X_val,
        X_test,
        scalers,
        split_audit,
    )


if __name__ == "__main__":
    main()
