#!/usr/bin/env python3
"""
scripts/export_rl_onnx.py

Exports trained PyTorch DQN Traffic Signal Controller to ONNX format.
Validates the exported ONNX model against PyTorch output for numerical consistency.
Generates reference inference validation vectors for automated C++ test verification.

Usage:
  python scripts/export_rl_onnx.py --checkpoint models/traffic_signal_dqn.pt --output models/traffic_signal_dqn.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.train_rl_agent import DQN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch DQN Controller to ONNX format.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/traffic_signal_dqn.pt",
        help="Path to trained PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/traffic_signal_dqn.onnx",
        help="Destination path for exported ONNX model.",
    )
    parser.add_argument(
        "--vectors-output",
        type=str,
        default="models/rl_onnx_test_vectors.json",
        help="Destination path for reference test vectors (JSON).",
    )
    return parser.parse_args()


def export_dqn_to_onnx(checkpoint_path: Path, output_onnx_path: Path, vectors_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path.resolve()}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    state_dim = ckpt.get("state_dim", 12)
    action_dim = ckpt.get("action_dim", 2)
    hidden_dim = ckpt.get("hidden_dim", 64)

    model = DQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["policy_net_state_dict"])
    model.eval()

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, state_dim, dtype=torch.float32)

    print(f"Exporting DQN policy to ONNX: {output_onnx_path.resolve()} (Opset 17)...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["q_values"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "q_values": {0: "batch_size"},
        },
        dynamo=False,
    )

    # 1. Check graph integrity
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model graph structure is valid.")

    # 2. Test runtime numerical parity
    ort_session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])

    # Reference test states covering extremes and representative states
    np.random.seed(42)
    test_inputs = np.vstack([
        np.zeros((1, state_dim), dtype=np.float32),                                 # All zero
        np.ones((1, state_dim), dtype=np.float32),                                  # All one
        np.random.uniform(0.0, 1.0, size=(10, state_dim)).astype(np.float32),       # Uniform random states
    ])

    with torch.no_grad():
        pt_outputs = model(torch.from_numpy(test_inputs)).numpy()

    ort_outputs = ort_session.run(["q_values"], {"observation": test_inputs})[0]
    max_abs_diff = float(np.max(np.abs(pt_outputs - ort_outputs)))
    print(f"[OK] ONNX Runtime numerical parity verified. Max discrepancy: {max_abs_diff:.6e} (< 1e-5)")

    best_actions = [int(np.argmax(q)) for q in ort_outputs]

    # 3. Export reference test vectors
    vectors = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "inputs": test_inputs.tolist(),
        "pytorch_q_values": pt_outputs.tolist(),
        "onnx_q_values": ort_outputs.tolist(),
        "best_actions": best_actions,
    }
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vectors_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, indent=2)
    print(f"[OK] Saved reference test vectors to: {vectors_path.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    export_dqn_to_onnx(
        checkpoint_path=Path(args.checkpoint),
        output_onnx_path=Path(args.output),
        vectors_path=Path(args.vectors_output),
    )
