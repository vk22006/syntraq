"""
tests/test_traffic_signal_env.py

Automated tests for Python Gymnasium TrafficSignalEnv and PyTorch RL compatibility:
- Gymnasium API compliance (spaces, reset, step)
- Observation vector dimensionality (12) and range bounds [0.0, 1.0]
- Action space Discrete(2) execution
- PyTorch tensor conversion and forward pass through neural network
- Full episode rollout loop and telemetry verification
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn as nn

# Ensure scripts/ directory is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.traffic_signal_env import TrafficSignalEnv


class TestTrafficSignalEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TrafficSignalEnv(
            step_duration_s=1.0,
            sim_dt_s=0.1,
            episode_duration_s=30.0,
            scenario="Medium",
            seed=42,
            min_green_s=4.0,
            max_green_s=25.0,
            yellow_s=3.0,
            all_red_s=2.0,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_spaces(self):
        self.assertIsNotNone(self.env.action_space)
        self.assertIsNotNone(self.env.observation_space)
        self.assertEqual(self.env.action_space.n, 2)
        self.assertEqual(self.env.observation_space.shape, (12,))
        self.assertEqual(self.env.observation_space.dtype, np.float32)

    def test_reset(self):
        obs, info = self.env.reset(seed=123)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (12,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

        self.assertIn("total_queue", info)
        self.assertIn("avg_wait_s", info)
        self.assertIn("congestion_ratio", info)
        self.assertIn("step_departures", info)
        self.assertIn("elapsed_s", info)

    def test_step_maintain_action(self):
        self.env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.env.step(0)

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (12,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        self.assertAlmostEqual(info["elapsed_s"], 1.0, delta=0.05)

    def test_step_switch_action_and_clearance(self):
        self.env.reset(seed=42)

        # Maintain for 4 steps to satisfy min_green_s (4.0s)
        for _ in range(4):
            obs, _, _, _, _ = self.env.step(0)

        self.assertEqual(obs[9], 1.0)  # can_switch == 1.0

        # Action 1: Switch phase
        obs, reward, _, _, info = self.env.step(1)
        self.assertEqual(obs[8], 1.0)  # is_clearing == 1.0 (Yellow/AllRed active)

    def test_pytorch_tensor_compatibility(self):
        obs, _ = self.env.reset(seed=99)
        tensor_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # Shape (1, 12)

        # Build a small MLP policy / Q-network
        q_net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        with torch.no_grad():
            q_values = q_net(tensor_obs)

        self.assertEqual(q_values.shape, (1, 2))
        action = torch.argmax(q_values, dim=1).item()
        self.assertIn(action, [0, 1])

        # Step environment using PyTorch selected action
        next_obs, reward, term, trunc, _ = self.env.step(action)
        self.assertEqual(next_obs.shape, (12,))

    def test_full_episode_rollout(self):
        obs, _ = self.env.reset(seed=2026)
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < 50:
            action = steps % 2  # Alternate maintain and switch attempts
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            steps += 1
            done = term or trunc

        self.assertTrue(done)
        self.assertGreater(steps, 0)
        self.assertGreaterEqual(info["elapsed_s"], 29.5)


if __name__ == "__main__":
    unittest.main()
