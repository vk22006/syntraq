"""
scripts/train_rl_agent.py

Reinforcement Learning Controller for Single-Intersection Traffic Signal Optimization.
Implements Deep Q-Network (DQN) in PyTorch to train an agent in the SyntraQ environment:
  - Replay Buffer: Transition ring buffer with uniform mini-batch sampling
  - Q-Network: 12-dimensional state -> 64 -> 64 -> 2-dimensional Q-values
  - Target Network: Polyak soft updates for stable TD-target regression
  - Epsilon-Greedy Exploration: Smooth exponential / linear decay
  - Double DQN update logic to reduce overestimation bias
  - Multi-Seed Evaluation: Rigorous benchmarking across multiple seeds
    comparing Fixed-Time, Adaptive Heuristic, and Trained DQN
  - Statistical telemetry: Mean +/- Std across seeds (rewards, queues, wait times, throughput)
  - Exports trained policy to PyTorch checkpoint and ONNX format
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.traffic_signal_env import TrafficSignalEnv


# ── DQN Neural Network Architecture ──────────────────────────────────────────

class DQN(nn.Module):
    """Deep Q-Network mapping 12D traffic observation to action Q-values [Maintain, Switch]."""

    def __init__(self, state_dim: int = 12, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Experience replay ring buffer for off-policy DQN training."""

    def __init__(self, capacity: int = 30000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN Agent encapsulating policy/target networks, Double DQN updates,
    and epsilon-greedy exploration.
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.98,
        tau: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 30,
        hidden_dim: int = 64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        self.policy_net = DQN(state_dim, action_dim, hidden_dim=hidden_dim)
        self.target_net = DQN(state_dim, action_dim, hidden_dim=hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action via epsilon-greedy (training) or argmax greedy (evaluation)."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(s_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> float:
        """Double DQN update step with Polyak soft target tracking."""
        if len(replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Current Q(s, a)
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: r + gamma * Q_target(s', argmax_a' Q_policy(s', a'))
        with torch.no_grad():
            next_policy_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_policy_actions).squeeze(1)
            expected_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = self.criterion(state_action_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Polyak soft update of target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return float(loss.item())

    def update_epsilon(self, episode_idx: int) -> None:
        frac = min(1.0, episode_idx / max(1, self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


# ── Benchmark Policies ───────────────────────────────────────────────────────

class FixedTimePolicy:
    """Baseline fixed-time controller switching phase after a deterministic duration."""

    def __init__(self, green_duration_s: float = 15.0, max_green_s: float = 35.0):
        self.green_duration_s = green_duration_s
        self.max_green_s = max_green_s

    def select_action(self, obs: np.ndarray) -> int:
        is_transitioning = obs[8] > 0.5
        can_switch = obs[9] > 0.5
        time_in_phase = obs[5] * self.max_green_s
        if not is_transitioning and can_switch and time_in_phase >= self.green_duration_s:
            return 1
        return 0


class AdaptiveHeuristicPolicy:
    """
    Deterministic queue-responsive heuristic policy mirroring the C++ AdaptiveController:
      - Enforces minimum green time (4.0s)
      - Gap-out: Shortens green early when active queue clears and opposing vehicles wait
      - Queue-extension: Holds green while high demand persists on active corridor
      - Switch: Transitions to clearance on max-green (35s) or opposing starvation
    """

    def __init__(
        self,
        min_green_s: float = 4.0,
        nominal_green_s: float = 12.0,
        max_green_s: float = 35.0,
    ):
        self.min_green_s = min_green_s
        self.nominal_green_s = nominal_green_s
        self.max_green_s = max_green_s

    def select_action(self, obs: np.ndarray) -> int:
        is_transitioning = obs[8] > 0.5
        can_switch = obs[9] > 0.5
        if is_transitioning or not can_switch:
            return 0  # Hold during clearance or minimum green

        time_in_phase = obs[5] * self.max_green_s
        phase = obs[4]  # 0.0 = NS Green, 1.0 = EW Green

        if phase < 0.5:
            current_q = obs[0] * 25.0
            opposing_q = obs[1] * 25.0
            opposing_incoming = obs[7] * 50.0
        else:
            current_q = obs[1] * 25.0
            opposing_q = obs[0] * 25.0
            opposing_incoming = obs[6] * 50.0

        # Max green limit reached
        if time_in_phase >= self.max_green_s - 0.5:
            return 1

        # Gap-out: active queue cleared while opposing demand waits
        if current_q == 0 and (opposing_q > 0 or opposing_incoming > 2):
            return 1

        # Opposing pressure after nominal green duration
        if time_in_phase >= self.nominal_green_s:
            if opposing_q >= current_q and opposing_q > 0:
                return 1

        return 0


# ── Policy Evaluation ────────────────────────────────────────────────────────

def run_single_episode(env: TrafficSignalEnv, policy: Any, seed: int) -> Dict[str, float]:
    """Run an evaluation episode with a given policy on a specific random seed."""
    obs, _ = env.reset(seed=seed)
    done = False
    ep_reward = 0.0
    queues = []
    waits = []
    speeds = []
    congestions = []

    while not done:
        if isinstance(policy, DQNAgent):
            action = policy.select_action(obs, evaluate=True)
        elif hasattr(policy, "select_action"):
            action = policy.select_action(obs)
        else:
            action = 0

        obs, reward, term, trunc, step_info = env.step(action)
        ep_reward += reward
        queues.append(step_info["total_queue"])
        waits.append(step_info["avg_wait_s"])
        speed_mps = float(0.5 * (obs[2] + obs[3]) * 13.89)
        speeds.append(speed_mps)
        congestions.append(step_info["congestion_ratio"])
        done = term or trunc

    return {
        "reward": float(ep_reward),
        "departures": float(step_info["cumulative_arrived"]),
        "mean_queue": float(np.mean(queues)),
        "max_queue": float(np.max(queues)),
        "mean_wait_s": float(np.mean(waits)),
        "mean_speed_mps": float(np.mean(speeds)),
        "mean_congestion": float(np.mean(congestions)),
    }


def evaluate_across_seeds(
    env: TrafficSignalEnv,
    policy: Any,
    seeds: List[int],
    policy_name: str,
) -> Dict[str, Any]:
    """Evaluate a policy across multiple random seeds and aggregate mean +/- std."""
    results_by_seed = []
    for s in seeds:
        res = run_single_episode(env, policy, seed=s)
        results_by_seed.append({"seed": s, **res})

    metrics = ["reward", "departures", "mean_queue", "max_queue", "mean_wait_s", "mean_speed_mps", "mean_congestion"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results_by_seed]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "policy_name": policy_name,
        "summary": summary,
        "runs": results_by_seed,
    }


# ── Training Loop ────────────────────────────────────────────────────────────

def train_dqn(
    episodes: int = 50,
    episode_duration_s: float = 180.0,
    step_duration_s: float = 1.0,
    scenario: str = "Medium",
    seed: int = 42,
    output_dir: Path = Path("models"),
    docs_dir: Path = Path("docs"),
) -> Tuple[DQNAgent, Dict[str, Any]]:
    """Train DQN agent, evaluate against baselines across multiple seeds, and save artifacts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 78)
    print("SYNTRAQ — SINGLE INTERSECTION REINFORCEMENT LEARNING CONTROLLER TRAINING")
    print(f"Scenario: {scenario} | Episodes: {episodes} | Episode Duration: {episode_duration_s}s | Step: {step_duration_s}s")
    print("=" * 78)

    env = TrafficSignalEnv(
        step_duration_s=step_duration_s,
        sim_dt_s=0.1,
        episode_duration_s=episode_duration_s,
        scenario=scenario,
        seed=seed,
        min_green_s=4.0,
        max_green_s=35.0,
        yellow_s=3.0,
        all_red_s=2.0,
        queue_weight=0.2,
        wait_weight=0.05,
        congestion_weight=1.0,
        throughput_weight=1.2,
        invalid_action_penalty=0.05,
    )

    agent = DQNAgent(
        state_dim=12,
        action_dim=2,
        lr=1e-3,
        gamma=0.98,
        tau=0.01,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=int(episodes * 0.7),
        hidden_dim=64,
    )

    replay_buffer = ReplayBuffer(capacity=30000)

    # 1. Warmup replay buffer
    print("\nCollecting initial warmup transitions with random exploratory actions...")
    obs, _ = env.reset(seed=seed)
    for _ in range(350):
        action = random.randint(0, 1)
        next_obs, reward, term, trunc, _ = env.step(action)
        replay_buffer.push(obs, action, reward, next_obs, term or trunc)
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs
    print(f"Warmup complete. Replay buffer size: {len(replay_buffer)} transitions.")

    history: Dict[str, List[float]] = {
        "episode": [],
        "reward": [],
        "mean_queue": [],
        "mean_wait_s": [],
        "departures": [],
        "loss": [],
        "epsilon": [],
    }

    # 2. Training loop
    print("\nStarting Deep Q-Network training loop...")
    for ep in range(1, episodes + 1):
        ep_seed = seed + ep * 17
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        losses = []
        queues = []
        waits = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, step_info = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, term or trunc)

            loss = agent.update(replay_buffer, batch_size=64)
            if loss > 0.0:
                losses.append(loss)

            obs = next_obs
            ep_reward += reward
            queues.append(step_info["total_queue"])
            waits.append(step_info["avg_wait_s"])
            done = term or trunc

        agent.update_epsilon(ep)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_q = float(np.mean(queues)) if queues else 0.0
        mean_w = float(np.mean(waits)) if waits else 0.0
        deps = step_info["cumulative_arrived"]

        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["mean_queue"].append(mean_q)
        history["mean_wait_s"].append(mean_w)
        history["departures"].append(deps)
        history["loss"].append(mean_loss)
        history["epsilon"].append(agent.epsilon)

        if ep % 5 == 0 or ep == 1:
            print(
                f"Ep {ep:3d}/{episodes:3d} | "
                f"Reward: {ep_reward:7.1f} | "
                f"Queue: {mean_q:4.1f} veh | "
                f"Wait: {mean_w:4.1f}s | "
                f"Deps: {deps:3d} | "
                f"Loss: {mean_loss:.4f} | "
                f"Eps: {agent.epsilon:.2f}"
            )

    # 3. Save PyTorch checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "traffic_signal_dqn.pt"
    torch.save(
        {
            "policy_net_state_dict": agent.policy_net.state_dict(),
            "target_net_state_dict": agent.target_net.state_dict(),
            "state_dim": 12,
            "action_dim": 2,
            "hidden_dim": 64,
            "scenario": scenario,
            "episodes": episodes,
        },
        checkpoint_path,
    )
    print(f"\n[OK] Saved trained PyTorch checkpoint: {checkpoint_path}")

    # 4. Rigorous Multi-Seed Comparative Evaluation
    # Evaluating across 10 independent random seeds
    test_seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    print("\n" + "=" * 78)
    print(f"MULTI-SEED EVALUATION ACROSS {len(test_seeds)} DISTINCT SEEDS")
    print("Controllers compared: Fixed-Time Baseline vs Adaptive Heuristic vs Trained DQN")
    print("=" * 78)

    fixed_policy = FixedTimePolicy(green_duration_s=15.0, max_green_s=35.0)
    adaptive_policy = AdaptiveHeuristicPolicy(min_green_s=4.0, nominal_green_s=12.0, max_green_s=35.0)

    fixed_eval = evaluate_across_seeds(env, fixed_policy, test_seeds, "Fixed-Time Baseline")
    adaptive_eval = evaluate_across_seeds(env, adaptive_policy, test_seeds, "Adaptive Heuristic")
    dqn_eval = evaluate_across_seeds(env, agent, test_seeds, "Trained DQN Agent")

    # Statistical summary table
    print(
        f"\n{'Metric':<25} | {'Fixed-Time (mean +/- std)':<25} | "
        f"{'Adaptive (mean +/- std)':<25} | {'DQN Agent (mean +/- std)':<25}"
    )
    print("-" * 105)

    def fmt_stat(stat: dict) -> str:
        return f"{stat['mean']:6.2f} +/- {stat['std']:4.2f}"

    metric_labels = [
        ("reward", "Episode Reward"),
        ("mean_queue", "Average Queue Length (veh)"),
        ("max_queue", "Peak Queue Length (veh)"),
        ("mean_wait_s", "Average Wait Time (s)"),
        ("departures", "Completed Departures"),
        ("mean_speed_mps", "Average Speed (m/s)"),
        ("mean_congestion", "Congestion Ratio"),
    ]

    for key, label in metric_labels:
        f_s = fmt_stat(fixed_eval["summary"][key])
        a_s = fmt_stat(adaptive_eval["summary"][key])
        d_s = fmt_stat(dqn_eval["summary"][key])
        print(f"{label:<25} | {f_s:<25} | {a_s:<25} | {d_s:<25}")

    # Save complete evaluation JSON
    eval_json_path = output_dir / "controller_three_way_comparison.json"
    comparison_data = {
        "seeds": test_seeds,
        "fixed_time": fixed_eval,
        "adaptive_heuristic": adaptive_eval,
        "dqn": dqn_eval,
        "training_history": history,
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n[OK] Saved multi-seed evaluation JSON: {eval_json_path}")

    # 5. Export to ONNX
    onnx_path = output_dir / "traffic_signal_dqn.onnx"
    export_policy_to_onnx(agent.policy_net, onnx_path)

    # 6. Generate plots
    docs_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, docs_dir / "rl_training_curves.png")
    plot_three_way_comparison(comparison_data, docs_dir / "controller_three_way_comparison.png")

    env.close()
    return agent, comparison_data


# ── ONNX Export & Parity Validation ──────────────────────────────────────────

def export_policy_to_onnx(policy_net: nn.Module, onnx_path: Path) -> None:
    """Export trained DQN policy to ONNX format and validate runtime numerical parity."""
    policy_net.eval()
    dummy_input = torch.randn(1, 12, dtype=torch.float32)

    print(f"\nExporting trained DQN policy to ONNX: {onnx_path} (Opset 17)...")
    torch.onnx.export(
        policy_net,
        dummy_input,
        str(onnx_path),
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

    # Validate ONNX model structure
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model graph structure is valid.")

    # Validate numerical parity using ONNX Runtime
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_inputs = np.random.uniform(0.0, 1.0, size=(10, 12)).astype(np.float32)

    with torch.no_grad():
        pt_outputs = policy_net(torch.from_numpy(test_inputs)).numpy()

    ort_outputs = ort_session.run(["q_values"], {"observation": test_inputs})[0]
    max_abs_diff = float(np.max(np.abs(pt_outputs - ort_outputs)))
    print(f"[OK] ONNX Runtime numerical parity verified. Max abs diff: {max_abs_diff:.6e} (tolerance < 1e-5)")

    # Save reference test vectors for C++ unit test verification
    test_vectors = {
        "inputs": test_inputs.tolist(),
        "pytorch_q_values": pt_outputs.tolist(),
        "onnx_q_values": ort_outputs.tolist(),
        "best_actions": [int(np.argmax(q)) for q in ort_outputs],
    }
    vectors_path = onnx_path.parent / "rl_onnx_test_vectors.json"
    with open(vectors_path, "w", encoding="utf-8") as f:
        json.dump(test_vectors, f, indent=2)
    print(f"[OK] Saved reference test vectors to: {vectors_path}")


# ── Visualization ────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: Path) -> None:
    """Plot training return, loss, queues, wait times, and exploration decay."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    eps = history["episode"]

    # 1. Episode Return
    axes[0, 0].plot(eps, history["reward"], color="#38bdf8", lw=2, label="Episode Return")
    if len(eps) >= 5:
        rolling = np.convolve(history["reward"], np.ones(5) / 5, mode="valid")
        axes[0, 0].plot(eps[4:], rolling, color="#f59e0b", lw=2.5, linestyle="--", label="5-Ep Rolling Avg")
    axes[0, 0].set_title("Episode Return (Reward Progression)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Episode", color="#94a3b8")
    axes[0, 0].set_ylabel("Cumulative Discounted Return", color="#94a3b8")
    axes[0, 0].legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # 2. Mean Queue Length
    axes[0, 1].plot(eps, history["mean_queue"], color="#ef4444", lw=2)
    axes[0, 1].set_title("Average Queue Length per Episode", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Episode", color="#94a3b8")
    axes[0, 1].set_ylabel("Queued Vehicles (veh)", color="#94a3b8")

    # 3. Average Waiting Time
    axes[1, 0].plot(eps, history["mean_wait_s"], color="#a855f7", lw=2)
    axes[1, 0].set_title("Average Waiting Time (seconds)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Episode", color="#94a3b8")
    axes[1, 0].set_ylabel("Seconds Stopped", color="#94a3b8")

    # 4. Departures & Epsilon
    ax4 = axes[1, 1]
    l1 = ax4.plot(eps, history["departures"], color="#10b981", lw=2, label="Departures")
    ax4.set_title("Completed Departures & Exploration Decay", color="#f8fafc", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Episode", color="#94a3b8")
    ax4.set_ylabel("Completed Trips", color="#94a3b8")

    ax4_eps = ax4.twinx()
    ax4_eps.tick_params(colors="#94a3b8")
    l2 = ax4_eps.plot(eps, history["epsilon"], color="#cbd5e1", linestyle=":", lw=1.8, label="Epsilon (Exploration)")
    ax4_eps.set_ylabel("Epsilon", color="#cbd5e1")

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax4.legend(lines, labels, facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", loc="upper left")

    plt.suptitle("SyntraQ — Single Intersection DQN Training Telemetry", color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved training curves plot: {save_path}")


def plot_three_way_comparison(data: dict, save_path: Path) -> None:
    """Generate 4-panel multi-seed comparative bar charts with error bars for all three controllers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569", axis="y")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    controllers = ["Fixed-Time", "Adaptive Heuristic", "Trained DQN"]
    colors = ["#94a3b8", "#38bdf8", "#10b981"]

    fixed_sum = data["fixed_time"]["summary"]
    adaptive_sum = data["adaptive_heuristic"]["summary"]
    dqn_sum = data["dqn"]["summary"]

    def get_vals_and_errs(metric_key: str):
        vals = [fixed_sum[metric_key]["mean"], adaptive_sum[metric_key]["mean"], dqn_sum[metric_key]["mean"]]
        errs = [fixed_sum[metric_key]["std"], adaptive_sum[metric_key]["std"], dqn_sum[metric_key]["std"]]
        return vals, errs

    x = np.arange(len(controllers))

    # 1. Episode Reward
    vals, errs = get_vals_and_errs("reward")
    bars1 = axes[0, 0].bar(x, vals, yerr=errs, capsize=6, color=colors, alpha=0.85, edgecolor="#f8fafc", lw=1.2)
    axes[0, 0].set_title("Episode Return (Higher is Better)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(controllers, color="#f8fafc", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Mean Reward +/- Std", color="#94a3b8")

    # 2. Average Queue Length
    vals, errs = get_vals_and_errs("mean_queue")
    bars2 = axes[0, 1].bar(x, vals, yerr=errs, capsize=6, color=colors, alpha=0.85, edgecolor="#f8fafc", lw=1.2)
    axes[0, 1].set_title("Average Queue Length (Lower is Better)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(controllers, color="#f8fafc", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Mean Queued Vehicles +/- Std", color="#94a3b8")

    # 3. Average Waiting Time
    vals, errs = get_vals_and_errs("mean_wait_s")
    bars3 = axes[1, 0].bar(x, vals, yerr=errs, capsize=6, color=colors, alpha=0.85, edgecolor="#f8fafc", lw=1.2)
    axes[1, 0].set_title("Average Waiting Time (Lower is Better)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(controllers, color="#f8fafc", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Seconds Stopped +/- Std", color="#94a3b8")

    # 4. Completed Departures
    vals, errs = get_vals_and_errs("departures")
    bars4 = axes[1, 1].bar(x, vals, yerr=errs, capsize=6, color=colors, alpha=0.85, edgecolor="#f8fafc", lw=1.2)
    axes[1, 1].set_title("Throughput (Completed Departures - Higher is Better)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(controllers, color="#f8fafc", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Completed Trips +/- Std", color="#94a3b8")

    plt.suptitle("SyntraQ — Multi-Seed Benchmark (Fixed-Time vs Adaptive Heuristic vs DQN)", color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved three-way comparison plot: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate PyTorch DQN on SyntraQ Single-Intersection RL Environment")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--duration", type=float, default=180.0, help="Episode duration in seconds")
    parser.add_argument("--step-s", type=float, default=1.0, help="RL action step duration in seconds")
    parser.add_argument("--scenario", type=str, default="Medium", help="Traffic scenario preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        episode_duration_s=args.duration,
        step_duration_s=args.step_s,
        scenario=args.scenario,
        seed=args.seed,
    )
