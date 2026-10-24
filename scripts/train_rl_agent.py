"""
scripts/train_rl_agent.py

Reinforcement Learning Training Pipeline for Single-Intersection Traffic Signal Optimization.
Implements Deep Q-Network (DQN) in PyTorch to train an agent in the SyntraQ environment:
  - Observation: 12-dimensional normalized traffic state
  - Action Space: Discrete(2) [0: Maintain current phase, 1: Switch phase]
  - Reward: Multi-objective penalizing queues, wait time, congestion, and rewarding throughput
  - Experience Replay Buffer & Target Network with Polyak soft updates
  - Evaluates trained policy against Fixed-Time baseline on identical traffic seeds
  - Generates training performance curves and saves trained PyTorch checkpoint
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.traffic_signal_env import TrafficSignalEnv


# ── DQN Neural Network Architecture ──────────────────────────────────────────

class DQN(nn.Module):
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
    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── Agent ────────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.98,
        tau: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 35,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        self.policy_net = DQN(state_dim, action_dim, hidden_dim=64)
        self.target_net = DQN(state_dim, action_dim, hidden_dim=64)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(s_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> float:
        if len(replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Q(s, a)
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
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

    def update_epsilon(self, episode_idx: int):
        frac = min(1.0, episode_idx / max(1, self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


# ── Evaluation Function ──────────────────────────────────────────────────────

def evaluate_policy(env: TrafficSignalEnv, agent: Optional[DQNAgent], num_episodes: int = 5, seed_base: int = 500) -> dict:
    """
    Evaluate policy (or Fixed-Time if agent is None) over several test episodes.
    """
    total_rewards = []
    total_departures = []
    avg_queues = []
    avg_waits = []
    avg_congestions = []

    for ep in range(num_episodes):
        ep_seed = seed_base + ep * 13
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        q_hist = []
        w_hist = []
        c_hist = []

        step_idx = 0
        while not done:
            if agent is not None:
                action = agent.select_action(obs, evaluate=True)
            else:
                # Baseline Fixed-Time simulation: Maintain (0) for 10s, then Switch (1)
                # Matches standard 10s green cycle
                action = 1 if (step_idx % 15 == 0 and step_idx > 0) else 0

            obs, reward, term, trunc, step_info = env.step(action)
            ep_reward += reward
            q_hist.append(step_info["total_queue"])
            w_hist.append(step_info["avg_wait_s"])
            c_hist.append(step_info["congestion_ratio"])
            step_idx += 1
            done = term or trunc

        total_rewards.append(ep_reward)
        total_departures.append(step_info["cumulative_arrived"])
        avg_queues.append(np.mean(q_hist))
        avg_waits.append(np.mean(w_hist))
        avg_congestions.append(np.mean(c_hist))

    return {
        "mean_reward": float(np.mean(total_rewards)),
        "mean_departures": float(np.mean(total_departures)),
        "mean_queue": float(np.mean(avg_queues)),
        "mean_wait_s": float(np.mean(avg_waits)),
        "mean_congestion": float(np.mean(avg_congestions)),
    }


# ── Main Training Loop ───────────────────────────────────────────────────────

def train(
    episodes: int = 50,
    episode_duration_s: float = 180.0,
    step_duration_s: float = 1.0,
    scenario: str = "Medium",
    seed: int = 42,
    output_dir: Path = Path("models"),
    docs_dir: Path = Path("docs"),
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 70)
    print("SYNTRAQ — SINGLE INTERSECTION REINFORCEMENT LEARNING TRAINING")
    print(f"Scenario: {scenario} | Episodes: {episodes} | Episode Duration: {episode_duration_s}s")
    print("=" * 70)

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
    )

    replay_buffer = ReplayBuffer(capacity=25000)

    history = {
        "episode": [],
        "reward": [],
        "mean_queue": [],
        "mean_wait_s": [],
        "departures": [],
        "loss": [],
        "epsilon": [],
    }

    # Warmup buffer with random exploration
    print("\nCollecting initial warmup experiences...")
    obs, _ = env.reset(seed=seed)
    for _ in range(300):
        action = random.randint(0, 1)
        next_obs, reward, term, trunc, _ = env.step(action)
        replay_buffer.push(obs, action, reward, next_obs, term or trunc)
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs
    print(f"Replay buffer warmup completed: {len(replay_buffer)} transitions.")

    # Training episodes
    print("\nStarting DQN training...")
    for ep in range(1, episodes + 1):
        ep_seed = seed + ep * 17
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        ep_losses = []
        q_list = []
        w_list = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, step_info = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, term or trunc)

            loss = agent.update(replay_buffer, batch_size=64)
            if loss > 0.0:
                ep_losses.append(loss)

            obs = next_obs
            ep_reward += reward
            q_list.append(step_info["total_queue"])
            w_list.append(step_info["avg_wait_s"])
            done = term or trunc

        agent.update_epsilon(ep)

        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        mean_q = float(np.mean(q_list)) if q_list else 0.0
        mean_w = float(np.mean(w_list)) if w_list else 0.0
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
                f"Mean Queue: {mean_q:4.1f} | "
                f"Mean Wait: {mean_w:4.1f}s | "
                f"Departures: {deps:3d} | "
                f"Loss: {mean_loss:.4f} | "
                f"Eps: {agent.epsilon:.2f}"
            )

    # Save model weights
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "traffic_signal_dqn.pt"
    torch.save(
        {
            "policy_net_state_dict": agent.policy_net.state_dict(),
            "target_net_state_dict": agent.target_net.state_dict(),
            "state_dim": 12,
            "action_dim": 2,
            "scenario": scenario,
            "episodes": episodes,
        },
        model_path,
    )
    print(f"\nSaved trained DQN model checkpoint to: {model_path}")

    # Evaluate trained RL Agent vs Baseline Fixed-Time Controller
    print("\n" + "=" * 70)
    print("EVALUATION — TRAINED RL AGENT VS FIXED-TIME BASELINE")
    print("=" * 70)

    rl_eval = evaluate_policy(env, agent, num_episodes=10, seed_base=900)
    fixed_eval = evaluate_policy(env, None, num_episodes=10, seed_base=900)

    print(f"{'Metric':<25} | {'Fixed-Time Baseline':<20} | {'Trained RL Agent':<20} | {'Delta (%)':<10}")
    print("-" * 80)

    def print_metric(name, f_val, r_val, lower_is_better=True):
        denom = max(1e-4, abs(f_val))
        delta = ((r_val - f_val) / denom) * 100.0
        delta_str = f"{delta:+.1f}%"
        print(f"{name:<25} | {f_val:<20.2f} | {r_val:<20.2f} | {delta_str:<10}")

    print_metric("Episode Reward", fixed_eval["mean_reward"], rl_eval["mean_reward"], lower_is_better=False)
    print_metric("Completed Departures", fixed_eval["mean_departures"], rl_eval["mean_departures"], lower_is_better=False)
    print_metric("Average Queue Length", fixed_eval["mean_queue"], rl_eval["mean_queue"], lower_is_better=True)
    print_metric("Average Waiting Time (s)", fixed_eval["mean_wait_s"], rl_eval["mean_wait_s"], lower_is_better=True)
    print_metric("Congestion Ratio", fixed_eval["mean_congestion"], rl_eval["mean_congestion"], lower_is_better=True)

    # Save evaluation summary to JSON
    eval_json_path = output_dir / "rl_evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({"rl": rl_eval, "fixed_time": fixed_eval, "history": history}, f, indent=2)
    print(f"Saved evaluation results to: {eval_json_path}")

    # Plot training curves
    docs_dir.mkdir(parents=True, exist_ok=True)
    plot_path = docs_dir / "rl_training_curves.png"
    generate_training_plots(history, plot_path)
    print(f"Saved training curves plot to: {plot_path}")

    env.close()
    return rl_eval, fixed_eval


def generate_training_plots(history: dict, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    eps = history["episode"]

    # 1. Episode Reward
    axes[0, 0].plot(eps, history["reward"], color="#38bdf8", lw=2, label="Episode Return")
    # Rolling mean
    if len(eps) >= 5:
        rolling = np.convolve(history["reward"], np.ones(5) / 5, mode="valid")
        axes[0, 0].plot(eps[4:], rolling, color="#f59e0b", lw=2.5, linestyle="--", label="5-Ep Rolling Avg")
    axes[0, 0].set_title("Episode Return (Reward Progression)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Episode", color="#94a3b8")
    axes[0, 0].set_ylabel("Total Cumulative Reward", color="#94a3b8")
    axes[0, 0].legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # 2. Mean Queue Length
    axes[0, 1].plot(eps, history["mean_queue"], color="#ef4444", lw=2)
    axes[0, 1].set_title("Average Queue Length per Episode", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Episode", color="#94a3b8")
    axes[0, 1].set_ylabel("Queued Vehicles", color="#94a3b8")

    # 3. Average Waiting Time
    axes[1, 0].plot(eps, history["mean_wait_s"], color="#a855f7", lw=2)
    axes[1, 0].set_title("Average Waiting Time (seconds)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Episode", color="#94a3b8")
    axes[1, 0].set_ylabel("Seconds Stopped", color="#94a3b8")

    # 4. Total Departures & Epsilon
    ax4 = axes[1, 1]
    l1 = ax4.plot(eps, history["departures"], color="#10b981", lw=2, label="Departures")
    ax4.set_title("Vehicle Departures & Exploration Decay", color="#f8fafc", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Episode", color="#94a3b8")
    ax4.set_ylabel("Completed Trips", color="#94a3b8")

    ax4_eps = ax4.twinx()
    ax4_eps.tick_params(colors="#94a3b8")
    l2 = ax4_eps.plot(eps, history["epsilon"], color="#cbd5e1", linestyle=":", lw=1.8, label="Epsilon (Exploration)")
    ax4_eps.set_ylabel("Epsilon", color="#cbd5e1")

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax4.legend(lines, labels, facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", loc="upper left")

    plt.suptitle("SyntraQ — Single Intersection RL (DQN) Training Telemetry", color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch DQN on SyntraQ Single-Intersection RL Environment")
    parser.add_argument("--episodes", type=int, default=40, help="Number of training episodes")
    parser.add_argument("--duration", type=float, default=150.0, help="Episode duration in seconds")
    parser.add_argument("--step-s", type=float, default=1.0, help="RL action step duration in seconds")
    parser.add_argument("--scenario", type=str, default="Medium", help="Traffic scenario preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        episode_duration_s=args.duration,
        step_duration_s=args.step_s,
        scenario=args.scenario,
        seed=args.seed,
    )
