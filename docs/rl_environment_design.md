# Reinforcement Learning Environment for Traffic Signal Control

This document details the formulation, architecture, and evaluation of the single-intersection traffic signal reinforcement learning (RL) environment in SyntraQ.

---

## 1. Markov Decision Process (MDP) Formulation

The traffic signal optimization problem for a single intersection is framed as a discrete-time Markov Decision Process $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

### State Space $\mathcal{S}$ (Compact Observation Vector)

The observation vector is a compact, normalized 12-dimensional array $\mathbf{s}_t \in [0.0, 1.0]^{12}$:

| Index | Feature | Description | Normalization |
| :---: | :--- | :--- | :--- |
| `0` | `queue_phase_0` | Queued vehicles on Phase 0 (North-South) approaches | $\min(1.0, Q_0 / 20)$ |
| `1` | `queue_phase_1` | Queued vehicles on Phase 1 (East-West) approaches | $\min(1.0, Q_1 / 20)$ |
| `2` | `approaching_phase_0` | Moving approaching vehicles on Phase 0 | $\min(1.0, V_{\text{app},0} / 20)$ |
| `3` | `approaching_phase_1` | Moving approaching vehicles on Phase 1 | $\min(1.0, V_{\text{app},1} / 20)$ |
| `4` | `avg_speed_phase_0` | Mean vehicle speed on Phase 0 approaches | $v_0 / v_{\text{limit}}$ |
| `5` | `avg_speed_phase_1` | Mean vehicle speed on Phase 1 approaches | $v_1 / v_{\text{limit}}$ |
| `6` | `current_phase` | Active signal phase index | $0.0 = \text{NS}, 1.0 = \text{EW}$ |
| `7` | `time_in_phase` | Elapsed time in current stage | $\min(1.0, t_{\text{elapsed}} / t_{\text{max\_green}})$ |
| `8` | `is_clearing` | Clearance stage flag (Yellow or All-Red) | $1.0 = \text{Transition}, 0.0 = \text{Green}$ |
| `9` | `can_switch` | Switch feasibility flag ($t_{\text{green}} \ge t_{\text{min\_green}}$) | $1.0 = \text{Allowed}, 0.0 = \text{Locked}$ |
| `10` | `avg_wait_phase_0` | Mean waiting time of queued vehicles on Phase 0 | $\min(1.0, W_0 / 60\text{s})$ |
| `11` | `avg_wait_phase_1` | Mean waiting time of queued vehicles on Phase 1 | $\min(1.0, W_1 / 60\text{s})$ |

### Action Space $\mathcal{A}$

The action space is discrete with two decisions: $\mathcal{A} = \{0, 1\}$:

- **Action `0` (`MaintainPhase`):** Keep the current green phase running, or continue clearance if in transition.
- **Action `1` (`SwitchPhase`):** Initiate phase change. If currently serving green and $t_{\text{elapsed}} \ge t_{\text{min\_green}}$, transitions to yellow clearance.

### Safety Guarantees & Clearance Protocol

The controller guarantees physical safety:
1. **Mandatory Minimum Green ($t_{\text{min\_green}} = 4.0\text{s}$):** Action `1` is safely ignored or held if the current green phase has not met minimum service time.
2. **Yellow Clearance ($t_{\text{yellow}} = 3.0\text{s}$):** Conflicting green is never displayed immediately; vehicles approaching are given clearance warning.
3. **All-Red Buffer ($t_{\text{all\_red}} = 2.0\text{s}$):** Intersection is completely cleared before the opposing phase turns green.
4. **Failsafe Maximum Green ($t_{\text{max\_green}} = 45.0\text{s}$):** Prevents starvation deadlock if an agent maintains a phase indefinitely.

### Reward Function $\mathcal{R}$

The reward penalizes delay, queue buildup, and network congestion while rewarding completed throughput:

$$R_t = -\Big( w_q \cdot Q_t + w_w \cdot W_t + w_c \cdot C_t \Big) + w_{\text{thr}} \cdot \Delta \text{Departures}_t$$

where:
- $Q_t = q_{\text{NS}} + q_{\text{EW}}$ is total queued vehicles.
- $W_t$ is mean stopped waiting time of queued vehicles.
- $C_t = \frac{Q_t}{\max(1, Q_t + V_{\text{approaching}})}$ is the instantaneous congestion ratio.
- $\Delta \text{Departures}_t$ is the count of vehicles that reached destinations during step interval $\Delta t_{\text{step}}$.
- Configurable weights: $w_q = 0.2$, $w_w = 0.05$, $w_c = 1.0$, $w_{\text{thr}} = 1.2$.

---

## 2. Architecture & Interop

```
┌────────────────────────────────────────────────────────┐
│                   Python / PyTorch                     │
│  - Deep Q-Network (DQN) / Policy Gradient Agent        │
│  - Experience Replay & Target Network                  │
│  - Gymnasium Environment Interface (TrafficSignalEnv)  │
└──────────────────────────┬─────────────────────────────┘
                           │  ctypes (Zero-Copy C ABI)
                           ▼
┌────────────────────────────────────────────────────────┐
│             syntraq_env_capi (Shared DLL)              │
│  - syntraq_env_create, syntraq_env_reset               │
│  - syntraq_env_step, syntraq_env_get_info              │
└──────────────────────────┬─────────────────────────────┘
                           │  C++ Engine Interface
                           ▼
┌────────────────────────────────────────────────────────┐
│                   SyntraQ Core (C++20)                 │
│  - TrafficSignalEnv (MDP Engine)                       │
│  - RLSignalController (ITrafficSignalController)       │
│  - Simulation (Headless Discrete Timestep Loop)        │
│  - RoadNetwork (make_single_intersection_map)          │
└────────────────────────────────────────────────────────┘
```

---

## 3. Training & Evaluation Pipeline

The environment is verified on a canonical 4-way isolated intersection (`make_single_intersection_map`) where 100% of vehicles interact with the signal controller.

### Running Training

```bash
python scripts/train_rl_agent.py --episodes 40 --duration 180.0 --scenario Medium
```

### Running Unit & Parity Tests

```bash
# C++ Test Suite (158 tests)
ctest --test-dir build --output-on-failure

# Python Gymnasium & PyTorch Compatibility Tests
python tests/test_traffic_signal_env.py
```
