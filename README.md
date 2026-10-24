# SyntraQ: Real-time AI-driven Urban Traffic Simulation & Optimization

A research-oriented C++20 simulation platform that models urban traffic and evaluates AI-based traffic management strategies.

![screenshot](docs/layout_1280x720.png)

---

## Status

> <u>**Note:**</u> SyntraQ is a research-oriented project and is currently **under active development**. Some features may be experimental or incomplete.

| System / Feature | Description | Status |
|---|---|---|
| Core Engine & Simulation Loop | Deterministic fixed-timestep ($\Delta t = 0.10\,\text{s}$), pause/resume, speed presets | ✅ Complete |
| Road Network Model | Directed multigraph topology, lane management, speed limits, geometric transforms | ✅ Complete |
| Vehicle Kinematics & Behavior | Car-following model, emergency braking, queue formation, turn slowdowns, collision avoidance | ✅ Complete |
| Visual UI & Layout | Three-region layout (Simulation Controls, Viewport, Traffic Telemetry & Signals) | ✅ Complete |
| Fixed-Time Traffic Signals | Configurable cycle splits, green/yellow/all-red phases, multi-intersection coordination | ✅ Complete |
| Routing Engine | Graph pathfinding with Dijkstra and A*, customizable heuristics, weighted travel costs | ✅ Complete |
| Telemetry & Metrics Engine | Decoupled tracking: throughput, waiting times, queue lengths, average speed, congestion % | ✅ Complete |
| Experiments & Batch Pipeline | Headless execution, multi-seed evaluation, automated statistical aggregation, CSV exports | ✅ Complete |
| Analysis & Visualization | Automated Python pipeline producing Pandas summaries and Matplotlib comparative charts | ✅ Complete |
| Data Generation & Preprocessing | Structured 26-feature dataset extraction, dual-horizon lookahead targets, leakage-free splitting | ✅ Complete |
| Classical Prediction Models | Baseline & ensemble models (Linear, Ridge, RF, GBDT) with R²=0.88, sub-millisecond inference | ✅ Complete |
| PyTorch Neural Predictor | 3-layer MLP, PyTorch DataLoader & training loop, checkpointing, R²=0.89 | ✅ Complete |
| Native AI Inference Engine | ONNX Runtime C++ API, feature scaler parity, async inference, latency telemetry | ✅ Complete |
| Adaptive Signal Optimization | Deterministic actuated/adaptive traffic signal controller (FixedTime vs. Adaptive) | ✅ Complete |
| Reinforcement Learning Environment | Single-intersection MDP, Gymnasium & PyTorch interface, DQN policy | ✅ Complete |
| RL Traffic Signal Controller | PyTorch Double DQN, multi-seed evaluation, ONNX export, C++ inference | ✅ Complete |

---

## Tech Stack

| Component      | Technology                     |
|----------------|--------------------------------|
| Core language  | C++20                          |
| Build system   | CMake 4.x + Ninja              |
| Rendering      | Raylib 6.0 (ARM64 MSVC)        |
| UI / Debug     | Dear ImGui + rlImGui           |
| Tests          | GoogleTest + CTest (163 tests) |
| CI             | GitHub Actions                 |
| Data Analysis  | Python 3 (Pandas, NumPy, Matplotlib) |
| Deep Learning  | PyTorch 2.x (CPU / CUDA)       |
| AI Inference   | ONNX Runtime C/C++ API         |

---

## Prerequisites

- **Windows ARM64** (tested) or x64 / Linux
- **Visual Studio 2022** with MSVC ARM64 toolchain (or GCC 13+ / Clang 17+)
- **CMake ≥ 3.25**
- **Ninja**
- **Git**
- Raylib 6.0 SDK
- **Python ≥ 3.10** with `pandas` and `matplotlib` (for experiment analysis)

---

## Building

Open a **Developer Command Prompt for VS 2022 (ARM64)** and run:

```powershell
# Configure (Debug)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build

# Run interactive simulation
.\build\bin\syntraq.exe
```

### Release build

```powershell
cmake -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel
```

### Custom Raylib SDK path

```powershell
cmake -B build -G Ninja -DRAYLIB_SDK_DIR="C:/path/to/raylib"
```

---

## Running Headless & CLI Flags

The simulator can run headlessly without Raylib/ImGui initialization, making it suitable for scripted benchmarks and automated pipelines:

```powershell
# Run headless simulation for 120 seconds with seed 42 and High traffic
.\build\bin\syntraq.exe --headless --duration 120 --scenario high --seed 42 --csv summary.csv

# Available CLI options:
#   --headless, -h           Run without GUI/rendering
#   --duration, -d <seconds> Simulation duration in seconds
#   --seed <number>          Random seed for deterministic generation
#   --scenario, -s <preset>  Traffic scenario preset: low, med, high, rush
#   --csv <path>             Export run summary metrics to CSV
#   --csv-timeseries <path>  Export periodic snapshot time-series to CSV
#   --screenshot <path>      Capture screenshot at start and exit
#   --debug-overlays         Enable technical debug overlays
```

---

## Baseline Experiments & Analysis Pipeline

SyntraQ includes an automated experiment suite to establish empirical performance baselines:

```powershell
# Run batch experiments (evaluates fixed-time controller across Low, Med, High, Rush Hour across multiple seeds)
.\build\bin\syntraq_experiments.exe

# Process results and generate publication-quality comparison charts
python scripts/analyze_baseline.py
```

Generated charts are saved to `data/experiments/charts/`:
- `baseline_metrics_dashboard.png` (Comprehensive 4-panel overview)
- `speed_and_throughput.png`
- `waiting_and_travel_time.png`
- `queues_and_congestion.png`

---

## Running Tests

SyntraQ maintains a test suite across kinematics, signal state machines, pathfinding algorithms, queue dynamics, metrics tracking, and training data generation:

```powershell
cd build
ctest --output-on-failure
```

Or execute targeted test suites directly:

```powershell
.\build\bin\test_rl_policy_inference.exe
.\build\bin\test_traffic_signal_env.exe
.\build\bin\test_adaptive_controller.exe
.\build\bin\test_traffic_predictor.exe
.\build\bin\test_data_generation.exe
.\build\bin\test_traffic_behavior.exe
.\build\bin\test_metrics.exe
.\build\bin\test_routing.exe
.\build\bin\test_signals.exe

# Run Python RL environment and PyTorch integration tests
python -m unittest tests/test_traffic_signal_env.py
```

---

## Machine Learning Data Generation & Preprocessing

SyntraQ provides a complete, leakage-free pipeline for training traffic prediction models:

```powershell
# 1. Generate multi-scenario raw traffic dataset with 15s and 30s future ground-truth targets
.\build\bin\syntraq_generate_dataset.exe --runs 5 --duration 120 --interval 1.0 --output data/dataset/raw_traffic_data.csv

# 2. Preprocess, standardize, and perform run-level leakage-free train/validation/test split
python scripts/preprocess_dataset.py --input data/dataset/raw_traffic_data.csv --output-dir data/dataset/processed
```

Generated dataset artifacts:
- `data/dataset/processed/train.csv`, `val.csv`, `test.csv`
- `data/dataset/processed/dataset.npz` (Ready-to-load NumPy matrices for PyTorch/TensorFlow)
- `data/dataset/processed/metadata.json` (Feature definitions, scaler statistics, and run audit)
- Full methodology documented in [docs/data_generation_and_features.md](docs/data_generation_and_features.md)

---

## Machine Learning & Neural Traffic Prediction

SyntraQ provides both classical regression baselines and a PyTorch neural network to predict future queue lengths and congestion from 26 state telemetry features:

```powershell
# 1. Train and benchmark classical models (Linear Regression, Ridge, Random Forest, Gradient Boosting)
python scripts/train_prediction_model.py --target target_queue_15s --seed 42

# 2. Train and evaluate PyTorch TrafficMLP neural network (with DataLoader, checkpointing, and scheduler)
python scripts/train_pytorch_mlp.py --epochs 120 --batch-size 32 --lr 0.003
```

### Model Architecture (PyTorch TrafficMLP)

```
Input (26 features) --> Dense (64) --> ReLU --> Dense (32) --> ReLU --> Output (1 scalar)
```
- Total trainable parameters: **3,841**
- Checkpoint size: **20.3 KB** (30x smaller than tree ensemble bundles)

### Comprehensive Model Benchmark Results (+15s Future Queue Length)

| Model Architecture | Type | Val MAE | Val RMSE | Val R^2 | Test MAE | Test RMSE | Test R^2 | Max Error | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Mean)** | Baseline | 6.40 veh | 8.91 veh | -0.0004 | 5.94 veh | 7.80 veh | -0.0024 | 35.73 veh | 0.03 us |
| **Linear Regression** | Classical | 2.90 veh | 3.85 veh | 0.8135 | 2.92 veh | 3.83 veh | 0.7585 | 13.46 veh | 0.21 us |
| **Ridge Regression** | Classical | 2.90 veh | 3.85 veh | 0.8131 | 2.91 veh | 3.81 veh | 0.7612 | 12.93 veh | 0.11 us |
| **Random Forest** | Classical | 2.26 veh | 3.46 veh | 0.8494 | 2.15 veh | 3.09 veh | 0.8431 | 12.77 veh | 33.65 us |
| **Gradient Boosting** | Classical | 2.09 veh | 3.07 veh | 0.8812 | 1.90 veh | 2.72 veh | 0.8779 | 10.80 veh | 1.91 us |
| **PyTorch TrafficMLP (Best)** | **Neural Net** | **1.79 veh** | **2.40 veh** | **0.9272** | **1.75 veh** | **2.56 veh** | **0.8919** | **9.94 veh** | **2.45 us** |

### Artifacts & Reports
- **PyTorch Model Checkpoint**: `models/best_traffic_mlp.pt`
- **PyTorch Metrics JSON**: `models/mlp_evaluation_metrics.json`
- **Classical Model Bundle**: `models/best_traffic_predictor.joblib`
- **Classical Metrics JSON**: `models/evaluation_metrics.json`
- **Visual Diagnostics**: [docs/mlp_training_and_evaluation.png](docs/mlp_training_and_evaluation.png)
- **Detailed Evaluation & Trade-off Analysis**: [docs/mlp_evaluation_report.md](docs/mlp_evaluation_report.md)

---

## Native AI Inference Engine (ONNX Runtime)

SyntraQ integrates the **ONNX Runtime C/C++ API** for low-latency, real-time neural inference inside the simulation engine without Python dependencies:

```powershell
# 1. Export trained PyTorch TrafficMLP to optimized ONNX graph (Opset 17)
python scripts/export_onnx_model.py

# 2. Run automated C++ inference parity and latency tests
.\build\bin\test_traffic_predictor.exe
```

### Architecture & Preprocessing Parity
- **Zero-Copy Continuous Scaling**: C++ `TrafficFeatureScaler` mirrors the Python training pipeline, standardizing 22 continuous features using precomputed dataset mean/std statistics.
- **Categorical One-Hot Encoding**: Encodes the 4 simulation scenario regimes (`Low Traffic`, `Medium Traffic`, `High Traffic`, `Rush Hour`) into indices [22:25] matching Python's `OneHotEncoder`.
- **C++ Inference Interface (`TrafficPredictor`)**:
  - `loadModel(const std::string& model_path)`: Thread-safe ONNX model initialization with graceful error handling (no crashing on missing/corrupt models).
  - `predict(const std::vector<float>& features)`: Synchronous native tensor evaluation returning future queue length.
  - `predictSample(const TrafficDataSample& sample)`: High-level helper combining extraction, scaling, and inference.
  - `predictAsync(const std::vector<float>& features)`: Non-blocking asynchronous inference returning `std::future<float>` so the rendering/simulation loop is never stalled.
  - `isLoaded()`: Read-only model readiness indicator.
  - Telemetry: `getLastInferenceLatencyUs()`, `getAverageInferenceLatencyUs()`, and `getInferenceCount()` for microsecond profiling.
- **Python-to-C++ Parity Validation**: Verified across reference test vectors with maximum numerical discrepancy **$< 10^{-5}$** (well within test tolerance of $10^{-3}$).

---

## Adaptive Traffic Signal Control

SyntraQ provides both a baseline **Fixed-Time Controller** and a deterministic **Adaptive Controller** sharing a unified `ITrafficSignalController` interface:

```powershell
# 1. Run headless simulation using adaptive traffic signal controllers
.\build\bin\syntraq.exe --headless --controller adaptive --duration 120 --scenario high

# 2. Run automated comparative benchmark across all scenarios and seeds
.\build\bin\syntraq_compare_controllers.exe --duration 120

# 3. Generate comparative charts and performance analysis
python scripts/analyze_adaptive_comparison.py
```

### Deterministic & Explainable Policy
- **Interface Uniformity**: Both `FixedTimeController` and `AdaptiveController` implement `ITrafficSignalController`.
- **Dynamic Queue-Responsive Decisions**:
  - `ExtendGreen`: Extends green while queue demand persists ($Q_{\text{current}} > 0$) up to `max_green_s` (30s) as long as opposing approaches have not exceeded starvation thresholds.
  - `ShortenPhase (Early Gap-Out)`: Cuts green time short once active queues dissipate ($Q_{\text{current}} = 0$) after mandatory `min_green_s` (4s), transitioning directly to yellow clearance to serve waiting opposing queues.
  - `SwitchPhase`: Triggered by reaching `max_green_s` or when opposing vehicles reach the starvation threshold (`max_wait_threshold_s = 20s`).
  - `Hold`: Guarantees mandatory driver predictability and minimum green safety times.
- **Empirical Performance Gains**:
  - **-14.6% waiting time reduction** and **+15.1% throughput improvement** under Rush Hour traffic.
  - **-12.5% waiting time reduction** and **+13.7% throughput improvement** under High traffic.
  - **-14.3% congestion reduction** under High traffic.
  - Detailed empirical benchmark breakdown documented in [docs/adaptive_controller_benchmark.md](docs/adaptive_controller_benchmark.md).

---

## Reinforcement Learning Environment (Single-Intersection)

SyntraQ provides a dedicated Gymnasium-compatible Reinforcement Learning (RL) environment for traffic signal control. It interfaces directly with native C++ simulation binaries via an in-memory C-ABI (`syntraq_env_capi`), achieving sub-millisecond step execution speeds without IPC or file I/O overhead.

### Markov Decision Process (MDP) Specifications

- **Observation Space**: `Box(12,)` bounded in `[0.0, 1.0]` capturing normalized queue lengths, approaching moving vehicles, average speeds per approach, active phase index, time in current phase, clearance status, switch legality, and average waiting times.
- **Action Space**: `Discrete(2)`:
  - `0` (`MaintainPhase`): Keep active green or continue ongoing clearance stage.
  - `1` (`SwitchPhase`): Initiate safe transition to next phase (enforcing minimum green time $t_{\text{min\_green}} \ge 4.0\text{s}$, 3.0s yellow clearance, and 2.0s all-red buffer).
- **Reward Function**: Penalizes queue lengths, waiting times, and network congestion while rewarding completed departures:
  $$R_t = -\Big( w_q \cdot Q_t + w_w \cdot W_t + w_c \cdot C_t \Big) + w_{\text{thr}} \cdot \Delta \text{Departures}_t$$

### Python / PyTorch Integration & Training

```powershell
# 1. Run Python environment & PyTorch integration tests
python -m unittest tests/test_traffic_signal_env.py

# 2. Train Deep Q-Network (DQN) agent and run 10-seed multi-benchmark
python scripts/train_rl_agent.py --episodes 50 --duration 180.0 --scenario Medium

# 3. Export trained DQN policy to ONNX with numerical validation
python scripts/export_rl_onnx.py

# 4. Execute C++ native ONNX inference validation test
.\build\bin\test_rl_policy_inference.exe
```

### Multi-Seed Benchmark Across 10 Independent Seeds ($N = 10$)

To guarantee statistical validity without claiming superiority on a single isolated run, all three controllers were evaluated on 10 independent random seeds (`42, 101, 202, 303, 404, 505, 606, 707, 808, 909`) under Medium traffic:

| Performance Metric | Fixed-Time Baseline | Adaptive Heuristic | Trained DQN Policy | DQN vs Fixed-Time ($\Delta$) | DQN vs Adaptive ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Episode Return (Reward)** | $-370.83 \pm 104.33$ | $-416.97 \pm 110.06$ | **$-162.36 \pm 21.62$** | **$+56.2\%$** | **$+61.1\%$** |
| **Average Queue Length** | $6.66 \pm 1.66\,\text{veh}$ | $7.24 \pm 1.68\,\text{veh}$ | **$3.32 \pm 0.35\,\text{veh}$** | **$-50.1\%$** | **$-54.1\%$** |
| **Peak Queue Length** | $18.10 \pm 3.38\,\text{veh}$ | $17.40 \pm 3.24\,\text{veh}$ | **$8.40 \pm 1.17\,\text{veh}$** | **$-53.6\%$** | **$-51.7\%$** |
| **Average Waiting Time** | $14.08 \pm 3.58\,\text{s}$ | $16.18 \pm 4.48\,\text{s}$ | **$6.88 \pm 0.86\,\text{s}$** | **$-51.1\%$** | **$-57.5\%$** |
| **Completed Departures** | $48.70 \pm 3.23\,\text{trips}$ | $48.50 \pm 4.01\,\text{trips}$ | **$53.50 \pm 2.37\,\text{trips}$** | **$+9.9\%$** | **$+10.3\%$** |
| **Average Approach Speed** | $3.20 \pm 0.09\,\text{m/s}$ | $3.17 \pm 0.09\,\text{m/s}$ | **$3.29 \pm 0.05\,\text{m/s}$** | **$+2.8\%$** | **$+3.8\%$** |
| **Congestion Ratio** | $0.35 \pm 0.06$ | $0.38 \pm 0.05$ | **$0.24 \pm 0.02$** | **$-31.4\%$** | **$-36.8\%$** |

### Key Findings & Variance Analysis
- **$4.8\times$ Variance Compression**: DQN achieves $\sigma_{\text{reward}} = 21.62$ compared to $104.33$ for Fixed-Time and $110.06$ for Adaptive, demonstrating high behavioral stability across variable traffic arrival profiles.
- **Queue Surge Suppression**: Peak queue surges are cut by **$-53.6\%$** ($18.1 \to 8.4\,\text{veh}$) as the network anticipates approaching platoons before queues accumulate.
- **Native ONNX Deployment**: Exported to `traffic_signal_dqn.onnx` (Opset 17) with **$< 15\,\mu\text{s}$** inference latency via C++ `RLPolicyInference` and exact float parity ($0.0$ max discrepancy).

### Artifacts & Reports
- **PyTorch Model Checkpoint**: `models/traffic_signal_dqn.pt`
- **Exported ONNX Model**: `models/traffic_signal_dqn.onnx`
- **Multi-Seed Benchmark JSON**: `models/controller_three_way_comparison.json`
- **C++ Reference Test Vectors**: `models/rl_onnx_test_vectors.json`
- **Diagnostic Visualizations**:
  - Training Telemetry: [docs/rl_training_curves.png](docs/rl_training_curves.png)
  - Multi-Seed 4-Panel Comparison: [docs/controller_three_way_comparison.png](docs/controller_three_way_comparison.png)
- **Detailed Evaluation Report**: [docs/rl_controller_evaluation.md](docs/rl_controller_evaluation.md)
- **Environment Design Document**: [docs/rl_environment_design.md](docs/rl_environment_design.md)

---

## Project Structure

```
SyntraQ/
├── include/syntraq/
│   ├── core/          # Config and shared types (.hpp)
│   ├── world/         # Road network graph (Intersections & Roads) (.hpp)
│   ├── vehicles/      # Vehicle kinematics, car-following, and spawner (.hpp)
│   ├── signals/       # Traffic signals (Interface, FixedTime, Adaptive, RL) (.hpp)
│   ├── routing/       # Routing engine (Dijkstra, A*, Heuristics, Costs) (.hpp)
│   ├── metrics/       # Telemetry & ML dataset collectors (.hpp)
│   ├── ai/            # Predictor, feature scaler, RL env, C-ABI, RLPolicyInference (.hpp / .h)
│   ├── simulation/    # Headless simulation loop & state (.hpp)
│   └── rendering/     # Raylib + ImGui renderer (.hpp)
├── src/
│   ├── core/          # Config implementation
│   ├── world/         # Road network implementation & single-intersection topology
│   ├── vehicles/      # Kinematics and queue formation systems
│   ├── signals/       # FixedTimeController, AdaptiveController, RLSignalController
│   ├── routing/       # Dijkstra, A*, and graph generator
│   ├── metrics/       # Telemetry collector, ML data generator, CSV exporters
│   ├── ai/            # Predictor, FeatureScaler, TrafficSignalEnv, env_c_api, RLPolicyInference
│   ├── simulation/    # Simulation loop & scenario management
│   ├── rendering/     # Renderer and three-panel UI layout
│   └── main.cpp       # CLI parsing and application entry point
├── experiments/       # Baseline experiments, controller comparison, and dataset generator
├── scripts/           # ML training, DQN pipeline, ONNX exporter, Gymnasium env
├── tests/             # GoogleTest & Python test suites (163 C++ tests + 6 Python tests)
├── benchmarks/        # Routing engine benchmarks (Dijkstra vs A*)
├── configs/           # Scenario JSON configurations
├── data/              # Experiment datasets, ML training data, and reference validation sets
├── assets/            # Fonts, textures (future)
├── models/            # Trained PyTorch, DQN, classical, and ONNX models (.pt, .onnx, .joblib)
└── docs/              # Layout screenshots, dataset specs, benchmarks, and evaluation reports
```

---

## Architecture Principles

1. `syntraq_core` (static lib) — Headless, zero graphics dependencies, 100% testable via unit tests and headless runners.
2. `syntraq_renderer` — All Raylib and ImGui rendering calls isolated here; consumes simulation state strictly read-only.
3. `syntraq_experiments` — Headless batch execution harness for multi-seed statistical evaluations.
4. `syntraq` — Lightweight CLI entry point coordinating configuration, simulation, and optional rendering.
5. Deterministic Simulation — Random seeds strictly govern vehicle routes, spawning intervals, and behavior.
6. Modular Routing Engine — Pathfinding isolated behind the `IRouteProvider` / `IRouter` abstraction (supporting A*, Dijkstra, and greedy heuristics).
7. Independent Telemetry — Performance metrics (`TrafficMetricsCollector`) compute statistics without rendering dependencies.

---

## Simulation Controls & UI Layout

### Three-Region Interface
- **Left Panel ("SyntraQ Simulation")**: Simulation controls (Play/Pause, Reset, Speed Presets), traffic scenario selector (Low, Med, High, Rush Hour), seed configuration, and road network statistics.
- **Center Viewport**: Real-time traffic rendering with dynamic camera controls and vehicle visualizations.
- **Right Panel ("SyntraQ Traffic")**: Live performance telemetry (Average Speed, Waiting Time, Queue Lengths, Throughput, Congestion Ratio) and live signal phase status for each junction.

### Camera Controls
- **Mouse Wheel**: Smooth zoom towards cursor position (0.25× – 4.0×).
- **Middle Mouse Drag** or **Right Mouse Drag**: Pan the viewport.
- **R Key** or **Reset View Button**: Frame and center the road network within the active viewport.

### Simulation Controls
- **Space** or **Pause / Resume Button**: Toggle simulation execution.
- **Reset Sim Button**: Resets simulation clock, vehicles, and network to $t=0$.
- **Speed Multipliers**: `0.25×`, `0.5×`, `1×`, `2×`, `5×` presets.

### Debug Visualizations (Toggleable)
- **Intersection IDs**: Technical ID badges on junction nodes.
- **Road IDs**: Segment ID labels centered along road links.
- **Lane Boundaries**: Visible lane boundary guidelines.
- **Vehicle Direction Vectors**: Forward heading vectors from vehicles.

### Fixed-Timestep Architecture
Simulation advancement is decoupled from rendering frame rate:
$$\text{Real elapsed time} \xrightarrow{} \text{Accumulator} \xrightarrow{} \text{Fixed timestep } (\Delta t = 0.10\,\text{s}) \xrightarrow{} \text{Simulation updates} \xrightarrow{} \text{Rendering}$$

Guarantees bit-identical, reproducible simulation progression regardless of display refresh rate or headless execution.

---

## License

This project uses the GNU GPL3.0 license. Refer to [LICENSE](LICENSE) for more details.
