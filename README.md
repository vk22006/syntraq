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
| Adaptive & AI Signal Optimization | Reinforcement Learning & Neural Network adaptive traffic signal controllers | ⏳ Planned |

---

## Tech Stack

| Component      | Technology                     |
|----------------|--------------------------------|
| Core language  | C++20                          |
| Build system   | CMake 4.x + Ninja              |
| Rendering      | Raylib 6.0 (ARM64 MSVC)        |
| UI / Debug     | Dear ImGui + rlImGui           |
| Tests          | GoogleTest + CTest (124 tests) |
| CI             | GitHub Actions                 |
| Data Analysis  | Python 3 (Pandas, Matplotlib)  |
| AI deployment  | ONNX Runtime (future)          |
| AI training    | Python / PyTorch (future)      |

---

## Prerequisites

- **Windows ARM64** (tested) or x64 / Linux
- **Visual Studio 2022** with MSVC ARM64 toolchain (or GCC 13+ / Clang 17+)
- **CMake ≥ 3.25**
- **Ninja**
- **Git**
- Raylib 6.0 SDK at `D:\Game development projects\raylib-6.0_winarm64_msvc16` (or set via `-DRAYLIB_SDK_DIR`)
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

SyntraQ maintains a test suite across kinematics, signal state machines, pathfinding algorithms, queue dynamics, and metrics tracking:

```powershell
cd build
ctest --output-on-failure
```

Or execute targeted test suites directly:

```powershell
.\build\bin\test_traffic_behavior.exe
.\build\bin\test_metrics.exe
.\build\bin\test_routing.exe
.\build\bin\test_signals.exe
```

---

## Project Structure

```
SyntraQ/
├── include/syntraq/
│   ├── core/          # Config and shared types
│   ├── world/         # Road network graph (Intersections & Roads)
│   ├── vehicles/      # Vehicle kinematics, car-following, and spawner
│   ├── signals/       # Traffic signals (Phases, Controllers)
│   ├── routing/       # Routing engine (Dijkstra, A*, Heuristics, Costs)
│   ├── metrics/       # Telemetry subsystem (TrafficMetricsCollector, RunResult)
│   ├── simulation/    # Headless simulation loop & state
│   └── rendering/     # Raylib + ImGui renderer
├── src/
│   ├── core/          # Config implementation
│   ├── world/         # Road network implementation
│   ├── vehicles/      # Kinematics and queue formation systems
│   ├── signals/       # Traffic signal controllers
│   ├── routing/       # Dijkstra, A*, and graph generator
│   ├── metrics/       # Telemetry collector and CSV exporters
│   ├── simulation/    # Simulation loop & scenario management
│   ├── rendering/     # Renderer and three-panel UI layout
│   └── main.cpp       # CLI parsing and application entry point
├── experiments/       # Automated multi-scenario batch experiment runner
├── scripts/           # Python statistical analysis and plotting pipelines
├── tests/             # GoogleTest suites (124 tests across all subsystems)
├── benchmarks/        # Routing engine benchmarks (Dijkstra vs A*)
├── configs/           # Scenario JSON configurations
├── data/              # Experiment CSV datasets and generated charts
├── assets/            # Fonts, textures (future)
├── models/            # ONNX models (future)
└── docs/              # Layout screenshots and architecture diagrams
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
