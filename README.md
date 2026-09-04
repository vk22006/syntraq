# SyntraQ

**Real-time AI-driven Urban Traffic Simulation & Optimization**

A research-oriented C++20 simulation platform that models urban traffic and evaluates AI-based traffic management strategies.

---

## Status

| Milestone | Description                              | Status      |
|-----------|------------------------------------------|-------------|
| M1        | Project foundation & simulation loop    | ✅ Complete |
| M2        | Road network, vehicles, fixed-time signals | Planned   |
| M3        | Routing, metrics, CSV export             | Planned     |
| M4        | Adaptive signal controller               | Planned     |
| M5        | ONNX AI integration                      | Planned     |

---

## Tech Stack

| Component      | Technology                     |
|----------------|--------------------------------|
| Core language  | C++20                          |
| Build system   | CMake 4.x + Ninja              |
| Rendering      | Raylib 6.0 (ARM64 MSVC)        |
| UI / Debug     | Dear ImGui + rlImGui           |
| Tests          | GoogleTest + CTest             |
| CI             | GitHub Actions                 |
| AI deployment  | ONNX Runtime (future)          |
| AI training    | Python / PyTorch (future)      |

---

## Prerequisites

- **Windows ARM64** (tested) or x64
- **Visual Studio 2022** with MSVC ARM64 toolchain
- **CMake ≥ 3.25**
- **Ninja**
- **Git**
- Raylib 6.0 SDK at `D:\Game development projects\raylib-6.0_winarm64_msvc16`

---

## Building

Open a **Developer Command Prompt for VS 2022 (ARM64)** and run:

```powershell
# Configure (Debug)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build

# Run
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

## Running Tests

```powershell
cd build
ctest --output-on-failure
```

Or run a specific test binary directly:

```powershell
.\build\bin\test_simulation.exe
```

---

## Project Structure

```
SyntraQ/
├── include/syntraq/
│   ├── core/          # Config and shared types
│   ├── simulation/    # Headless simulation loop
│   └── rendering/     # Raylib + ImGui renderer
├── src/
│   ├── core/          # Config implementation
│   ├── simulation/    # Simulation implementation
│   ├── rendering/     # Renderer implementation
│   └── main.cpp       # Entry point
├── tests/             # GoogleTest suites
├── configs/           # Scenario JSON files
├── assets/            # Fonts, textures (future)
├── models/            # ONNX models (future)
├── scripts/           # Python training scripts (future)
├── benchmarks/        # Performance benchmarks (future)
├── data/              # Runtime metrics output (gitignored)
└── docs/              # Documentation
```

---

## Architecture Principles

1. `syntraq_core` (static lib) — headless, no Raylib, fully testable
2. `syntraq_renderer` — all Raylib/ImGui calls isolated here
3. `syntraq` — thin entry point only
4. Simulation reads Config once at startup; no global state
5. Renderer reads `SimState`; never writes to Simulation

---

## License

MIT — see [LICENSE](LICENSE)
