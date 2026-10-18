# SyntraQ Traffic Prediction Dataset: Specification & Methodology

This document details the feature engineering, target formulation, sampling strategy, scenario generation, and leakage-free dataset splitting protocols implemented for training predictive machine learning models in SyntraQ.

---

## 1. Task Formulation & Target Definitions

The predictive objective is to forecast future traffic dynamics at an intersection or across a network given current and recent historical observations. SyntraQ formulates two complementary prediction horizons:

- **Short-Term Horizon ($H_1 = 15.0\,\text{s}$)**: Captures immediate signal phase transitions, stopping shockwaves, and queue startup delays.
- **Medium-Term Horizon ($H_2 = 30.0\,\text{s}$)**: Captures macro queue accumulation, cycle-to-cycle spillover, and network congestion onset.

### Prediction Targets

For each horizon $H \in \{15\,\text{s}, 30\,\text{s}\}$, the following ground-truth targets are captured:

1. **Future Queue Length (`target_queue_15s`, `target_queue_30s`)**:
   - **Definition**: Total number of vehicles in the network stopped or moving slower than $0.5\,\text{m/s}$ at $t + H$.
   - **Type**: Continuous non-negative integer ($\mathbb{Z}_{\ge 0}$) / regression target.
   - **Utility**: Allows signal controllers to dynamically extend or truncate green phases before vehicle queues spill over.

2. **Future Peak Lane Queue (`target_max_queue_15s`, `target_max_queue_30s`)**:
   - **Definition**: The longest queue formed on any single approach lane at $t + H$.
   - **Type**: Integer ($\mathbb{Z}_{\ge 0}$).
   - **Utility**: Critical for detecting bottleneck approaches and preventing intersection blockages.

3. **Future Congestion Ratio (`target_congestion_15s`, `target_congestion_30s`)**:
   - **Definition**:
     $$\text{Congestion}(t + H) = \frac{N_{\text{queue}}(t + H)}{\max(1, N_{\text{active}}(t + H))}$$
   - **Type**: Continuous ratio $[0.0, 1.0]$.
   - **Utility**: Scale-invariant metric representing the proportion of the active vehicle fleet stalled in traffic.

---

## 2. Feature Definitions

At each sample timestamp $t$, 26 input features are extracted across 6 structural categories:

### A. Vehicle Count & Inflow Features
| Feature Name | Type | Description |
|---|---|---|
| `vehicle_count` | Integer | Total active vehicles currently traveling or stopped on the network. |
| `spawn_rate_vpm` | Float | Vehicle inflow generation rate ($\text{veh/min}$), derived from the scenario configuration. |

### B. Queue Dynamics Features
| Feature Name | Type | Description |
|---|---|---|
| `queue_vehicle_count` | Integer | Vehicles currently stopped ($v < 0.5\,\text{m/s}$) across all approaches. |
| `max_queue_length` | Integer | Longest vehicle queue observed on any individual approach lane. |
| `avg_queue_per_lane` | Float | Mean queue length averaged across all lanes: $\frac{N_{\text{queue}}}{L_{\text{total}}}$. |
| `queue_ratio` | Float | Fraction of the active fleet currently queued: $\frac{N_{\text{queue}}}{\max(1, N_{\text{active}})}$. |

### C. Speed & Travel Time Features
| Feature Name | Type | Description |
|---|---|---|
| `avg_speed_mps` | Float | Mean instantaneous speed across active vehicles ($\text{m/s}$). |
| `avg_speed_kmh` | Float | Mean instantaneous speed converted to $\text{km/h}$ ($v_{\text{mps}} \times 3.6$). |
| `speed_ratio` | Float | Mean speed normalized against the design speed limit: $\min\left(1.0, \frac{v_{\text{avg}}}{v_{\max}}\right)$. |
| `avg_wait_time_s` | Float | Mean accumulated standstill/waiting time across all active vehicles ($\text{s}$). |
| `avg_travel_time_s` | Float | Mean trip duration since spawning across all active vehicles ($\text{s}$). |

### D. Road Occupancy & Density Features
| Feature Name | Type | Description |
|---|---|---|
| `road_density_vpm` | Float | Network spatial density: $\frac{N_{\text{active}}}{\sum \text{length}_{\text{road}}}$ ($\text{veh/m}$). |
| `road_occupancy_pct` | Float | Total length of all vehicle bounding boxes divided by total lane length: $\frac{\sum \text{len}_v}{\sum (\text{len}_r \times \text{lanes}_r)} \times 100\%$. |
| `critical_occupancy_pct` | Float | Highest single-road occupancy percentage observed anywhere in the network. |

### E. Signal Phase & Intersection State Features
| Feature Name | Type | Description |
|---|---|---|
| `signal_phase_index` | Integer | Active signal phase index ($0, 1, \dots$). |
| `time_in_phase_s` | Float | Elapsed time within the current signal phase ($\text{s}$). |
| `phase_duration_s` | Float | Total nominal duration allocated to the active phase ($\text{s}$). |
| `phase_elapsed_ratio` | Float | Normalized phase progression: $\frac{t_{\text{phase}}}{T_{\text{phase}}} \in [0.0, 1.0]$. |
| `is_all_red` | Binary | 1 if the intersection is in an all-red transition/clearance stage, 0 otherwise. |
| `green_approaches_count` | Integer | Number of incoming approaches with an active green light. |

### F. Temporal Momentum & Rolling Statistics
Computed over a rolling 10-second historical window $[t - 10\,\text{s}, t]$:
| Feature Name | Type | Description |
|---|---|---|
| `recent_throughput_vph` | Float | Vehicle completion rate: $\frac{\Delta N_{\text{arrived}}}{\Delta t} \times 3600$ ($\text{veh/h}$). |
| `recent_queue_delta` | Float | Queue rate of change: $\frac{\Delta N_{\text{queue}}}{\Delta t}$ ($\text{veh/s}$). Positive values indicate congestion accumulation; negative values indicate queue dissipation. |
| `recent_speed_delta` | Float | Fleet acceleration trend: $\frac{\Delta v_{\text{avg}}}{\Delta t}$ ($\text{m/s}^2$). |

### G. Categorical Scenario Features (One-Hot Encoded)
| Feature Name | Type | Description |
|---|---|---|
| `scenario_low_traffic` | Binary | 1 if Low Traffic demand preset, 0 otherwise. |
| `scenario_medium_traffic` | Binary | 1 if Medium Traffic demand preset, 0 otherwise. |
| `scenario_high_traffic` | Binary | 1 if High Traffic demand preset, 0 otherwise. |
| `scenario_rush_hour` | Binary | 1 if Rush Hour demand preset, 0 otherwise. |

---

## 3. Sampling Strategy & Target Binding

### Fixed-Interval Sampling
- The simulation advances at a deterministic fixed timestep ($\Delta t = 0.10\,\text{s}$).
- States are sampled at regular intervals (default: $\Delta t_{\text{sample}} = 1.0\,\text{s}$), producing 1 observation per simulated second.

### Retroactive Ground-Truth Lookahead Binding
To eliminate lookahead bias:
1. When sample $S_i$ is recorded at simulation time $t$, all 26 feature values are extracted from the instantaneous state. Target fields are marked pending.
2. The collector holds sample index $i$ in FIFO queues for $H_1$ (15s) and $H_2$ (30s).
3. As the simulation progresses to $t + 15.0\,\text{s}$, the exact ground-truth values at $t + 15$ are retroactively bound to $S_i$:
   $$S_i.\text{target\_queue\_15s} = N_{\text{queue}}(t + 15)$$
   $$S_i.\text{target\_congestion\_15s} = \text{Congestion}(t + 15)$$
4. Similarly, when the simulation reaches $t + 30.0\,\text{s}$, $S_i$'s 30-second targets are populated.
5. **No Fabricated Data**: Samples recorded during the final 30 seconds of a simulation run do not have ground-truth future data within that run. These terminal samples are flagged `has_target_30s = false` and excluded from model training to prevent boundary distortion.

---

## 4. Scenario Generation & Demand Diversity

To ensure the dataset covers diverse traffic regimes, `syntraq_generate_dataset` generates multi-seed runs across 4 standard scenario presets:

| Scenario Preset | Vehicle Inflow Interval | Max Fleet Size | Regimes Captured |
|---|---|---|---|
| **Low Traffic** | $4.0\,\text{s}$ ($15\,\text{veh/min}$) | 10 | Free-flow traffic, sparse single-vehicle arrivals, minimal queueing. |
| **Medium Traffic** | $2.0\,\text{s}$ ($30\,\text{veh/min}$) | 25 | Moderate density, regular small queue formation at red lights, orderly dissipation. |
| **High Traffic** | $1.2\,\text{s}$ ($50\,\text{veh/min}$) | 45 | Heavy demand, multi-vehicle queue stacking, near-saturation throughput. |
| **Rush Hour** | $0.8\,\text{s}$ ($75\,\text{veh/min}$) | 60 | Severe oversaturation, persistent queues across multiple signal cycles. |

Each scenario is simulated across multiple pseudo-random seeds ($1000, 1137, 1274, \dots$) that vary vehicle routing choices, origin-destination pairs, and spawn headways.

---

## 5. Train / Validation / Test Splitting Protocol

### Data Leakage Prevention Guarantees

In sequential time-series simulation data, naive random sample shuffling introduces severe **data leakage** because sample $t$ and sample $t+1$ within the same run are strongly auto-correlated. Evaluating a model on randomly shuffled samples yields artificially inflated accuracy that fails to generalize.

SyntraQ enforces three strict leakage prevention protocols in `scripts/preprocess_dataset.py`:

```
               All Simulation Trajectories (Runs 1 .. N)
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
         Train Runs (60-70%) Val Runs (15-20%) Test Runs (15-20%)
         [Runs 1,2,4,5...]    [Runs 3,8,12...]  [Runs 7,14,19...]
               │                │                │
      Fit Scaler (μ, σ)         │                │
               │                │                │
        Transform Train ──► Transform Val ──► Transform Test
```

1. **Trajectory-Level (Run-Disjoint) Splitting**:
   - Splitting occurs strictly at the **simulation run (`run_id`)** level.
   - All observations from a given simulation run reside exclusively in one partition.
   - Run ID intersection between any two splits is mathematically guaranteed to be empty:
     $$\text{Runs}_{\text{train}} \cap \text{Runs}_{\text{val}} = \emptyset, \quad \text{Runs}_{\text{train}} \cap \text{Runs}_{\text{test}} = \emptyset, \quad \text{Runs}_{\text{val}} \cap \text{Runs}_{\text{test}} = \emptyset$$

2. **Stratified Group Splitting**:
   - Partitioning is stratified by traffic scenario. Every split (train, validation, test) contains independent runs from Low, Medium, High, and Rush Hour scenarios in equal proportions.

3. **Strict Scaler Isolation**:
   - Feature normalization parameters (mean $\mu$ and standard deviation $\sigma$) are computed **strictly on the training partition**.
   - Validation and test splits are transformed using the training set's frozen scalers, preventing distribution leakage.

---

## 6. Generated Artifacts & File Structure

Executing `syntraq_generate_dataset` and `scripts/preprocess_dataset.py` produces the following structured outputs under `data/dataset/`:

```
data/dataset/
├── raw_traffic_data.csv        # Complete raw simulation export with all features & targets
└── processed/
    ├── train.csv               # Standardized training split (60-70% of runs)
    ├── val.csv                 # Standardized validation split (15-20% of runs)
    ├── test.csv                # Standardized test split (15-20% of runs)
    ├── dataset.npz             # Compressed NumPy arrays (X_train, y_train, X_val, y_val, etc.)
    └── metadata.json           # Feature registry, scalers (μ, σ), and split audit manifest
```
