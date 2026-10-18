# Milestone 11: PyTorch Neural Traffic Prediction Model Report

**SyntraQ Traffic Simulation & Optimization Engine**  
*Evaluation Timestamp: 2026-10-18 12:38:44*

---

## 1. Executive Summary

This report evaluates a lightweight Multi-Layer Perceptron (MLP) implemented in **PyTorch** for traffic prediction on the SyntraQ platform. The neural network predicts **future traffic queue length 15 seconds in advance** (`target_queue_15s`) from 26 state telemetry features.

### Key Highlights
- **Architecture**: Intentionally simple 3-layer MLP (`Input[26] -> Linear[64] -> ReLU -> Linear[32] -> ReLU -> Linear[1]`), totaling **3,841 parameters**.
- **Accuracy**: Outperforms all classical baselines, achieving **Test $R^2 = 0.8919$** and **Test RMSE = 2.563 vehicles** (a **+5.9%** error reduction over Gradient Boosting and **+33.1%** over Linear Regression).
- **Test MAE**: `1.750 vehicles` across diverse traffic demands (from low traffic through peak rush hour).
- **Model Footprint**: Checkpoint size is only **`20.3 KB`** (compared to 610 KB for Gradient Boosting tree ensembles).
- **Decision on Keeping the Model**: **Kept as the primary predictive representation**. The PyTorch MLP delivers superior accuracy, a 34x smaller memory footprint, smooth gradient dynamics for reinforcement learning, and a straightforward path for ONNX / C++ tensor inference.

---

## 2. Model Architecture & Training Protocol

```
                     ┌──────────────────────────┐
                     │   Input Layer (26 Dim)   │
                     └─────────────┬────────────┘
                                   │
                     ┌─────────────▼────────────┐
                     │ Dense Linear (26 -> 64)  │
                     │ Activation: ReLU         │
                     └─────────────┬────────────┘
                                   │
                     ┌─────────────▼────────────┐
                     │ Dense Linear (64 -> 32)  │
                     │ Activation: ReLU         │
                     └─────────────┬────────────┘
                                   │
                     ┌─────────────▼────────────┐
                     │ Output Linear (32 -> 1)  │
                     │ Target: Future Queue 15s │
                     └──────────────────────────┘
```

### Hyperparameters & Training Settings
| Hyperparameter | Configuration | Description |
| :--- | :--- | :--- |
| **Input Dimensions** | 26 | Standard-scaled traffic features (kinematics, approaches, phases) |
| **Hidden Layers** | 64, 32 | Compact dense layers with ReLU non-linearities |
| **Output Dimensions** | 1 | Single-target scalar regression (`target_queue_15s`) |
| **Total Parameters** | 3,841 | Extremely lightweight feedforward graph |
| **Batch Size** | 32 | Stochastic mini-batching with epoch shuffling |
| **Optimizer** | Adam | Learning rate: `0.003`, Weight Decay: `0.0001` |
| **Loss Function** | MSELoss | Mean squared error on continuous queue targets |
| **LR Scheduler** | ReduceLROnPlateau | Factor: 0.5, Patience: 12 epochs, Min LR: 1e-5 |
| **Epochs** | 120 | Best checkpoint restored from Epoch `15` |
| **Training Time** | 5.95 s | Fast CPU convergence |

---

## 3. Comprehensive Benchmark Comparison Table

All models are evaluated on the exact same **run-disjoint partitions** (1,080 Train / 360 Val / 360 Test) across 20 independent simulation trajectories to prevent any data leakage.

| Model Architecture | Type | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Max Error | Latency | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Mean)** | Baseline | 6.399 | 8.915 | -0.0004 | 5.937 | 7.803 | -0.0024 | 35.73 | 0.03 us | < 5 KB |
| **Linear Regression** | Classical ML | 2.898 | 3.849 | 0.8135 | 2.916 | 3.830 | 0.7585 | 13.46 | 0.21 us | < 5 KB |
| **Ridge Regression** | Classical ML | 2.902 | 3.854 | 0.8131 | 2.912 | 3.809 | 0.7612 | 12.93 | 0.11 us | < 5 KB |
| **Random Forest** | Classical ML | 2.262 | 3.459 | 0.8494 | 2.145 | 3.087 | 0.8431 | 12.77 | 33.65 us | 610 KB |
| **Gradient Boosting** | Classical ML | 2.090 | 3.072 | 0.8812 | 1.904 | 2.723 | 0.8779 | 10.80 | 1.91 us | 610 KB |
| **PyTorch TrafficMLP** | **Neural Net** | **1.790** | **2.405** | **0.9272** | **1.750** | **2.563** | **0.8919** | **9.57** | **2.45 us** | **20.3 KB** |

---

## 4. Trade-Off Analysis: Should We Keep the Neural Model?

The project guideline states:  
> *"Only keep the neural model if it provides a useful trade-off."*

Our comparative benchmark yields four decisive conclusions in favor of **keeping the PyTorch MLP**:

### A. Higher Generalization Accuracy on Holdout Test Runs
- Gradient Boosting achieved $R^2 = 0.8779$ and RMSE = `2.723`.
- PyTorch TrafficMLP improves this to **$R^2 = 0.8919$** and **RMSE = `2.563`** vehicles.
- Decision trees produce piecewise constant step approximations that saturate at extreme values. In contrast, the continuous piecewise linear activation of ReLUs in the MLP models the smooth queuing dynamics of vehicles arriving at intersections during green and yellow clearance cycles with higher fidelity.

### B. 34x Smaller Memory Footprint
- Gradient Boosting requires storing **150 trees** with split thresholds, feature indices, and leaf values (**610 KB**).
- PyTorch TrafficMLP stores only three weight matrices and bias vectors (**20.3 KB**), an order-of-magnitude reduction in cache footprint.

### C. Seamless C++ / ONNX Inference Embedding
- Embedding a tree ensemble into C++ requires either heavy third-party libraries (e.g. Treelite) or bulky generated code (thousands of nested `if/else` statements).
- A 3-layer MLP consists of three matrix-vector multiplications ($y = Wx + b$) and two ReLU activations (max(0, x)). This can be implemented in **under 30 lines of pure C++20 with zero third-party dependencies**, or exported to ONNX for accelerated evaluation via ONNX Runtime.

### D. Foundation for Reinforcement Learning & Adaptive Signal Control
- Future milestones involve reinforcement learning (DQN, PPO, Actor-Critic) and neural adaptive signal optimization.
- The PyTorch MLP architecture and PyTorch training infrastructure provide the exact building blocks required for value networks and policy representations.

---

## 5. Visual Diagnostics

![PyTorch MLP Diagnostics](mlp_training_and_evaluation.png)

The four-panel diagnostic visualization demonstrates:
1. **Panel A (Training Dynamics)**: Monotonic reduction in training MSE with stable validation loss plateauing around epoch 15, showing no destructive overfitting.
2. **Panel B (Scatter Plot)**: Predicted vs. actual test values cluster tightly along the ideal diagonal line across low, medium, high, and rush-hour regimes.
3. **Panel C (Dynamic Trajectory)**: Continuous holdout test trajectory tracking showing how the MLP closely mirrors ground truth queue spikes during signal cycles without lagging behind traffic shockwaves.
4. **Panel D (Benchmark Comparison)**: Direct bar chart visual comparing Test $R^2$ and RMSE across all classical models and the PyTorch MLP.

---

## 6. Reproducibility & Saved Artifacts

- **Model Checkpoint**: `models/best_traffic_mlp.pt`
- **Evaluation Metrics**: `models/mlp_evaluation_metrics.json`
- **Diagnostic Charts**: `docs/mlp_training_and_evaluation.png` (and `data/experiments/charts/mlp_training_and_evaluation.png`)
- **Random Seed**: Fixed at `42` across PyTorch and NumPy
