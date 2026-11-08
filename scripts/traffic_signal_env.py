"""
scripts/traffic_signal_env.py

Gymnasium-compatible Reinforcement Learning Environment for Single-Intersection
Traffic Signal Optimization in SyntraQ.

Binds to the high-performance C-API shared library (syntraq_env_capi) via ctypes,
enabling in-memory, zero-copy, sub-millisecond step execution for PyTorch-based RL.

Observation Space:
    spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
    [0] queue_phase_0       Normalized queue on Phase 0 (North-South) approaches [0..1]
    [1] queue_phase_1       Normalized queue on Phase 1 (East-West) approaches [0..1]
    [2] approaching_phase_0 Normalized moving approaching vehicles on Phase 0 [0..1]
    [3] approaching_phase_1 Normalized moving approaching vehicles on Phase 1 [0..1]
    [4] avg_speed_phase_0   Normalized average vehicle speed on Phase 0 [0..1]
    [5] avg_speed_phase_1   Normalized average vehicle speed on Phase 1 [0..1]
    [6] current_phase       Active phase index (0.0: Phase 0, 1.0: Phase 1)
    [7] time_in_phase       Elapsed time in current stage / max_green_s [0..1]
    [8] is_clearing         1.0 if Yellow or AllRed clearance transition, 0.0 otherwise
    [9] can_switch          1.0 if Green and elapsed >= min_green_s, 0.0 otherwise
   [10] avg_wait_phase_0    Normalized average vehicle waiting time on Phase 0 [0..1]
   [11] avg_wait_phase_1    Normalized average vehicle waiting time on Phase 1 [0..1]

Action Space:
    spaces.Discrete(2)
    0: Maintain current phase (continue current green or ongoing clearance)
    1: Switch phase (initiate transition via safe yellow/all-red clearance)

Reward Function:
    Penalizes queue lengths, waiting time, and congestion ratio.
    Rewards vehicle throughput departures:
    R_t = - (w_queue * Q + w_wait * W + w_cong * C) + w_thr * delta_Departures
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None


class SyntraqEnvConfigC(ctypes.Structure):
    _fields_ = [
        ("step_duration_s", ctypes.c_float),
        ("sim_dt_s", ctypes.c_float),
        ("episode_duration_s", ctypes.c_float),
        ("target_intersection", ctypes.c_uint32),
        ("use_single_intersection_map", ctypes.c_int32),
        ("scenario_id", ctypes.c_int32),
        ("seed", ctypes.c_uint32),
        ("min_green_s", ctypes.c_float),
        ("max_green_s", ctypes.c_float),
        ("yellow_s", ctypes.c_float),
        ("all_red_s", ctypes.c_float),
        ("queue_penalty_weight", ctypes.c_float),
        ("wait_penalty_weight", ctypes.c_float),
        ("congestion_penalty_weight", ctypes.c_float),
        ("throughput_reward_weight", ctypes.c_float),
        ("invalid_action_penalty", ctypes.c_float),
    ]


def _find_capi_dll() -> Optional[Path]:
    """Search for syntraq_env_capi shared library in common build locations."""
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "build" / "bin" / "syntraq_env_capi.dll",
        root / "build" / "bin" / "Release" / "syntraq_env_capi.dll",
        root / "build" / "bin" / "Debug" / "syntraq_env_capi.dll",
        root / "build" / "lib" / "libsyntraq_env_capi.so",
        root / "build" / "lib" / "libsyntraq_env_capi.dylib",
        root / "bin" / "syntraq_env_capi.dll",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class TrafficSignalEnv(gym.Env if gym is not None else object):
    """
    Gymnasium environment for single-intersection reinforcement learning.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        step_duration_s: float = 1.0,
        sim_dt_s: float = 0.1,
        episode_duration_s: float = 300.0,
        scenario: str = "Medium",
        seed: int = 42,
        min_green_s: float = 4.0,
        max_green_s: float = 45.0,
        yellow_s: float = 3.0,
        all_red_s: float = 2.0,
        queue_weight: float = 0.2,
        wait_weight: float = 0.05,
        congestion_weight: float = 1.0,
        throughput_weight: float = 1.0,
        invalid_action_penalty: float = 0.05,
        use_single_intersection_map: bool = True,
        dll_path: Optional[str | Path] = None,
    ):
        super().__init__()

        self.step_duration_s = step_duration_s
        self.sim_dt_s = sim_dt_s
        self.episode_duration_s = episode_duration_s
        self.scenario_name = scenario
        self.seed_val = seed
        self.min_green_s = min_green_s
        self.max_green_s = max_green_s
        self.yellow_s = yellow_s
        self.all_red_s = all_red_s
        self.queue_weight = queue_weight
        self.wait_weight = wait_weight
        self.congestion_weight = congestion_weight
        self.throughput_weight = throughput_weight
        self.invalid_action_penalty = invalid_action_penalty
        self.use_single_intersection_map = use_single_intersection_map

        scenario_map = {"Low": 0, "Light": 0, "Medium": 1, "High": 2, "Heavy": 2, "RushHour": 3}
        self.scenario_id = scenario_map.get(scenario, 1)

        # Observation & action spaces
        self.obs_dim = 12
        if spaces is not None:
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
            )
        else:
            self.action_space = None
            self.observation_space = None

        # Load C-API library
        dll_file = Path(dll_path) if dll_path else _find_capi_dll()
        if dll_file is None or not dll_file.exists():
            raise FileNotFoundError(
                f"Could not locate syntraq_env_capi shared library. Please build the project with CMake first."
            )

        self._lib = ctypes.CDLL(str(dll_file))
        self._setup_c_signatures()

        # Create C environment instance
        c_cfg = SyntraqEnvConfigC(
            step_duration_s=ctypes.c_float(self.step_duration_s),
            sim_dt_s=ctypes.c_float(self.sim_dt_s),
            episode_duration_s=ctypes.c_float(self.episode_duration_s),
            target_intersection=ctypes.c_uint32(0),
            use_single_intersection_map=ctypes.c_int32(1 if self.use_single_intersection_map else 0),
            scenario_id=ctypes.c_int32(self.scenario_id),
            seed=ctypes.c_uint32(self.seed_val),
            min_green_s=ctypes.c_float(self.min_green_s),
            max_green_s=ctypes.c_float(self.max_green_s),
            yellow_s=ctypes.c_float(self.yellow_s),
            all_red_s=ctypes.c_float(self.all_red_s),
            queue_penalty_weight=ctypes.c_float(self.queue_weight),
            wait_penalty_weight=ctypes.c_float(self.wait_weight),
            congestion_penalty_weight=ctypes.c_float(self.congestion_weight),
            throughput_reward_weight=ctypes.c_float(self.throughput_weight),
            invalid_action_penalty=ctypes.c_float(self.invalid_action_penalty),
        )

        self._handle = self._lib.syntraq_env_create(ctypes.byref(c_cfg))
        if not self._handle:
            raise RuntimeError("Failed to initialize native SyntraQ RL environment.")

        # Buffers for ctypes interop
        self._obs_buffer = (ctypes.c_float * self.obs_dim)()
        self._info_buffer = (ctypes.c_float * 6)()

    def _setup_c_signatures(self):
        self._lib.syntraq_env_create.argtypes = [ctypes.POINTER(SyntraqEnvConfigC)]
        self._lib.syntraq_env_create.restype = ctypes.c_void_p

        self._lib.syntraq_env_destroy.argtypes = [ctypes.c_void_p]
        self._lib.syntraq_env_destroy.restype = None

        self._lib.syntraq_env_reset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ]
        self._lib.syntraq_env_reset.restype = None

        self._lib.syntraq_env_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._lib.syntraq_env_step.restype = ctypes.c_int32

        self._lib.syntraq_env_get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ]
        self._lib.syntraq_env_get_info.restype = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        """
        if seed is not None:
            self.seed_val = seed

        self._lib.syntraq_env_reset(
            self._handle,
            ctypes.c_uint32(self.seed_val),
            self._obs_buffer,
            ctypes.c_int32(self.obs_dim),
        )

        obs = np.array(self._obs_buffer, dtype=np.float32)
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the simulation by one action step:
          action 0: Maintain current phase
          action 1: Switch phase
        """
        action_c = ctypes.c_int32(int(action))
        reward_c = ctypes.c_float(0.0)
        term_c = ctypes.c_int32(0)
        trunc_c = ctypes.c_int32(0)

        status = self._lib.syntraq_env_step(
            self._handle,
            action_c,
            self._obs_buffer,
            ctypes.c_int32(self.obs_dim),
            ctypes.byref(reward_c),
            ctypes.byref(term_c),
            ctypes.byref(trunc_c),
        )

        if status != 0:
            raise RuntimeError(f"syntraq_env_step failed with error code {status}")

        obs = np.array(self._obs_buffer, dtype=np.float32)
        reward = float(reward_c.value)
        terminated = bool(term_c.value)
        truncated = bool(trunc_c.value)
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        self._lib.syntraq_env_get_info(
            self._handle,
            self._info_buffer,
            ctypes.c_int32(6),
        )
        return {
            "total_queue": int(self._info_buffer[0]),
            "avg_wait_s": float(self._info_buffer[1]),
            "congestion_ratio": float(self._info_buffer[2]),
            "step_departures": int(self._info_buffer[3]),
            "elapsed_s": float(self._info_buffer[4]),
            "cumulative_arrived": int(self._info_buffer[5]),
        }

    def close(self):
        if hasattr(self, "_handle") and self._handle and hasattr(self, "_lib"):
            self._lib.syntraq_env_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
