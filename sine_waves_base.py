"""Sinusoidal gait helpers (SDK joint space, Laikago ordering)."""

from __future__ import annotations

import math

import numpy as np

import robo_common as rc


def sine_leg_offsets(t: float, freq_hz: float, leg_phase: float) -> np.ndarray:
    """Returns length-3 abduction, hip, knee deltas for one leg."""
    w = 2.0 * math.pi * freq_hz
    s = math.sin(w * t + leg_phase)
    return np.array([0.06 * s, 0.18 * s, -0.22 * s], dtype=np.float64)


def sine_sdk_gait(t: float, freq_hz: float = 1.0) -> np.ndarray:
    """Four legs with alternating phases (trot-like)."""
    base = np.array(rc.INIT_SDK, dtype=np.float64)
    phases = (0.0, math.pi, math.pi, 0.0)
    for leg in range(4):
        off = sine_leg_offsets(t, freq_hz, phases[leg])
        base[leg * 3 : leg * 3 + 3] += off
    return base
