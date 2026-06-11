#!/usr/bin/env python3
"""
Continuous f0-based string vs body balance (no per-note hacks).
"""
from __future__ import annotations

import math


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _high_soften_t(f0: float, *, threshold: float = 300.0, full: float = 620.0) -> float:
    if f0 <= threshold:
        return 0.0
    return _smoothstep(threshold, full, f0)


def _low_soften_t(f0: float, *, full: float = 165.0) -> float:
    if f0 >= full:
        return 0.0
    return 1.0 - _smoothstep(0.0, full, f0)


def string_direct_scale_by_f0(f0: float) -> float:
    """Reduce direct pluck dominance at low and high fundamentals."""
    f0 = max(40.0, float(f0))
    low_t = _low_soften_t(f0)
    high_t = _high_soften_t(f0)
    low_scale = 1.0 - low_t * 0.10
    high_scale = 1.0 - high_t * 0.42
    return low_scale * high_scale


def pitch_layer_scale_by_f0(f0: float) -> float:
    """Pitch layer: slightly softer lows, much softer highs."""
    f0 = max(40.0, float(f0))
    low_t = _low_soften_t(f0)
    high_t = _high_soften_t(f0)
    low_scale = 1.0 - low_t * 0.06
    high_scale = 1.0 - high_t * 0.38
    return low_scale * high_scale


def body_color_gain_by_f0(f0: float) -> float:
    """More body color at mid/high where string often dominates."""
    f0 = max(40.0, float(f0))
    mid_t = _smoothstep(120.0, 320.0, f0)
    high_t = _high_soften_t(f0)
    return 1.0 + 0.08 * mid_t + 0.28 * high_t


def fundamental_anchor_scale_by_body_strength(
    f0: float,
    *,
    body_rms: float,
    string_rms: float,
    base_scale: float = 1.0,
) -> float:
    """Lower fundamental anchor when body already supports the note."""
    f0 = max(40.0, float(f0))
    low_t = _low_soften_t(f0)
    if low_t <= 0:
        return base_scale
    ratio = body_rms / max(string_rms, 1e-12)
    body_support = _smoothstep(1.5, 4.5, ratio)
    return base_scale * (1.0 - low_t * 0.55 * body_support)
