#!/usr/bin/env python3
"""Synthesis A/B tuning presets (constants only — no FEM)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_SYNTHESIS_PRESET = "current"
BODY_LOW_FREQ_TILT_HZ = 160.0


@dataclass(frozen=True)
class SynthTuning:
    name: str
    body_to_string_target_ratio: float
    body_modal_richness_gain: float
    body_modal_bandwidth_widening: float
    body_modal_gain: float
    string_pluck_gain: float
    string_pitch_layer_gain: float
    body_low_mode_weight: float
    high_note_pluck_gain_floor: float
    high_note_pluck_transient_reduction: float
    high_note_pitch_layer_attack_soften: float
    high_note_attack_decay_shorten: float
    high_note_hf_rolloff_k_power: float
    high_note_pitch_layer_high_scale: float

    def to_metadata_dict(self) -> Dict[str, Any]:
        return asdict(self)


SYNTHESIS_PRESETS: Dict[str, SynthTuning] = {
    "current": SynthTuning(
        name="current",
        body_to_string_target_ratio=4.2,
        body_modal_richness_gain=1.20,
        body_modal_bandwidth_widening=1.12,
        body_modal_gain=1.0,
        string_pluck_gain=0.10,
        string_pitch_layer_gain=0.055,
        body_low_mode_weight=1.0,
        high_note_pluck_gain_floor=0.40,
        high_note_pluck_transient_reduction=0.65,
        high_note_pitch_layer_attack_soften=1.90,
        high_note_attack_decay_shorten=0.48,
        high_note_hf_rolloff_k_power=0.34,
        high_note_pitch_layer_high_scale=0.82,
    ),
    "classical_balanced": SynthTuning(
        name="classical_balanced",
        body_to_string_target_ratio=3.9,
        body_modal_richness_gain=1.12,
        body_modal_bandwidth_widening=1.10,
        body_modal_gain=1.0,
        string_pluck_gain=0.095,
        string_pitch_layer_gain=0.052,
        body_low_mode_weight=0.90,
        high_note_pluck_gain_floor=0.42,
        high_note_pluck_transient_reduction=0.62,
        high_note_pitch_layer_attack_soften=1.88,
        high_note_attack_decay_shorten=0.46,
        high_note_hf_rolloff_k_power=0.30,
        high_note_pitch_layer_high_scale=0.86,
    ),
    "less_metal_highs": SynthTuning(
        name="less_metal_highs",
        body_to_string_target_ratio=4.0,
        body_modal_richness_gain=1.16,
        body_modal_bandwidth_widening=1.11,
        body_modal_gain=1.0,
        string_pluck_gain=0.088,
        string_pitch_layer_gain=0.046,
        body_low_mode_weight=0.95,
        high_note_pluck_gain_floor=0.34,
        high_note_pluck_transient_reduction=0.72,
        high_note_pitch_layer_attack_soften=2.0,
        high_note_attack_decay_shorten=0.52,
        high_note_hf_rolloff_k_power=0.38,
        high_note_pitch_layer_high_scale=0.76,
    ),
    "lighter_body": SynthTuning(
        name="lighter_body",
        body_to_string_target_ratio=3.55,
        body_modal_richness_gain=1.05,
        body_modal_bandwidth_widening=1.08,
        body_modal_gain=1.0,
        string_pluck_gain=0.10,
        string_pitch_layer_gain=0.055,
        body_low_mode_weight=0.82,
        high_note_pluck_gain_floor=0.38,
        high_note_pluck_transient_reduction=0.66,
        high_note_pitch_layer_attack_soften=1.92,
        high_note_attack_decay_shorten=0.47,
        high_note_hf_rolloff_k_power=0.32,
        high_note_pitch_layer_high_scale=0.84,
    ),
}

_active_tuning: SynthTuning = SYNTHESIS_PRESETS[DEFAULT_SYNTHESIS_PRESET]


def list_synthesis_preset_names() -> List[str]:
    return list(SYNTHESIS_PRESETS.keys())


def get_synthesis_preset(name: str) -> SynthTuning:
    key = str(name or DEFAULT_SYNTHESIS_PRESET).strip().lower()
    if key not in SYNTHESIS_PRESETS:
        raise ValueError(f"unknown synthesis preset: {name!r}")
    return SYNTHESIS_PRESETS[key]


def active_tuning() -> SynthTuning:
    return _active_tuning


def set_active_tuning(preset_name: str) -> SynthTuning:
    global _active_tuning
    _active_tuning = get_synthesis_preset(preset_name)
    return _active_tuning


@contextmanager
def use_synthesis_preset(preset_name: Optional[str]) -> Iterator[SynthTuning]:
    global _active_tuning
    if not preset_name:
        yield _active_tuning
        return
    saved = _active_tuning
    _active_tuning = get_synthesis_preset(preset_name)
    try:
        yield _active_tuning
    finally:
        _active_tuning = saved
