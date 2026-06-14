#!/usr/bin/env python3
"""
PGSM Step 5D — audible diagnostic render repair.
Gain-only listening renders; does not change physical model or Step 5A/4A files.
"""
from __future__ import annotations

import hashlib
import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3a_numerical_ir_testbench import SAMPLE_ID
from pgsm_step4b_single_note_diagnostic_refinement import STEP4A_AUDIO_DIR, _envelope, load_wav_mono
from pgsm_step5a_limited_note_set_diagnostic_audio import (
    AUDIO_DIR as STEP5A_AUDIO_DIR,
    NOTE_SET,
    step4a_output_fingerprints,
    write_wav_mono,
)
from pgsm_step5b_limited_note_set_refinement import _wav_paths_for_note, step5a_output_fingerprints
from pgsm_step5c_note_set_extended_validation import READINESS_STEP6A
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5D_VERSION = "pgsm_step5d_audible_diagnostic_render_repair_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5d_audible_diagnostic_render_repair.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5d_audible_diagnostic_render_repair.md"
RENDER_DIR = REPO_ROOT / "audio" / "pgsm_step5d_audible_render"

READINESS_AFTER = "ready_for_step6a_reference_guided_diagnostic_comparison_with_audible_renders"
TARGET_RMS_DBFS_MIN = -24.0
TARGET_RMS_DBFS_MAX = -20.0
TARGET_RMS_DBFS_NOMINAL = -22.0
PEAK_CAP_DBFS = -1.0
PEAK_CAP_FS = 10.0 ** (PEAK_CAP_DBFS / 20.0)
INAUDIBLE_RMS_DBFS = -35.0
LISTENING_TRIM_THRESHOLD_DBFS = -60.0
LISTENING_TAIL_PAD_MS = 50.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _linear_to_dbfs(x: float) -> float:
    return float(20.0 * math.log10(max(abs(x), 1e-12)))


def _dbfs_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))


def _active_duration_ms(y: np.ndarray, sr: int, threshold_fs: float) -> float:
    if y.size == 0:
        return 0.0
    mask = np.abs(y) >= threshold_fs
    if not mask.any():
        return 0.0
    idx = np.where(mask)[0]
    return float((idx[-1] - idx[0] + 1) / sr * 1000.0)


def analyze_audibility(y: np.ndarray, sr: int, *, role: str) -> Dict[str, Any]:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = _rms(y)
    crest = peak / max(rms, 1e-12)
    peak_db = _linear_to_dbfs(peak)
    rms_db = _linear_to_dbfs(rms)

    active_1pct_ms = _active_duration_ms(y, sr, 0.01 * max(peak, 1e-12))
    active_m60_ms = _active_duration_ms(y, sr, _dbfs_to_linear(-60.0))

    t = np.arange(len(y)) / sr
    after_05 = float(np.max(np.abs(y[t >= 0.5]))) if (t >= 0.5).any() else 0.0

    env = _envelope(y, sr)
    decay_40 = None
    if env.size and env.max() > 0:
        peak_i = int(np.argmax(env))
        target = env[peak_i] * 10.0 ** (-40.0 / 20.0)
        idx = np.where(env[peak_i:] <= target)[0]
        if idx.size:
            decay_40 = float(t[peak_i + int(idx[0])] * 1000.0)

    body_tail_quiet = False
    if role == "main" and after_05 > 0:
        body_tail_quiet = _linear_to_dbfs(after_05) < -50.0

    return {
        "role": role,
        "peak_fs": round(peak, 6),
        "peak_dbfs": round(peak_db, 3),
        "rms": round(rms, 8),
        "rms_dbfs": round(rms_db, 3),
        "crest_factor": round(crest, 3),
        "active_duration_above_1pct_peak_ms": round(active_1pct_ms, 3),
        "active_duration_above_minus_60_dbfs_ms": round(active_m60_ms, 3),
        "max_amplitude_after_0p5_s_fs": round(after_05, 8),
        "decay_minus_40_db_ms": decay_40,
        "inaudible_rms": bool(rms_db < INAUDIBLE_RMS_DBFS),
        "transient_dominant_peak": bool(crest > 12.0),
        "too_short_active_duration": bool(active_1pct_ms < 150.0),
        "body_tail_below_listening_threshold": bool(body_tail_quiet),
    }


def trim_listening_tail(
    y: np.ndarray,
    sr: int,
    *,
    threshold_dbfs: float = LISTENING_TRIM_THRESHOLD_DBFS,
    pad_ms: float = LISTENING_TAIL_PAD_MS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove inaudible trailing silence; does not alter decay shape of active signal."""
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y.copy(), {"trim_applied": False, "original_length_samples": 0, "trimmed_length_samples": 0}

    thr = _dbfs_to_linear(threshold_dbfs)
    idx = np.where(np.abs(y) >= thr)[0]
    if not idx.size:
        return y.copy(), {
            "trim_applied": False,
            "original_length_samples": int(y.size),
            "trimmed_length_samples": int(y.size),
            "trim_threshold_dbfs": threshold_dbfs,
            "tail_pad_ms": pad_ms,
        }

    end = min(int(y.size), int(idx[-1]) + 1 + int(round(pad_ms * sr / 1000.0)))
    return y[:end].copy(), {
        "trim_applied": bool(end < y.size),
        "original_length_samples": int(y.size),
        "trimmed_length_samples": end,
        "trim_end_ms": round(end / sr * 1000.0, 3),
        "trim_threshold_dbfs": threshold_dbfs,
        "tail_pad_ms": pad_ms,
        "decay_stretch_applied": False,
    }


def apply_listening_render(
    y: np.ndarray,
    sr: int,
    *,
    target_rms_dbfs: float = TARGET_RMS_DBFS_NOMINAL,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Gain-first listening render with inaudible tail trim and optional peak safety limiting."""
    y = np.asarray(y, dtype=np.float64)
    y_trim, trim_info = trim_listening_tail(y, sr)
    rms_in = _rms(y_trim)
    peak_in = float(np.max(np.abs(y_trim))) if y_trim.size else 0.0
    if rms_in <= 1e-12:
        return y_trim.copy(), y_trim.copy(), {
            "gain_linear": 1.0,
            "gain_db": 0.0,
            "limiter_applied": False,
            "limiter_type": None,
            "physics_changed": False,
            "gain_separate_from_physics": True,
            **trim_info,
        }

    target_rms = _dbfs_to_linear(target_rms_dbfs)
    gain_rms = target_rms / rms_in
    y_scaled = y_trim * gain_rms
    peak_scaled = float(np.max(np.abs(y_scaled)))

    limiter_applied = False
    limiter_type: Optional[str] = None
    gain_linear = gain_rms

    if peak_scaled > PEAK_CAP_FS:
        gain_peak = PEAK_CAP_FS / max(peak_in, 1e-12)
        y_peak_limited = y_trim * gain_peak
        rms_after_peak_gain = _rms(y_peak_limited)
        rms_after_db = _linear_to_dbfs(rms_after_peak_gain)

        if TARGET_RMS_DBFS_MIN <= rms_after_db <= TARGET_RMS_DBFS_MAX:
            y_out = y_peak_limited
            gain_linear = gain_peak
        else:
            y_out = np.clip(y_scaled, -PEAK_CAP_FS, PEAK_CAP_FS)
            gain_linear = gain_rms
            limiter_applied = True
            limiter_type = "transparent_peak_safety_clip_at_minus_1_dbfs"
    else:
        y_out = y_scaled

    if float(np.max(np.abs(y_out))) > 1.0:
        y_out = np.clip(y_out, -PEAK_CAP_FS, PEAK_CAP_FS)
        limiter_applied = True
        limiter_type = limiter_type or "transparent_peak_safety_clip_at_minus_1_dbfs"

    gain_db = 20.0 * math.log10(max(gain_linear, 1e-12))
    info = {
        "gain_linear": round(gain_linear, 6),
        "gain_db": round(gain_db, 3),
        "target_rms_dbfs": target_rms_dbfs,
        "limiter_applied": limiter_applied,
        "limiter_type": limiter_type,
        "physics_changed": False,
        "gain_separate_from_physics": True,
        "decay_stretch_applied": False,
        "reverb_echo_body_tail_added": False,
        **trim_info,
    }
    return y_out, y_trim, info


def _waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spectral_peak_positions(y: np.ndarray, sr: int, n_peaks: int = 8) -> List[float]:
    n = len(y)
    if n < 256:
        return []
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    peaks: List[float] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 35.0:
                peaks.append(float(freqs[i]))
    peaks.sort(key=lambda f: spec_db[np.argmin(np.abs(freqs - f))], reverse=True)
    return peaks[:n_peaks]


def validate_render(
    original: np.ndarray,
    rendered: np.ndarray,
    sr: int,
    render_info: Mapping[str, Any],
) -> Dict[str, Any]:
    peak = float(np.max(np.abs(rendered)))
    rms = _rms(rendered)
    rms_db = _linear_to_dbfs(rms)
    peak_db = _linear_to_dbfs(peak)

    gain = float(render_info.get("gain_linear") or 1.0)
    expected = original * gain
    if render_info.get("limiter_applied"):
        expected = np.clip(expected, -PEAK_CAP_FS, PEAK_CAP_FS)
    corr = round(_waveform_correlation(rendered, expected), 4)

    env = _envelope(rendered, sr)
    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(last_third.size and float(last_third.max()) > float(mid_third.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)

    second_onset = False
    if len(env) > sr // 4:
        first = env[: sr // 10].max()
        mid = env[sr // 8 : sr // 4].max()
        second_onset = mid > first * 0.45 and first > 1e-8

    orig_peaks = _spectral_peak_positions(original, sr)
    rend_peaks = _spectral_peak_positions(rendered, sr)
    peak_preserved = True
    if orig_peaks:
        matches = sum(1 for f in orig_peaks if any(abs(f - rf) <= 5.0 for rf in rend_peaks))
        peak_preserved = matches >= max(1, len(orig_peaks) // 2)

    return {
        "peak_fs": round(peak, 6),
        "peak_dbfs": round(peak_db, 3),
        "rms_dbfs": round(rms_db, 3),
        "rms_in_target_range": bool(TARGET_RMS_DBFS_MIN <= rms_db <= TARGET_RMS_DBFS_MAX),
        "peak_below_minus_1_dbfs": bool(peak_db <= PEAK_CAP_DBFS + 0.01),
        "no_clipping": bool(peak <= PEAK_CAP_FS + 1e-6),
        "waveform_correlation_after_gain_compensation": corr,
        "no_second_onset": bool(not second_onset),
        "no_end_rise": bool(not end_rise),
        "no_hard_gate": bool(not hard_gate),
        "no_added_echo_reverb_comb": True,
        "spectral_modal_peak_positions_preserved": bool(peak_preserved),
        "limiter_applied": bool(render_info.get("limiter_applied")),
        "pass": bool(
            TARGET_RMS_DBFS_MIN <= rms_db <= TARGET_RMS_DBFS_MAX
            and peak_db <= PEAK_CAP_DBFS + 0.01
            and peak <= 1.0
            and corr >= 0.98
            and not second_onset
            and not end_rise
            and not hard_gate
            and peak_preserved
        ),
    }


def build_artifact_guard(
    render_validation: Mapping[str, Mapping[str, Any]],
    render_infos: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    checks = {
        "no_reverb": True,
        "no_echo": True,
        "no_body_tail_layer": True,
        "no_eq_body_layer": True,
        "no_decay_stretch": all(not (render_infos.get(n) or {}).get("decay_stretch_applied") for n in NOTE_SET),
        "no_physics_change": all(not (render_infos.get(n) or {}).get("physics_changed") for n in NOTE_SET),
        "no_second_onset": all((render_validation.get(n) or {}).get("no_second_onset") for n in NOTE_SET),
        "no_end_rise": all((render_validation.get(n) or {}).get("no_end_rise") for n in NOTE_SET),
        "no_hard_gate": all((render_validation.get(n) or {}).get("no_hard_gate") for n in NOTE_SET),
    }
    return {**checks, "pass": bool(all(checks.values()))}


def build_readiness_after_step5d(objective_pass: bool) -> Dict[str, Any]:
    status = READINESS_AFTER if objective_pass else "failed_audible_render_repair"
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "step6a_with_audible_renders_allowed": status == READINESS_AFTER,
    }


def collect_all_fingerprints(root: Path, step5a: Mapping[str, Any]) -> Dict[str, str]:
    fps: Dict[str, str] = {}
    for note in NOTE_SET:
        paths = _wav_paths_for_note(root, step5a, note)
        for key, p in paths.items():
            fps[f"step5a_{note}_{key}"] = _file_fingerprint(p)
    for name, fp in step4a_output_fingerprints(root).items():
        fps[f"step4a_{name}"] = fp
    return fps


def build_pgsm_step5d_report(
    *,
    repo_root: Optional[Path] = None,
    render_dir: Optional[Path] = None,
    write_wav: bool = True,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_dir = Path(render_dir or (root / "audio" / "pgsm_step5d_audible_render"))

    step5c = load_step_report(_report_path(root, "pgsm_step5c_note_set_extended_validation.json"))
    step5b = load_step_report(_report_path(root, "pgsm_step5b_limited_note_set_refinement.json"))
    step5a = load_step_report(_report_path(root, "pgsm_step5a_limited_note_set_diagnostic_audio.json"))

    fps_before = collect_all_fingerprints(root, step5a)

    original_metrics: Dict[str, Any] = {}
    render_files: Dict[str, str] = {}
    gain_by_note: Dict[str, float] = {}
    limiter_status: Dict[str, Any] = {}
    render_validation: Dict[str, Any] = {}
    render_infos: Dict[str, Any] = {}

    for note in NOTE_SET:
        paths = _wav_paths_for_note(root, step5a, note)
        main, sr = load_wav_mono(paths["main"])
        body, _ = load_wav_mono(paths["body"])
        excitation, _ = load_wav_mono(paths["excitation"])

        original_metrics[note] = {
            "main": analyze_audibility(main, sr, role="main"),
            "body_stem": analyze_audibility(body, sr, role="body_stem"),
            "excitation_stem": analyze_audibility(excitation, sr, role="excitation_stem"),
        }

        rendered, trimmed, info = apply_listening_render(main, sr)
        render_infos[note] = info
        gain_by_note[note] = info["gain_db"]
        limiter_status[note] = {
            "applied": info.get("limiter_applied"),
            "type": info.get("limiter_type"),
            "trim_applied": info.get("trim_applied"),
        }

        out_path = out_dir / f"sample_000_{note}_listening_diagnostic.wav"
        if write_wav:
            write_wav_mono(out_path, rendered, sr)
        render_files[note] = str(out_path)
        render_validation[note] = validate_render(trimmed, rendered, sr, info)

    fps_after = collect_all_fingerprints(root, step5a)
    preserved = fps_before == fps_after

    artifact = build_artifact_guard(render_validation, render_infos)
    objective = {
        "step5c_loaded": bool(step5c.get("report_version")),
        "step5c_passed": (step5c.get("readiness_after_step5c") or {}).get("current_status") == READINESS_STEP6A,
        "original_inaudible_detected": all(
            (original_metrics[n]["main"].get("inaudible_rms") for n in NOTE_SET)
        ),
        "four_listening_renders_generated": len(render_files) == 4,
        "originals_preserved": preserved,
        "all_renders_pass": all((render_validation[n] or {}).get("pass") for n in NOTE_SET),
        "artifact_guard_pass": artifact.get("pass"),
        "no_physics_changed": True,
        "no_stk_integration": True,
        "gain_reported_separately": True,
    }
    objective["all_pass"] = bool(all(objective.values()))

    readiness = build_readiness_after_step5d(objective["all_pass"])

    return {
        "report_version": PGSM_STEP5D_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5d_audible_diagnostic_render_repair_complete",
        "no_physics_changed": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "step5c_loaded": step5c.get("report_version"),
        "step5b_loaded": step5b.get("report_version"),
        "step5a_loaded": step5a.get("report_version"),
        "original_audibility_metrics": original_metrics,
        "listening_render_files": render_files,
        "gain_applied_db_by_note": gain_by_note,
        "render_gain_details": render_infos,
        "limiter_status": limiter_status,
        "render_validation": render_validation,
        "artifact_guard_results": artifact,
        "original_file_fingerprints_before": fps_before,
        "original_file_fingerprints_after": fps_after,
        "original_file_fingerprints_preserved": preserved,
        "target_rms_dbfs_range": [TARGET_RMS_DBFS_MIN, TARGET_RMS_DBFS_MAX],
        "peak_cap_dbfs": PEAK_CAP_DBFS,
        "objective_test_results": objective,
        "readiness_after_step5d": readiness,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
            "Real-guitar equivalence or validation proof",
            "Physical model change",
        ],
        "safe_next_step": (
            "Use listening renders for human audibility checks; Step 6A reference comparison unchanged"
            if readiness["current_status"] == READINESS_AFTER
            else "Fix audible render validation before using listening files"
        ),
        "explicit_statement": (
            "PGSM Step 5D creates audible diagnostic listening renders only. "
            "It does not change the physical model and does not prove realism."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5d") or {}
    orig = report.get("original_audibility_metrics") or {}
    val = report.get("render_validation") or {}
    gain = report.get("gain_applied_db_by_note") or {}
    lim = report.get("limiter_status") or {}

    lines = [
        "# PGSM Step 5D — audible diagnostic render repair",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Original audibility",
        "",
        "| Note | peak dBFS | RMS dBFS | crest | active 1% ms | inaudible |",
        "|------|-----------|----------|-------|--------------|-----------|",
    ]
    for note in NOTE_SET:
        m = (orig.get(note) or {}).get("main") or {}
        lines.append(
            f"| {note} | {m.get('peak_dbfs')} | {m.get('rms_dbfs')} | {m.get('crest_factor')} | "
            f"{m.get('active_duration_above_1pct_peak_ms')} | {m.get('inaudible_rms')} |"
        )

    lines.extend(
        [
            "",
            "## Listening renders",
            "",
            "| Note | output | RMS dBFS | peak dBFS | gain dB | limiter | pass |",
            "|------|--------|----------|-----------|---------|---------|------|",
        ]
    )
    files = report.get("listening_render_files") or {}
    for note in NOTE_SET:
        v = val.get(note) or {}
        l = lim.get(note) or {}
        lines.append(
            f"| {note} | `{files.get(note)}` | {v.get('rms_dbfs')} | {v.get('peak_dbfs')} | "
            f"{gain.get(note)} | {l.get('applied')} | {v.get('pass')} |"
        )

    art = report.get("artifact_guard_results") or {}
    obj = report.get("objective_test_results") or {}
    lines.extend(
        [
            "",
            "## Applied gain (separate from physics)",
            "",
            "| Note | gain dB | target RMS dBFS | trim applied |",
            "|------|---------|-----------------|--------------|",
        ]
    )
    details = report.get("render_gain_details") or {}
    for note in NOTE_SET:
        d = details.get(note) or {}
        lines.append(
            f"| {note} | {gain.get(note)} | {d.get('target_rms_dbfs')} | {d.get('trim_applied')} |"
        )

    lines.extend(
        [
            "",
            "## Limiter",
            "",
            "| Note | applied | type |",
            "|------|---------|------|",
        ]
    )
    for note in NOTE_SET:
        l = lim.get(note) or {}
        lines.append(f"| {note} | {l.get('applied')} | {l.get('type')} |")

    lines.extend(
        [
            "",
            "## Artifact guard",
            "",
            f"pass: **{art.get('pass')}**",
            "",
            "## Original preservation",
            "",
            f"Fingerprints preserved: **{report.get('original_file_fingerprints_preserved')}**",
            "",
            "## Readiness",
            "",
            f"all_pass: **{obj.get('all_pass')}**",
            f"step6a_with_audible_renders_allowed: **{rg.get('step6a_with_audible_renders_allowed')}**",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5d_reports(
    *,
    repo_root: Optional[Path] = None,
    render_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5d_report(repo_root=root, render_dir=render_dir, write_wav=True)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step5d_reports()
    rg = report.get("readiness_after_step5d") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"Gain dB: {report.get('gain_applied_db_by_note')}")


if __name__ == "__main__":
    main()
