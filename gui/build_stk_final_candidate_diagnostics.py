#!/usr/bin/env python3
"""Stage 5.1H STK final candidate validation and freeze diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import (  # noqa: E402
    STK_BODY_TRANSFER_FINAL_V1,
    STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
    STK_FINAL_CANDIDATE_CANONICAL,
    STK_FINAL_DE_THUMP_CANONICAL,
    audio_timbre_vector,
    build_batch_contrast_context,
    build_body_identity_vector,
    compare_audio_to_reference,
    distance_consistency_report,
    is_v4_1_identity_space_mode,
    nearest_neighbor_preservation_report,
    requires_identity_contrast_context,
)
from body_response_synth import DEFAULT_SAMPLE_RATE, concatenate_audio_with_crossfade, synthesize_note_with_body_response  # noqa: E402
from build_sample_comparison import (  # noqa: E402
    load_lhs_sample_entries,
    m4_surrogate_model_available,
    parse_notes_arg,
    read_wav_float_mono,
    resolve_modal_data_for_sample,
    segment_metadata_from_synthesis,
    write_wav_mono,
)
from diagnostic_synthesis import average_spectral_similarity, summarize_diagnostic_mode  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stage48_timbre_decomposition_report import _attack_time_ms, _spectral_flux  # noqa: E402
from stage51h_stk_final_candidate_report import build_stage51h_final_candidate_report  # noqa: E402
from synthesis_presets import DEFAULT_SYNTHESIS_PRESET  # noqa: E402
from transient_thump_diagnostics import analyze_transient_thump  # noqa: E402

DEFAULT_OUT = REPO / "audio" / "stk_final_candidate_diagnostics"
DEFAULT_NOTES = "A2,A3,A4,E5"
V41_MODE = "modal_body_hybrid_v4_1_full"
F25_MODE = "modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75"
DEFAULT_MODES = (
    "modal_body_hybrid_v4_1_full,"
    "modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75,"
    "modal_body_hybrid_v4_1_identity_contrast_g_30_70,"
    "stk_body_transfer_final_v1,"
    "modal_body_hybrid_v4_1_identity_contrast_g_30_70_de_thump,"
    "stk_body_transfer_final_v1_de_thump_candidate"
)

# One int16 LSB after WAV round-trip
ALIAS_EQ_MAX_ABS = 1.0 / 32767.0
ALIAS_EQ_MAX_RMS = 1e-7


def _check_alias_equivalence(
    canonical: np.ndarray,
    alias: np.ndarray,
) -> Dict[str, Any]:
    a = np.asarray(canonical, dtype=np.float64)
    b = np.asarray(alias, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 1:
        return {"max_abs_diff": None, "rms_diff": None, "passed": True, "samples": n}
    diff = a[:n] - b[:n]
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff**2)))
    passed = max_abs <= ALIAS_EQ_MAX_ABS and rms <= ALIAS_EQ_MAX_RMS
    return {
        "max_abs_diff": round(max_abs, 12),
        "rms_diff": round(rms, 12),
        "passed": passed,
        "samples": n,
    }


def _segment_row(
    meta: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    note_name: str,
    frequency_hz: float,
    seg_i: int,
    audio: np.ndarray,
    sample_rate: int,
    diagnostic_mode: str,
    z_body: Optional[Mapping[str, Any]] = None,
    vs_v41: Optional[Mapping[str, Any]] = None,
    v41_transient_baseline: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = segment_metadata_from_synthesis(
        meta,
        sample=sample,
        note_name=note_name,
        frequency_hz=frequency_hz,
        seg_start=0.0,
        seg_dur=len(audio) / float(sample_rate),
        seg_i=seg_i,
        audio=audio,
        sample_rate=sample_rate,
        diagnostic_mode=diagnostic_mode,
    )
    peak = meta.get("output_peak_dbfs")
    if peak is None:
        pk = float(np.max(np.abs(audio))) if len(audio) else 0.0
        peak = 20.0 * float(np.log10(max(pk, 1e-12)))
    row["output_peak_dbfs"] = peak
    row["clipping_detected"] = float(peak) > -0.5
    row["spectral_flux"] = _spectral_flux(audio, sample_rate)
    row["attack_time_ms"] = _attack_time_ms(audio, sample_rate)
    row["fundamental_hz"] = frequency_hz
    row["timbre_vector"] = audio_timbre_vector(audio, sample_rate=sample_rate, segment_meta=row)
    if z_body is not None:
        row["body_identity_vector"] = z_body
    if vs_v41 is not None:
        row["vs_v41_reference"] = vs_v41
    if meta.get("identity_vs_v41_reference"):
        row["vs_v41_reference"] = meta["identity_vs_v41_reference"]
    baseline = v41_transient_baseline if diagnostic_mode != V41_MODE else None
    row["transient_thump"] = analyze_transient_thump(
        audio,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        baseline=baseline,
    )
    return row


def _precompute_z_bodies(
    *,
    repo_root: Path,
    samples: Sequence[Mapping[str, Any]],
    notes: Sequence[Tuple[str, float]],
    use_surrogate: bool,
) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    by_note: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for note_name, frequency_hz in notes:
        z_map: Dict[str, Mapping[str, Any]] = {}
        for sample in samples:
            sid = str(sample["sample_id"])
            params = normalize_sample_parameters(sample.get("parameters"))
            params = {**params, "sample_id": sid}
            modal_data, _ = resolve_modal_data_for_sample(repo_root, sample, use_surrogate=use_surrogate)
            z_map[sid] = build_body_identity_vector(
                parameters=params,
                modal_data=modal_data,
                frequency_hz=frequency_hz,
                repo_root=repo_root,
                sample_id=sid,
            )
        by_note[note_name] = z_map
    return by_note


def build_stk_final_candidate_diagnostics(
    *,
    repo_root: Path,
    out_dir: Path,
    notes: Sequence[Tuple[str, float]],
    modes: Sequence[str],
    max_samples: int = 10,
    duration_s: float = 1.2,
    silence_s: float = 0.2,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    use_surrogate: bool = True,
    listening_review: Optional[Mapping[str, Any]] = None,
    vm_validated: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_lhs_sample_entries(repo_root, max_samples=max_samples)
    if not samples:
        raise RuntimeError("no LHS samples")

    z_bodies_by_note = _precompute_z_bodies(
        repo_root=repo_root,
        samples=samples,
        notes=notes,
        use_surrogate=use_surrogate,
    )
    contrast_context_by_note = {
        note: build_batch_contrast_context(z_map) for note, z_map in z_bodies_by_note.items()
    }

    mode_summaries: Dict[str, Any] = {}
    distance_samples_by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    distance_samples_by_mode_note: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        m: {n: [] for n, _ in notes} for m in modes
    }
    nn_by_mode: Dict[str, Any] = {}
    nn_by_mode_note: Dict[str, Dict[str, Any]] = {m: {} for m in modes}
    stitched: List[str] = []
    v41_audio_cache: Dict[Tuple[str, str], np.ndarray] = {}
    v41_transient_by_note_sample: Dict[Tuple[str, str], Dict[str, Any]] = {}
    final_canonical_cache: Dict[Tuple[str, str], np.ndarray] = {}
    de_thump_canonical_cache: Dict[Tuple[str, str], np.ndarray] = {}
    alias_equivalence: List[Dict[str, Any]] = []

    for mode in modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        note_rows: List[Dict[str, Any]] = []
        for note_name, frequency_hz in notes:
            segments: List[np.ndarray] = []
            seg_rows: List[Dict[str, Any]] = []
            contrast_ctx_note = contrast_context_by_note.get(note_name) or {}
            for sample in samples:
                sid = str(sample["sample_id"])
                params = normalize_sample_parameters(sample.get("parameters"))
                params = {**params, "sample_id": sid}
                if requires_identity_contrast_context(mode):
                    params = {**params, "identity_contrast_context": contrast_ctx_note.get(sid)}
                modal_data, modal_source = resolve_modal_data_for_sample(
                    repo_root, sample, use_surrogate=use_surrogate
                )
                z_body = z_bodies_by_note[note_name][sid]
                wav = mode_dir / f"{note_name}_{sid}.wav"
                meta = synthesize_note_with_body_response(
                    frequency_hz=frequency_hz,
                    note_name=note_name,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    modal_data=modal_data,
                    output_wav=wav,
                    diagnostic_mode=mode,
                    sample_parameters=params,
                    modal_source=modal_source,
                    synthesis_preset=DEFAULT_SYNTHESIS_PRESET,
                    repo_root=repo_root,
                    sample_id=sid,
                )
                audio, _ = read_wav_float_mono(wav)
                segments.append(audio)

                vs_v41: Optional[Dict[str, Any]] = None
                if mode == V41_MODE:
                    v41_audio_cache[(note_name, sid)] = audio.copy()
                    v41_transient_by_note_sample[(note_name, sid)] = analyze_transient_thump(
                        audio,
                        sample_rate=sample_rate,
                        frequency_hz=frequency_hz,
                    )
                elif is_v4_1_identity_space_mode(mode):
                    ref = v41_audio_cache.get((note_name, sid))
                    if ref is not None:
                        vs_v41 = compare_audio_to_reference(audio, ref)

                if mode == STK_FINAL_CANDIDATE_CANONICAL:
                    final_canonical_cache[(note_name, sid)] = audio.copy()
                elif mode == STK_BODY_TRANSFER_FINAL_V1:
                    canon = final_canonical_cache.get((note_name, sid))
                    if canon is not None:
                        eq = _check_alias_equivalence(canon, audio)
                        alias_equivalence.append(
                            {
                                "note": note_name,
                                "sample_id": sid,
                                "pair": "final_v1_vs_g30_70",
                                **eq,
                            }
                        )
                elif mode == STK_FINAL_DE_THUMP_CANONICAL:
                    de_thump_canonical_cache[(note_name, sid)] = audio.copy()
                elif mode == STK_BODY_TRANSFER_FINAL_V1_DE_THUMP:
                    canon = de_thump_canonical_cache.get((note_name, sid))
                    if canon is not None:
                        eq = _check_alias_equivalence(canon, audio)
                        alias_equivalence.append(
                            {
                                "note": note_name,
                                "sample_id": sid,
                                "pair": "de_thump_v1_vs_g30_70_de_thump",
                                **eq,
                            }
                        )

                seg = _segment_row(
                    meta,
                    sample=sample,
                    note_name=note_name,
                    frequency_hz=frequency_hz,
                    seg_i=len(seg_rows),
                    audio=audio,
                    sample_rate=sample_rate,
                    diagnostic_mode=mode,
                    z_body=z_body if is_v4_1_identity_space_mode(mode) else None,
                    vs_v41=vs_v41,
                    v41_transient_baseline=v41_transient_by_note_sample.get((note_name, sid)),
                )
                seg_rows.append(seg)
                distance_samples_by_mode[mode].append(
                    {
                        "sample_id": sid,
                        "z_body": z_body,
                        "timbre": seg.get("timbre_vector") or [],
                    }
                )
                distance_samples_by_mode_note[mode][note_name].append(
                    {
                        "sample_id": sid,
                        "z_body": z_body,
                        "timbre": seg.get("timbre_vector") or [],
                    }
                )
            stitch = out_dir / f"{note_name}_{mode}_10_guitars.wav"
            write_wav_mono(
                stitch,
                concatenate_audio_with_crossfade(segments, sample_rate=sample_rate, silence_ms=silence_s * 1000),
                sample_rate,
            )
            stitched.append(str(stitch))
            note_rows.append(
                {
                    "note": note_name,
                    "frequency_hz": frequency_hz,
                    "segments": seg_rows,
                    "average_spectral_similarity": average_spectral_similarity(segments),
                }
            )
        mode_summaries[mode] = summarize_diagnostic_mode(note_rows, diagnostic_mode=mode)
        for nr in note_rows:
            nk = nr["note"]
            if nk in (mode_summaries[mode].get("notes") or {}):
                mode_summaries[mode]["notes"][nk]["segments"] = nr["segments"]
        nn_by_mode[mode] = nearest_neighbor_preservation_report(distance_samples_by_mode[mode])
        for note_name, _ in notes:
            rows = distance_samples_by_mode_note[mode][note_name]
            nn_by_mode_note[mode][note_name] = nearest_neighbor_preservation_report(rows)

    distance_by_mode = {
        mode: distance_consistency_report(rows) for mode, rows in distance_samples_by_mode.items()
    }
    distance_by_mode_note: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        distance_by_mode_note[mode] = {
            note: distance_consistency_report(distance_samples_by_mode_note[mode][note])
            for note, _ in notes
        }

    alias_all_passed = bool(alias_equivalence) and all(r.get("passed") for r in alias_equivalence)

    manifest = {
        "stage": "5.1H",
        "out_dir": str(out_dir),
        "sample_ids": [str(s["sample_id"]) for s in samples],
        "notes": [n for n, _ in notes],
        "note_frequencies_hz": {n: f for n, f in notes},
        "modes": list(modes),
        "final_alias": STK_BODY_TRANSFER_FINAL_V1,
        "final_canonical_mode": STK_FINAL_CANDIDATE_CANONICAL,
        "alias_equivalence": alias_equivalence,
        "alias_equivalence_all_passed": alias_all_passed,
        "use_surrogate": use_surrogate,
        "fem_launched": False,
        "stitched_files": stitched,
        "distance_consistency_by_mode": distance_by_mode,
        "distance_consistency_by_mode_note": distance_by_mode_note,
        "nearest_neighbor_by_mode": nn_by_mode,
        "nearest_neighbor_by_mode_note": nn_by_mode_note,
        "listening_review": dict(listening_review) if listening_review else None,
        "vm_validated": bool(vm_validated),
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    report_dir = repo_root / "audio" / "debug_reports"
    build_stage51h_final_candidate_report(
        mode_summaries=mode_summaries,
        distance_by_mode=distance_by_mode,
        distance_by_mode_note=distance_by_mode_note,
        nn_by_mode=nn_by_mode,
        nn_by_mode_note=nn_by_mode_note,
        notes=[n for n, _ in notes],
        modes=modes,
        build_manifest=manifest,
        alias_equivalence=alias_equivalence,
        out_json=report_dir / "stage51h_stk_final_candidate_report.json",
        out_md=report_dir / "stage51h_stk_final_candidate_report.md",
        listening_review=listening_review,
        vm_validated=vm_validated,
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5.1H STK final candidate freeze validation")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--notes", type=str, default=DEFAULT_NOTES)
    parser.add_argument("--modes", type=str, default=DEFAULT_MODES)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--silence", type=float, default=0.2)
    parser.add_argument("--no-surrogate", action="store_true")
    parser.add_argument(
        "--vm-validated",
        action="store_true",
        help="Mark run as VM-validated (enables metric-based de-thump switch if criteria met)",
    )
    parser.add_argument(
        "--listening-heard-drum-thump",
        action="store_true",
        help="Listening review: artificial drum-like thump heard on final v1",
    )
    parser.add_argument(
        "--listening-de-thump-more-natural",
        action="store_true",
        help="Listening review: de-thump variant sounds more natural than final v1",
    )
    args = parser.parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    notes = parse_notes_arg(args.notes)
    use_surrogate = not args.no_surrogate
    listening_review: Optional[Dict[str, Any]] = None
    if args.listening_heard_drum_thump or args.listening_de_thump_more_natural:
        listening_review = {}
        if args.listening_heard_drum_thump:
            listening_review["heard_drum_thump"] = True
        if args.listening_de_thump_more_natural:
            listening_review["de_thump_sounds_more_natural"] = True
    if use_surrogate and not m4_surrogate_model_available(REPO):
        raise RuntimeError("M4 surrogate missing — use --no-surrogate for offline tests")
    manifest = build_stk_final_candidate_diagnostics(
        repo_root=REPO,
        out_dir=args.out_dir,
        notes=notes,
        modes=modes,
        max_samples=args.max_samples,
        duration_s=args.duration,
        silence_s=args.silence,
        use_surrogate=use_surrogate,
        listening_review=listening_review,
        vm_validated=args.vm_validated,
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
