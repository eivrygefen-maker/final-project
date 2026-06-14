#!/usr/bin/env python3
"""
STK V6.2.2 review pack builder — A4 only, max 10 WAVs (no FEM/ROM).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from body_response_synth import DEFAULT_SAMPLE_RATE, write_wav_int16  # noqa: E402
from build_sample_comparison import load_lhs_sample_entries  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v5_design_helpers import synthesize_mode_to_wav  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402
from stk_v6_2_physical_routing import (  # noqa: E402
    DEFAULT_DURATION_S,
    load_reference_modal_from_audit,
    synthesize_v6_2_physical_routing,
)
from stk_v6_2_2_onset_tail_repair import (  # noqa: E402
    V6_2_2_VARIANTS,
    compute_v622_diagnostics,
    synthesize_v6_2_2_onset_tail_repair,
)

V621_MODE = "stk_v6_2_1_soft_pluck_tail_alpha"

DEFAULT_REVIEW_OUT = REPO / "audio" / "stk_v6_2_2_review_audio"
DEFAULT_V621_SOURCE = REPO / "audio" / "stk_v6_2_diagnostic_audio"
DEFAULT_JSON_REPORT = REPO / "audio" / "debug_reports" / "stk_v6_2_2_onset_tail_repair_report.json"
DEFAULT_MD_REPORT = REPO / "audio" / "debug_reports" / "stk_v6_2_2_onset_tail_repair_report.md"
DEFAULT_SAMPLE_ID = "sample_000"
NOTE_NAME = "A4"
NOTE_HZ = 440.0
MAX_REVIEW_WAVS = 10

REVIEW_WAV_SPEC: Tuple[Tuple[str, str], ...] = (
    ("current_final_v1_A4_sample_000.wav", "baseline"),
    ("v5_alpha_s20_b80_A4_sample_000.wav", "baseline"),
    ("stk_v6_2_1_soft_pluck_tail_alpha_A4_sample_000.wav", "v621_ref"),
    ("stk_v6_2_2_single_onset_soft_tail_alpha_A4_sample_000.wav", "v622_final"),
    ("stk_v6_2_2_no_thump_body_tail_alpha_A4_sample_000.wav", "v622_final"),
    ("stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha_A4_sample_000.wav", "v622_final"),
    ("stk_v6_2_2_single_onset_soft_tail_alpha_A4_pluck_stem.wav", "v622_stem"),
    ("stk_v6_2_2_single_onset_soft_tail_alpha_A4_body_tail_stem.wav", "v622_stem"),
    ("stk_v6_2_2_no_thump_body_tail_alpha_A4_body_tail_stem.wav", "v622_stem"),
    ("stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha_A4_body_tail_stem.wav", "v622_stem"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _pick_recommended(v622_results: Mapping[str, Mapping[str, Any]]) -> str:
    # V6.3 quarantine: never recommend V6.2.2 variants
    return "none — quarantined in STK V6.3 (do not recommend V6.2.2 variants)"


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    recommended = report.get("recommended_candidate") or "none"
    lines = [
        "# STK V6.2.2 onset / tail repair report",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        f"**Scope:** sample_000, A4 only, 2.5 s",
        "",
        "**This still does not prove multi-guitar differentiation. Not solved.**",
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## Comparison with V6.2.1 soft_pluck_tail",
        "",
    ]
    v621 = report.get("v621_baseline") or {}
    v621d = v621.get("v622_diagnostics") or v621.get("diagnostics") or {}
    lines.append(
        f"- V6.2.1 soft_pluck: onset peaks={v621d.get('onset_peak_count_0_250ms')} | "
        f"second_ratio={v621d.get('second_onset_ratio')} | thump={v621d.get('thump_index_0_300ms')} | "
        f"tail_continuity={v621d.get('tail_continuity_ratio')}"
    )
    lines.extend(["", "## V6.2.2 variants", ""])
    for mode, vrec in (report.get("v622_variants") or {}).items():
        d = vrec.get("v622_diagnostics") or {}
        lines.append(f"### `{mode}`")
        lines.append(
            f"- onset peaks={d.get('onset_peak_count_0_250ms')} | second_ratio={d.get('second_onset_ratio')} | "
            f"double_risk={d.get('double_pluck_risk_score')} | coherence={d.get('onset_coherence_pass')}"
        )
        lines.append(
            f"- thump_index={d.get('thump_index_0_300ms')} | boom_disc={d.get('boom_decay_discontinuity_score')} | "
            f"drum_tap={d.get('drum_tap_risk_score')}"
        )
        lines.append(
            f"- rms 300–800={d.get('rms_300_800ms')} | 800–1500={d.get('rms_800_1500ms')} | "
            f"1500–2500={d.get('rms_1500_2500ms')} | continuity={d.get('tail_continuity_ratio')}"
        )
        for w in d.get("v622_warnings") or []:
            lines.append(f"  - ⚠ {w}")

    lines.extend(
        [
            "",
            f"## Recommended candidate: `{recommended}`",
            "",
            f"Best-effort for listening (highest tail continuity): `{report.get('best_effort_candidate', 'none')}`",
            "",
            str(report.get("recommendation_rationale", "")),
            "",
            "## Recommended listening order",
            "",
        ]
    )
    for i, item in enumerate(report.get("listening_order") or [], start=1):
        lines.append(f"{i}. `{item}`")

    lines.extend(["", "## Review pack", ""])
    lines.append(f"Directory: `{report.get('review_dir')}` ({report.get('review_wav_count')} WAVs)")
    for name in report.get("review_wav_files") or []:
        lines.append(f"- `{name}`")

    lines.extend(["", "## Limitations", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_stk_v6_2_2_review(
    *,
    repo_root: Path,
    review_dir: Path,
    v621_source_dir: Path,
    sample_id: str = DEFAULT_SAMPLE_ID,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    audit = load_audit_report(audit_path)
    modal_data = load_reference_modal_from_audit(audit, repo_root)
    samples = load_lhs_sample_entries(repo_root, max_samples=26)
    sample = next((s for s in samples if str(s["sample_id"]) == sample_id), None)
    if sample is None:
        raise RuntimeError(f"sample {sample_id} not found")
    params = normalize_sample_parameters(sample.get("parameters"))
    params["sample_id"] = sample_id

    t0 = time.perf_counter()
    generated: Dict[str, Any] = {}

    # Baselines
    for label, mode in (
        ("current_final_v1", STK_BODY_TRANSFER_FINAL_V1),
        ("v5_alpha_s20_b80", "v5_alpha_s20_b80"),
    ):
        fname = f"{label}_{NOTE_NAME}_{sample_id}.wav"
        out_path = review_dir / fname
        synthesize_mode_to_wav(
            mode=mode,
            frequency_hz=NOTE_HZ,
            note_name=NOTE_NAME,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=out_path,
            sample_parameters=params,
            repo_root=repo_root,
            sample_id=sample_id,
            experiment=label,
        )
        generated[fname] = {"kind": "baseline", "path": _rel_path(out_path, repo_root)}

    # V6.2.1 reference — copy from prior diagnostics if present, else synthesize once
    v621_fname = f"stk_v6_2_1_soft_pluck_tail_alpha_{NOTE_NAME}_{sample_id}.wav"
    v621_review = review_dir / v621_fname
    v621_src = v621_source_dir / v621_fname
    if v621_src.is_file():
        shutil.copy2(v621_src, v621_review)
    else:
        _, final_v621, _ = synthesize_v6_2_physical_routing(
            frequency_hz=NOTE_HZ,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            sample_parameters=params,
            audit=audit,
            sample_id=sample_id,
            repo_root=repo_root,
            variant=V621_MODE,
        )
        write_wav_int16(v621_review, final_v621, sample_rate, duration_s=duration_s)

    from body_response_synth import read_wav_float_mono

    v621_audio, _ = read_wav_float_mono(v621_review)
    v621_diag = compute_v622_diagnostics(
        v621_audio,
        sample_rate=sample_rate,
        frequency_hz=NOTE_HZ,
        duration_s=duration_s,
    )
    generated[v621_fname] = {"kind": "v621_ref", "path": _rel_path(v621_review, repo_root)}

    v622_results: Dict[str, Any] = {}
    for mode in V6_2_2_VARIANTS:
        stems, final, meta = synthesize_v6_2_2_onset_tail_repair(
            frequency_hz=NOTE_HZ,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            sample_parameters=params,
            audit=audit,
            sample_id=sample_id,
            repo_root=repo_root,
            variant=mode,
        )
        short = mode.replace("stk_v6_2_2_", "")
        mix_fname = f"{mode}_{NOTE_NAME}_{sample_id}.wav"
        write_wav_int16(review_dir / mix_fname, final, sample_rate, duration_s=duration_s)
        generated[mix_fname] = {"kind": "v622_final", "path": _rel_path(review_dir / mix_fname, repo_root)}
        v622_results[mode] = {
            "v622_diagnostics": meta.get("v622_diagnostics"),
            "stem_gains": meta.get("stem_gains"),
            "wav_path": _rel_path(review_dir / mix_fname, repo_root),
        }

        if mode == "stk_v6_2_2_single_onset_soft_tail_alpha":
            pluck_fname = f"stk_v6_2_2_single_onset_soft_tail_alpha_{NOTE_NAME}_pluck_stem.wav"
            write_wav_int16(review_dir / pluck_fname, stems["pluck_attack_stem"], sample_rate, duration_s=duration_s)
            generated[pluck_fname] = {"kind": "v622_stem", "path": _rel_path(review_dir / pluck_fname, repo_root)}

        body_fname = f"stk_v6_2_2_{short}_{NOTE_NAME}_body_tail_stem.wav"
        write_wav_int16(review_dir / body_fname, stems["body_tail_stem"], sample_rate, duration_s=duration_s)
        generated[body_fname] = {"kind": "v622_stem", "path": _rel_path(review_dir / body_fname, repo_root)}

    review_files = sorted(p.name for p in review_dir.glob("*.wav"))
    if len(review_files) > MAX_REVIEW_WAVS:
        raise RuntimeError(f"review pack has {len(review_files)} WAVs (max {MAX_REVIEW_WAVS})")

    recommended = _pick_recommended(v622_results)
    listening_order = [
        f"current_final_v1_{NOTE_NAME}_{sample_id}.wav",
        f"v5_alpha_s20_b80_{NOTE_NAME}_{sample_id}.wav",
        f"stk_v6_2_1_soft_pluck_tail_alpha_{NOTE_NAME}_{sample_id}.wav",
        f"stk_v6_2_2_single_onset_soft_tail_alpha_{NOTE_NAME}_{sample_id}.wav",
        f"stk_v6_2_2_no_thump_body_tail_alpha_{NOTE_NAME}_{sample_id}.wav",
        f"stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha_{NOTE_NAME}_{sample_id}.wav",
        f"stk_v6_2_2_single_onset_soft_tail_alpha_{NOTE_NAME}_pluck_stem.wav",
        f"stk_v6_2_2_single_onset_soft_tail_alpha_{NOTE_NAME}_body_tail_stem.wav",
    ]

    improvements = {}
    for mode, rec in v622_results.items():
        d = rec.get("v622_diagnostics") or {}
        improvements[mode] = {
            "second_onset_ratio_vs_v621": round(
                float(v621_diag.get("second_onset_ratio") or 0.0)
                - float(d.get("second_onset_ratio") or 0.0),
                4,
            ),
            "thump_index_vs_v621": round(
                float(v621_diag.get("thump_index_0_300ms") or 0.0)
                - float(d.get("thump_index_0_300ms") or 0.0),
                4,
            ),
            "tail_continuity_vs_v621": round(
                float(d.get("tail_continuity_ratio") or 0.0)
                - float(v621_diag.get("tail_continuity_ratio") or 0.0),
                4,
            ),
        }

    report: Dict[str, Any] = {
        "report_version": "stk_v6_2_2_onset_tail_repair_v1",
        "timestamp": _utc_now(),
        "status": "stk_v6_2_2_diagnostic_complete_not_solved",
        "sample_id": sample_id,
        "note_name": NOTE_NAME,
        "frequency_hz": NOTE_HZ,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "production_synthesis_unchanged": True,
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "review_dir": _rel_path(review_dir, repo_root),
        "review_wav_count": len(review_files),
        "review_wav_files": review_files,
        "max_review_wavs": MAX_REVIEW_WAVS,
        "v621_baseline": {
            "mode": V621_MODE,
            "wav_path": _rel_path(v621_review, repo_root),
            "v622_diagnostics": v621_diag,
        },
        "v622_variants": v622_results,
        "improvements_vs_v621": improvements,
        "recommended_candidate": recommended,
        "best_effort_candidate": (
            max(
                v622_results.items(),
                key=lambda kv: float((kv[1].get("v622_diagnostics") or {}).get("tail_continuity_ratio") or 0.0),
            )[0]
            if v622_results
            else "none"
        ),
        "recommendation_rationale": (
            "Best aggregate onset coherence, thump reduction, and tail continuity vs V6.2.1. "
            "Listening required — metrics improved but not solved."
        ),
        "listening_order": listening_order,
        "render_time_sec": round(time.perf_counter() - t0, 4),
        "limitations": [
            "sample_000 A4 only — no multi-guitar proof.",
            "Does not claim solved.",
            "Website default remains stk_body_transfer_final_v1.",
        ],
        "explicit_flags": {
            "website_default_unchanged": True,
            "no_fem_run": True,
            "no_rom_run": True,
            "multi_guitar_not_proven": True,
            "not_solved": True,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="STK V6.2.2 A4 review pack (max 10 WAVs)")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_OUT)
    parser.add_argument("--v621-source", type=Path, default=DEFAULT_V621_SOURCE)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    args = parser.parse_args()

    report = build_stk_v6_2_2_review(
        repo_root=args.repo_root,
        review_dir=args.review_dir,
        v621_source_dir=args.v621_source,
        duration_s=args.duration_s,
        audit_path=args.audit_json,
    )
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, args.md_report)
    print(f"Wrote {args.json_report}")
    print(f"Wrote {args.md_report}")
    print(f"Review pack: {args.review_dir} ({report['review_wav_count']} WAVs)")
    print(f"Recommended: {report.get('recommended_candidate')}")


if __name__ == "__main__":
    main()
