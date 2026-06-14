#!/usr/bin/env python3
"""STK V6.4 current-anchor repair review pack (A4, max 8 WAVs)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from body_response_synth import DEFAULT_SAMPLE_RATE, write_wav_int16  # noqa: E402
from build_sample_comparison import load_lhs_sample_entries  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v5_design_helpers import synthesize_mode_to_wav  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402
from stk_v6_2_physical_routing import DEFAULT_DURATION_S, load_reference_modal_from_audit  # noqa: E402
from stk_v6_3_artifact_quarantine import ARTIFACT_QUARANTINE, V6_3_MODE  # noqa: E402
from stk_v6_4_current_anchor_repair import (  # noqa: E402
    SOUND_BASE_REJECTED,
    V6_4_MODES,
    compute_v64_metrics,
    evaluate_v64_candidate,
    render_current_final_v1_anchor,
    repair_current_anchor,
)

DEFAULT_REVIEW_OUT = REPO / "audio" / "stk_v6_4_review_audio"
DEFAULT_V63_REVIEW = REPO / "audio" / "stk_v6_3_review_audio"
DEFAULT_V622_REVIEW = REPO / "audio" / "stk_v6_2_2_review_audio"
DEFAULT_JSON = REPO / "audio" / "debug_reports" / "stk_v6_4_current_anchor_repair_report.json"
DEFAULT_MD = REPO / "audio" / "debug_reports" / "stk_v6_4_current_anchor_repair_report.md"
SAMPLE_ID = "sample_000"
NOTE_NAME = "A4"
NOTE_HZ = 440.0
MAX_WAVS = 8


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(p)


def _pick_recommended(variants: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    best: Optional[str] = None
    best_score = -1e9
    for mode, rec in variants.items():
        acc = rec.get("acceptance") or {}
        if acc.get("candidate_acceptance_status") == "rejected":
            continue
        m = rec.get("metrics") or {}
        score = (
            (2.0 if acc.get("thump_improved_vs_anchor") else 0.0)
            + float(m.get("current_identity_similarity") or 0.0)
            - float(m.get("drum_tap_risk_score") or 0.0)
            + float(m.get("tail_continuity_ratio") or 0.0)
        )
        if score > best_score:
            best_score = score
            best = mode
    if best is None:
        return None
    rec = variants.get(best) or {}
    if (rec.get("acceptance") or {}).get("candidate_acceptance_status") == "rejected":
        return None
    return best


def write_md(report: Mapping[str, Any], path: Path) -> None:
    recommended = report.get("recommended_candidate")
    lines = [
        "# STK V6.4 current-anchor repair report",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        "**V6.4 is not a final model and does not prove multi-guitar differentiation.**",
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## Why V6.2 / V6.3 rejected as sound bases",
        "",
        str((report.get("sound_base_rejected") or {}).get("reason_summary", "")),
        "",
        "## Anchor",
        "",
        f"`current_final_v1` ({STK_BODY_TRANSFER_FINAL_V1}) used as primary sonic anchor.",
        "",
        "## Anchor metrics",
        "",
    ]
    am = report.get("anchor_metrics") or {}
    lines.append(
        f"- drum_tap={am.get('drum_tap_risk_score')} | thump_band={am.get('thump_band_rms_250_700_0_120ms')} | "
        f"tail_continuity={am.get('tail_continuity_ratio')} | end_noise={am.get('end_noise_score')}"
    )
    lines.extend(["", "## V6.4 candidates", ""])
    for mode, vrec in (report.get("v64_variants") or {}).items():
        m = vrec.get("metrics") or {}
        a = vrec.get("acceptance") or {}
        lines.append(f"### `{mode}` — **{a.get('candidate_acceptance_status')}**")
        lines.append(
            f"- identity_similarity={m.get('current_identity_similarity')} | drum_tap={m.get('drum_tap_risk_score')} | "
            f"thump_band={m.get('thump_band_rms_250_700_0_120ms')} | tail={m.get('tail_continuity_ratio')}"
        )
        lines.append(
            f"- end_noise={m.get('end_noise_score')} | fade_ok={m.get('final_200ms_fade_ok')} | "
            f"double_onset={m.get('double_onset_fail')}"
        )
    lines.extend(["", f"## Recommended candidate: `{recommended}`", "", "## Review pack", ""])
    lines.append(f"`{report.get('review_dir')}` ({report.get('review_wav_count')} WAVs)")
    for f in report.get("review_wav_files") or []:
        lines.append(f"- `{f}`")
    lines.extend(["", "## Listening order", ""])
    for i, item in enumerate(report.get("listening_order") or [], start=1):
        lines.append(f"{i}. `{item}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_v64_review(
    *,
    repo_root: Path,
    review_dir: Path,
    v63_review: Path,
    v622_review: Path,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    audit = load_audit_report(audit_path)
    modal = load_reference_modal_from_audit(audit, repo_root)
    sample = next(s for s in load_lhs_sample_entries(repo_root) if s["sample_id"] == SAMPLE_ID)
    params = normalize_sample_parameters(sample.get("parameters"))
    params["sample_id"] = SAMPLE_ID

    t0 = time.perf_counter()

    anchor, sr, anchor_meta = render_current_final_v1_anchor(
        frequency_hz=NOTE_HZ,
        note_name=NOTE_NAME,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal,
        sample_parameters=params,
        repo_root=repo_root,
        sample_id=SAMPLE_ID,
    )
    anchor_name = f"current_final_v1_{NOTE_NAME}_{SAMPLE_ID}.wav"
    write_wav_int16(review_dir / anchor_name, anchor, sr, duration_s=duration_s)
    anchor_metrics = compute_v64_metrics(anchor, sample_rate=sr, duration_s=duration_s)

    for label, mode, exp in (
        ("v5_alpha_s20_b80", "v5_alpha_s20_b80", "v5_alpha_s20_b80"),
    ):
        fname = f"{label}_{NOTE_NAME}_{SAMPLE_ID}.wav"
        synthesize_mode_to_wav(
            mode=mode,
            frequency_hz=NOTE_HZ,
            note_name=NOTE_NAME,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal,
            output_wav=review_dir / fname,
            sample_parameters=params,
            repo_root=repo_root,
            sample_id=SAMPLE_ID,
            experiment=exp,
        )

    v63_name = f"{V6_3_MODE}_{NOTE_NAME}_{SAMPLE_ID}.wav"
    v63_src = v63_review / v63_name
    if v63_src.is_file():
        shutil.copy2(v63_src, review_dir / v63_name)

    v622_ref = f"stk_v6_4_rejected_reference_v622_single_onset_{NOTE_NAME}_{SAMPLE_ID}.wav"
    v622_src = v622_review / f"stk_v6_2_2_single_onset_soft_tail_alpha_{NOTE_NAME}_{SAMPLE_ID}.wav"
    include_v622 = False
    if v622_src.is_file() and len(list(review_dir.glob("*.wav"))) + 1 <= MAX_WAVS - 2:
        include_v622 = True

    variants: Dict[str, Any] = {}
    for mode in V6_4_MODES:
        repaired, atk_dbg, tail_dbg, meta = repair_current_anchor(
            anchor,
            sample_rate=sr,
            duration_s=duration_s,
            variant=mode,
        )
        short = mode.replace("stk_v6_4_", "").replace("_alpha", "")
        final_name = f"{mode}_{NOTE_NAME}_{SAMPLE_ID}.wav"
        write_wav_int16(review_dir / final_name, repaired, sr, duration_s=duration_s)

        if mode == "stk_v6_4_current_anchor_soft_attack_alpha":
            dbg_name = f"stk_v6_4_current_anchor_soft_attack_alpha_{NOTE_NAME}_attack_window_debug.wav"
            write_wav_int16(review_dir / dbg_name, atk_dbg, sr, duration_s=min(0.35, duration_s))
        if mode == "stk_v6_4_current_anchor_sustain_smooth_alpha":
            dbg_name = f"stk_v6_4_current_anchor_sustain_smooth_alpha_{NOTE_NAME}_tail_window_debug.wav"
            tail_dur = min(1.3, duration_s - 1.0)
            write_wav_int16(review_dir / dbg_name, tail_dbg, sr, duration_s=max(0.5, tail_dur))

        metrics = compute_v64_metrics(
            repaired, sample_rate=sr, duration_s=duration_s, anchor=anchor
        )
        acceptance = evaluate_v64_candidate(metrics, anchor_metrics=anchor_metrics)
        variants[mode] = {
            "metrics": metrics,
            "acceptance": acceptance,
            "meta": meta,
            "wav_path": _rel(review_dir / final_name, repo_root),
        }

    if include_v622 and len(list(review_dir.glob("*.wav"))) < MAX_WAVS:
        shutil.copy2(v622_src, review_dir / v622_ref)

    review_files = sorted(p.name for p in review_dir.glob("*.wav"))
    if len(review_files) > MAX_WAVS:
        keep = {
            anchor_name,
            f"v5_alpha_s20_b80_{NOTE_NAME}_{SAMPLE_ID}.wav",
            v63_name,
            f"stk_v6_4_current_anchor_soft_attack_alpha_{NOTE_NAME}_{SAMPLE_ID}.wav",
            f"stk_v6_4_current_anchor_sustain_smooth_alpha_{NOTE_NAME}_{SAMPLE_ID}.wav",
            f"stk_v6_4_current_anchor_soft_attack_alpha_{NOTE_NAME}_attack_window_debug.wav",
            f"stk_v6_4_current_anchor_sustain_smooth_alpha_{NOTE_NAME}_tail_window_debug.wav",
        }
        if include_v622:
            keep.add(v622_ref)
        for p in review_dir.glob("*.wav"):
            if p.name not in keep:
                p.unlink()
        review_files = sorted(p.name for p in review_dir.glob("*.wav"))

    recommended = _pick_recommended(variants)
    listening = [
        anchor_name,
        f"stk_v6_4_current_anchor_soft_attack_alpha_{NOTE_NAME}_{SAMPLE_ID}.wav",
        f"stk_v6_4_current_anchor_sustain_smooth_alpha_{NOTE_NAME}_{SAMPLE_ID}.wav",
        f"v5_alpha_s20_b80_{NOTE_NAME}_{SAMPLE_ID}.wav",
        v63_name,
    ]

    return {
        "report_version": "stk_v6_4_current_anchor_repair_v1",
        "timestamp": _utc(),
        "status": "stk_v6_4_diagnostic_complete_not_solved",
        "sample_id": SAMPLE_ID,
        "note_name": NOTE_NAME,
        "frequency_hz": NOTE_HZ,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "production_synthesis_unchanged": True,
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "artifact_quarantine": ARTIFACT_QUARANTINE,
        "sound_base_rejected": SOUND_BASE_REJECTED,
        "v63_rejected_as_sound_base": True,
        "anchor_mode": STK_BODY_TRANSFER_FINAL_V1,
        "anchor_metrics": anchor_metrics,
        "anchor_meta": {"render": "current_final_v1", "path": _rel(review_dir / anchor_name, repo_root)},
        "v64_variants": variants,
        "recommended_candidate": recommended,
        "do_not_recommend_modes": ARTIFACT_QUARANTINE.get("rejected_modes", []) + [V6_3_MODE],
        "review_dir": _rel(review_dir, repo_root),
        "review_wav_count": len(review_files),
        "review_wav_files": review_files,
        "max_review_wavs": MAX_WAVS,
        "listening_order": listening,
        "render_time_sec": round(time.perf_counter() - t0, 4),
        "limitations": [
            "V6.4 is not a final model and does not prove multi-guitar differentiation.",
            "Not solved — listening required.",
        ],
        "explicit_flags": {
            "website_default_unchanged": True,
            "no_fem_run": True,
            "no_rom_run": True,
            "multi_guitar_not_proven": True,
            "not_solved": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="STK V6.4 current-anchor repair review pack")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_OUT)
    parser.add_argument("--v63-review-dir", type=Path, default=DEFAULT_V63_REVIEW)
    parser.add_argument("--v622-review-dir", type=Path, default=DEFAULT_V622_REVIEW)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    report = build_v64_review(
        repo_root=args.repo_root,
        review_dir=args.review_dir,
        v63_review=args.v63_review_dir,
        v622_review=args.v622_review_dir,
        duration_s=args.duration_s,
    )
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(report, args.md_report)
    print(f"Wrote {args.json_report}")
    print(f"Wrote {args.md_report}")
    print(f"Review: {args.review_dir} ({report['review_wav_count']} WAVs)")
    print(f"Recommended: {report.get('recommended_candidate')}")


if __name__ == "__main__":
    main()
