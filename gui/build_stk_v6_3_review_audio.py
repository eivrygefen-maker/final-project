#!/usr/bin/env python3
"""
STK V6.3 artifact quarantine scan + clean candidate review pack (A4, max 8 WAVs).
"""
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
from stk_v6_3_artifact_quarantine import (  # noqa: E402
    ARTIFACT_QUARANTINE,
    REJECTED_V622_MODES,
    V6_3_MODE,
    evaluate_v63_acceptance,
    scan_artifacts,
    scan_wav_file,
    synthesize_v6_3_clean_pluck_body,
)

DEFAULT_REVIEW_OUT = REPO / "audio" / "stk_v6_3_review_audio"
DEFAULT_V622_REVIEW = REPO / "audio" / "stk_v6_2_2_review_audio"
DEFAULT_JSON_REPORT = REPO / "audio" / "debug_reports" / "stk_v6_3_artifact_quarantine_report.json"
DEFAULT_MD_REPORT = REPO / "audio" / "debug_reports" / "stk_v6_3_artifact_quarantine_report.md"
DEFAULT_SAMPLE_ID = "sample_000"
NOTE_NAME = "A4"
NOTE_HZ = 440.0
MAX_REVIEW_WAVS = 8

SCAN_MANIFEST = (
    ("current_final_v1_A4_sample_000.wav", False, "baseline"),
    ("v5_alpha_s20_b80_A4_sample_000.wav", False, "baseline"),
    ("stk_v6_2_1_soft_pluck_tail_alpha_A4_sample_000.wav", False, "v621_baseline"),
    ("stk_v6_2_2_single_onset_soft_tail_alpha_A4_sample_000.wav", False, "v622_rejected"),
    ("stk_v6_2_2_no_thump_body_tail_alpha_A4_sample_000.wav", False, "v622_rejected"),
    ("stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha_A4_sample_000.wav", False, "v622_rejected"),
    ("stk_v6_2_2_single_onset_soft_tail_alpha_A4_pluck_stem.wav", False, "v622_pluck_stem"),
    ("stk_v6_2_2_single_onset_soft_tail_alpha_A4_body_tail_stem.wav", True, "v622_body_tail_stem"),
    ("stk_v6_2_2_no_thump_body_tail_alpha_A4_body_tail_stem.wav", True, "v622_body_tail_stem"),
    ("stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha_A4_body_tail_stem.wav", True, "v622_body_tail_stem"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _root_cause_statement(scanner_rows: List[Mapping[str, Any]]) -> str:
    body_tail_fails = [
        r for r in scanner_rows if r.get("is_body_tail_stem") and r.get("delayed_body_event_fail")
    ]
    pluck_fails = [
        r
        for r in scanner_rows
        if "pluck_stem" in str(r.get("file_label", "")) and r.get("double_pluck_fail")
    ]
    end_fails = [r for r in scanner_rows if r.get("end_click_or_gate_fail")]
    parts = [
        "Root cause: V6.2.2 body_tail stems use delayed resonator/ramp paths that produce "
        "impulsive low/mid pulses around 140–240 ms (second musical event), not a smooth tail.",
    ]
    if body_tail_fails:
        parts.append(
            f"Confirmed in {len(body_tail_fails)} V6.2.2 body_tail stem scan(s): "
            "delayed_body_event_fail=true."
        )
    if pluck_fails:
        parts.append("Pluck stems also show double-onset risk in some V6.2.x paths.")
    else:
        parts.append("Primary failure is body_tail delayed pulse, not pluck_attack alone.")
    if end_fails:
        parts.append(f"End noise/gating detected in {len(end_fails)} file(s).")
    parts.append("V6.2.2 variants are quarantined; V6.3 rebuilds from unified excitation without resonator IR.")
    return " ".join(parts)


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    q = report.get("artifact_quarantine") or {}
    lines = [
        "# STK V6.3 artifact quarantine report",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        "**V6.3 does not prove multi-guitar differentiation. Not solved.**",
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## Quarantined modes (do not recommend)",
        "",
    ]
    for mode in q.get("rejected_modes") or []:
        reasons = (q.get("reason") or {}).get(mode, [])
        lines.append(f"- `{mode}`: {', '.join(reasons)}")
    lines.append(f"\nAllowed future use: **{q.get('allowed_future_use')}**")
    lines.extend(["", "## Root cause", "", str(report.get("root_cause_statement", "")), ""])
    lines.extend(["", "## Artifact scanner", ""])
    for row in report.get("artifact_scanner") or []:
        lines.append(f"### `{row.get('file_label')}`")
        lines.append(
            f"- double_pluck_fail={row.get('double_pluck_fail')} | thump_fail={row.get('thump_fail')} | "
            f"delayed_body_event_fail={row.get('delayed_body_event_fail')} | "
            f"tail_collapse_fail={row.get('tail_collapse_fail')} | end_fail={row.get('end_click_or_gate_fail')}"
        )
        lines.append(
            f"- onset_peaks={row.get('onset_peak_count_0_250ms')} | second_ratio={row.get('second_onset_ratio')} | "
            f"body_tail_peaks_80_350={row.get('body_tail_peak_count_80_350ms')} | "
            f"body_tail_peak_ms={row.get('body_tail_peak_time_ms')}"
        )

    lines.extend(["", "## Artifact location summary", ""])
    lines.append(f"- Bad artifact in body_tail stem: **{report.get('bad_artifact_in_body_tail_stem')}**")
    lines.append(f"- Bad artifact in pluck stem: **{report.get('bad_artifact_in_pluck_stem')}**")
    lines.append(f"- Bad artifact at file ending: **{report.get('bad_artifact_at_file_ending')}**")

    acc = report.get("clean_candidate_acceptance") or {}
    lines.extend(
        [
            "",
            f"## Clean candidate: `{V6_3_MODE}`",
            "",
            f"**Acceptance status:** `{acc.get('candidate_acceptance_status', 'unknown')}`",
            "",
        ]
    )
    for k, v in (acc.get("checks") or {}).items():
        lines.append(f"- {k}: {v}")

    lines.extend(["", "## Review pack", ""])
    lines.append(f"`{report.get('review_dir')}` — {report.get('review_wav_count')} WAVs (max {MAX_REVIEW_WAVS})")
    for f in report.get("review_wav_files") or []:
        lines.append(f"- `{f}`")

    lines.extend(["", "## Listening order", ""])
    for i, item in enumerate(report.get("listening_order") or [], start=1):
        lines.append(f"{i}. `{item}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_v63_quarantine_report(
    *,
    repo_root: Path,
    review_dir: Path,
    v622_review_dir: Path,
    sample_id: str = DEFAULT_SAMPLE_ID,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    review_dir = Path(review_dir)
    v622_review_dir = Path(v622_review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    audit = load_audit_report(audit_path)
    modal_data = load_reference_modal_from_audit(audit, repo_root)
    samples = load_lhs_sample_entries(repo_root, max_samples=26)
    sample = next(s for s in samples if str(s["sample_id"]) == sample_id)
    params = normalize_sample_parameters(sample.get("parameters"))
    params["sample_id"] = sample_id

    t0 = time.perf_counter()
    scanner_rows: List[Dict[str, Any]] = []

    for fname, is_body_tail, category in SCAN_MANIFEST:
        src = v622_review_dir / fname
        if not src.is_file() and fname.startswith("current_final_v1"):
            src = review_dir / fname
        if not src.is_file() and fname.startswith("v5_alpha"):
            synthesize_mode_to_wav(
                mode="v5_alpha_s20_b80",
                frequency_hz=NOTE_HZ,
                note_name=NOTE_NAME,
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                output_wav=v622_review_dir / fname if v622_review_dir.is_dir() else review_dir / fname,
                sample_parameters=params,
                repo_root=repo_root,
                sample_id=sample_id,
                experiment="v5_alpha_s20_b80",
            )
            src = (v622_review_dir if (v622_review_dir / fname).is_file() else review_dir) / fname
        if not src.is_file() and fname.startswith("current_final_v1"):
            out = review_dir / fname
            synthesize_mode_to_wav(
                mode=STK_BODY_TRANSFER_FINAL_V1,
                frequency_hz=NOTE_HZ,
                note_name=NOTE_NAME,
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                output_wav=out,
                sample_parameters=params,
                repo_root=repo_root,
                sample_id=sample_id,
                experiment="current_final_v1",
            )
            src = out
        if src.is_file():
            row = scan_wav_file(
                src,
                duration_s=duration_s,
                file_label=fname,
                is_body_tail_stem=is_body_tail,
            )
            row["category"] = category
            row["source_path"] = _rel(src, repo_root)
            scanner_rows.append(row)

    bad_body_tail = any(
        r.get("is_body_tail_stem") and (r.get("delayed_body_event_fail") or r.get("thump_fail"))
        for r in scanner_rows
    )
    bad_pluck = any(
        "pluck_stem" in str(r.get("file_label", "")) and r.get("double_pluck_fail")
        for r in scanner_rows
    )
    bad_end = any(r.get("end_click_or_gate_fail") for r in scanner_rows)

    stems, final, pre_fin, meta = synthesize_v6_3_clean_pluck_body(
        frequency_hz=NOTE_HZ,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal_data,
        sample_parameters=params,
        audit=audit,
        sample_id=sample_id,
        repo_root=repo_root,
    )

    v63_final_name = f"{V6_3_MODE}_{NOTE_NAME}_{sample_id}.wav"
    write_wav_int16(review_dir / v63_final_name, final, sample_rate, duration_s=duration_s)
    write_wav_int16(
        review_dir / f"{V6_3_MODE}_{NOTE_NAME}_pluck_stem.wav",
        stems["pluck_stem"],
        sample_rate,
        duration_s=duration_s,
    )
    write_wav_int16(
        review_dir / f"{V6_3_MODE}_{NOTE_NAME}_body_tail_stem.wav",
        stems["body_tail_stem"],
        sample_rate,
        duration_s=duration_s,
    )
    write_wav_int16(
        review_dir / f"{V6_3_MODE}_{NOTE_NAME}_pre_finalize.wav",
        pre_fin,
        sample_rate,
        duration_s=duration_s,
    )

    for baseline in (
        "current_final_v1_A4_sample_000.wav",
        "v5_alpha_s20_b80_A4_sample_000.wav",
    ):
        dst = review_dir / baseline
        if not dst.is_file():
            src = v622_review_dir / baseline
            if src.is_file():
                shutil.copy2(src, dst)
            elif baseline.startswith("current_final_v1"):
                synthesize_mode_to_wav(
                    mode=STK_BODY_TRANSFER_FINAL_V1,
                    frequency_hz=NOTE_HZ,
                    note_name=NOTE_NAME,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    modal_data=modal_data,
                    output_wav=dst,
                    sample_parameters=params,
                    repo_root=repo_root,
                    sample_id=sample_id,
                    experiment="current_final_v1",
                )
            else:
                synthesize_mode_to_wav(
                    mode="v5_alpha_s20_b80",
                    frequency_hz=NOTE_HZ,
                    note_name=NOTE_NAME,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    modal_data=modal_data,
                    output_wav=dst,
                    sample_parameters=params,
                    repo_root=repo_root,
                    sample_id=sample_id,
                    experiment="v5_alpha_s20_b80",
                )

    v622_ref = "stk_v6_2_2_single_onset_soft_tail_alpha_A4_sample_000.wav"
    if not (review_dir / v622_ref).is_file():
        src = v622_review_dir / v622_ref
        if src.is_file():
            shutil.copy2(src, review_dir / v622_ref)

    bad_ref = "stk_v6_3_artifact_reference_bad_body_tail_stem.wav"
    bad_src = v622_review_dir / "stk_v6_2_2_single_onset_soft_tail_alpha_A4_body_tail_stem.wav"
    if bad_src.is_file() and not (review_dir / bad_ref).is_file():
        shutil.copy2(bad_src, review_dir / bad_ref)

    clean_pluck_diag = scan_artifacts(
        stems["pluck_stem"], sample_rate=sample_rate, duration_s=duration_s,
        file_label=f"{V6_3_MODE}_pluck_stem", is_body_tail_stem=False,
    )
    clean_body_diag = scan_artifacts(
        stems["body_tail_stem"], sample_rate=sample_rate, duration_s=duration_s,
        file_label=f"{V6_3_MODE}_body_tail_stem", is_body_tail_stem=True,
    )
    clean_final_diag = scan_artifacts(
        final, sample_rate=sample_rate, duration_s=duration_s,
        file_label=v63_final_name, is_body_tail_stem=False,
    )

    final_v1_row = next((r for r in scanner_rows if "current_final_v1" in r.get("file_label", "")), {})
    v622_row = next(
        (r for r in scanner_rows if "stk_v6_2_2_single_onset_soft_tail_alpha_A4_sample_000" in r.get("file_label", "")),
        {},
    )
    acceptance = evaluate_v63_acceptance(clean_final_diag, final_v1_diag=final_v1_row, v622_diag=v622_row)

    review_files = sorted(p.name for p in review_dir.glob("*.wav"))
    if len(review_files) > MAX_REVIEW_WAVS:
        keep = {
            "current_final_v1_A4_sample_000.wav",
            "v5_alpha_s20_b80_A4_sample_000.wav",
            v622_ref,
            v63_final_name,
            f"{V6_3_MODE}_{NOTE_NAME}_pluck_stem.wav",
            f"{V6_3_MODE}_{NOTE_NAME}_body_tail_stem.wav",
            f"{V6_3_MODE}_{NOTE_NAME}_pre_finalize.wav",
            bad_ref,
        }
        for p in review_dir.glob("*.wav"):
            if p.name not in keep:
                p.unlink()
        review_files = sorted(p.name for p in review_dir.glob("*.wav"))

    listening_order = [
        "current_final_v1_A4_sample_000.wav",
        "v5_alpha_s20_b80_A4_sample_000.wav",
        v622_ref,
        v63_final_name,
        f"{V6_3_MODE}_{NOTE_NAME}_pluck_stem.wav",
        f"{V6_3_MODE}_{NOTE_NAME}_body_tail_stem.wav",
    ]

    report: Dict[str, Any] = {
        "report_version": "stk_v6_3_artifact_quarantine_v1",
        "timestamp": _utc_now(),
        "status": "stk_v6_3_quarantine_and_clean_candidate_complete_not_solved",
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
        "artifact_quarantine": ARTIFACT_QUARANTINE,
        "quarantined_v622_modes": list(REJECTED_V622_MODES),
        "recommended_candidate": None,
        "do_not_recommend_modes": ARTIFACT_QUARANTINE["rejected_modes"],
        "artifact_scanner": scanner_rows,
        "root_cause_statement": _root_cause_statement(scanner_rows),
        "bad_artifact_in_body_tail_stem": bad_body_tail,
        "bad_artifact_in_pluck_stem": bad_pluck,
        "bad_artifact_at_file_ending": bad_end,
        "clean_candidate": {
            "mode": V6_3_MODE,
            "meta": meta,
            "diagnostics": {
                "final_mix": clean_final_diag,
                "pluck_stem": clean_pluck_diag,
                "body_tail_stem": clean_body_diag,
            },
        },
        "clean_candidate_acceptance": acceptance,
        "candidate_acceptance_status": acceptance.get("candidate_acceptance_status"),
        "review_dir": _rel(review_dir, repo_root),
        "review_wav_count": len(review_files),
        "review_wav_files": review_files,
        "max_review_wavs": MAX_REVIEW_WAVS,
        "listening_order": listening_order,
        "render_time_sec": round(time.perf_counter() - t0, 4),
        "limitations": [
            "V6.3 does not prove multi-guitar differentiation.",
            "V6.2.1/V6.2.2 modes quarantined — baseline comparison only.",
            "Not solved until listening confirms clean candidate.",
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
    parser = argparse.ArgumentParser(description="STK V6.3 artifact quarantine + clean candidate")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_OUT)
    parser.add_argument("--v622-review-dir", type=Path, default=DEFAULT_V622_REVIEW)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    args = parser.parse_args()

    report = build_v63_quarantine_report(
        repo_root=args.repo_root,
        review_dir=args.review_dir,
        v622_review_dir=args.v622_review_dir,
        duration_s=args.duration_s,
        audit_path=args.audit_json,
    )
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, args.md_report)
    print(f"Wrote {args.json_report}")
    print(f"Wrote {args.md_report}")
    print(f"Review pack: {args.review_dir} ({report['review_wav_count']} WAVs)")
    print(f"Acceptance: {report.get('candidate_acceptance_status')}")
    print("Quarantined modes — do not recommend.")


if __name__ == "__main__":
    main()
