#!/usr/bin/env python3
"""
STK V6.2.1 pluck/tail/balance repair diagnostics (sample_000, no FEM/ROM batch).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from body_response_synth import DEFAULT_SAMPLE_RATE, write_wav_int16  # noqa: E402
from build_sample_comparison import load_lhs_sample_entries, parse_notes_arg  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v5_design_helpers import synthesize_mode_to_wav  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402
from stk_v6_2_physical_routing import (  # noqa: E402
    DEFAULT_DURATION_S,
    STK_V6_2_MODE,
    V6_2_1_MODES,
    V6_2_1_VARIANTS,
    compute_balance_diagnostics,
    load_reference_modal_from_audit,
    metrics_for_stems_and_final,
    synthesize_v6_2_physical_routing,
)

DEFAULT_OUT = REPO / "audio" / "stk_v6_2_diagnostic_audio"
DEFAULT_JSON_REPORT = REPO / "audio" / "debug_reports" / "stk_v6_2_1_balance_repair_report.json"
DEFAULT_MD_REPORT = REPO / "audio" / "debug_reports" / "stk_v6_2_1_balance_repair_report.md"
DEFAULT_NOTES = "A3,A4,E5"
DEFAULT_SAMPLE_ID = "sample_000"

V621_STEM_SUFFIXES = (
    "pluck_attack_stem",
    "direct_string_short_stem",
    "top_radiation_stem",
    "soundhole_air_stem",
    "cavity_body_tail_stem",
    "final_mix",
)

COMPARISON_MODES = (
    ("current_final_v1", STK_BODY_TRANSFER_FINAL_V1),
    ("v5_alpha_s20_b80", "v5_alpha_s20_b80"),
    (STK_V6_2_MODE, STK_V6_2_MODE),
    *[(m, m) for m in V6_2_1_VARIANTS],
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _pick_recommended_variant(note_rows: Sequence[Mapping[str, Any]]) -> str:
    scores: Dict[str, float] = {m: 0.0 for m in V6_2_1_VARIANTS}
    for row in note_rows:
        variants = row.get("v6_2_1_variants") or {}
        for mode, vrec in variants.items():
            bal = vrec.get("balance_diagnostics") or {}
            score = (
                float(bal.get("tail_audibility_score") or 0.0) * 2.0
                + float(bal.get("sustain_body_presence_score") or 0.0) * 1.5
                - float(bal.get("drum_tap_risk_score") or 0.0) * 1.2
                - max(0.0, float(bal.get("attack_to_tail_ratio") or 0.0) - 6.0) * 0.08
            )
            if bal.get("balance_pass"):
                score += 0.5
            scores[mode] = scores.get(mode, 0.0) + score
    if not scores:
        return "stk_v6_2_1_balanced_tail_alpha"
    return max(scores, key=lambda k: scores[k])


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rec = report.get("recommended_variant") or "stk_v6_2_1_balanced_tail_alpha"
    lines = [
        "# STK V6.2.1 pluck / tail / balance repair report",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        f"**Sample:** {report.get('sample_id')}",
        "",
        "**This still does not prove multi-guitar differentiation.**",
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## What changed from V6.2",
        "",
    ]
    for item in report.get("changes_from_v6_2") or []:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            f"## Recommended variant: `{rec}`",
            "",
            str(report.get("recommendation_rationale", "")),
            "",
            "## Recommended listening order",
            "",
        ]
    )
    for i, item in enumerate(report.get("listening_order") or [], start=1):
        lines.append(f"{i}. `{item}`")

    lines.extend(["", "## Attack / body / tail metrics", ""])
    for row in report.get("notes") or []:
        lines.append(f"### {row.get('note_name')} ({row.get('frequency_hz')} Hz)")
        v62 = row.get("v6_2_baseline_balance") or {}
        lines.append(
            f"- V6.2 baseline: attack/tail={v62.get('attack_to_tail_ratio')} | "
            f"tail_rms={v62.get('tail_rms_1_2p5s')} | drum_tap={v62.get('drum_tap_risk_score')}"
        )
        for mode in V6_2_1_VARIANTS:
            vrec = (row.get("v6_2_1_variants") or {}).get(mode) or {}
            bal = vrec.get("balance_diagnostics") or {}
            lines.append(
                f"- `{mode}`: attack/tail={bal.get('attack_to_tail_ratio')} | "
                f"tail_rms={bal.get('tail_rms_1_2p5s')} | tail_aud={bal.get('tail_audibility_score')} | "
                f"norm={vrec.get('norm_method')}"
            )
            for w in bal.get("balance_warnings") or []:
                lines.append(f"  - ⚠ {w}")

    lines.extend(["", "## Normalization methods", ""])
    for mode, desc in (report.get("normalization_by_mode") or {}).items():
        lines.append(f"- `{mode}`: {desc}")

    lines.extend(["", "## Limitations", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_stk_v6_2_1_diagnostics(
    *,
    repo_root: Path,
    out_dir: Path,
    sample_id: str = DEFAULT_SAMPLE_ID,
    notes: Sequence[Tuple[str, float]] = (),
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = load_audit_report(audit_path)
    modal_data = load_reference_modal_from_audit(audit, repo_root)

    samples = load_lhs_sample_entries(repo_root, max_samples=26)
    sample = next((s for s in samples if str(s["sample_id"]) == sample_id), None)
    if sample is None:
        raise RuntimeError(f"sample {sample_id} not found in lhs_pool")
    params = normalize_sample_parameters(sample.get("parameters"))
    params["sample_id"] = sample_id

    note_rows: List[Dict[str, Any]] = []
    normalization_by_mode: Dict[str, str] = {}

    for note_name, frequency_hz in notes:
        t0 = time.perf_counter()
        wav_paths: Dict[str, str] = {}
        comparisons: Dict[str, Any] = {}

        for comp_label, comp_mode in COMPARISON_MODES:
            if comp_mode == STK_V6_2_MODE:
                v62_path = out_dir / f"stk_v6_2_physical_routing_alpha_{note_name}_{sample_id}.wav"
                if v62_path.is_file():
                    from body_response_synth import read_wav_float_mono

                    final_v62, _ = read_wav_float_mono(v62_path)
                    _, _, v62_meta = synthesize_v6_2_physical_routing(
                        frequency_hz=frequency_hz,
                        duration_s=duration_s,
                        sample_rate=sample_rate,
                        modal_data=modal_data,
                        sample_parameters=params,
                        audit=audit,
                        sample_id=sample_id,
                        repo_root=repo_root,
                        variant=STK_V6_2_MODE,
                    )
                    v62_balance = compute_balance_diagnostics(
                        final_v62,
                        sample_rate=sample_rate,
                        frequency_hz=frequency_hz,
                        duration_s=duration_s,
                    )
                    wav_paths[STK_V6_2_MODE] = _rel_path(v62_path, repo_root)
                    comparisons[STK_V6_2_MODE] = {
                        "balance_diagnostics": v62_balance,
                        "norm_method": v62_meta.get("norm_method"),
                        "wav_path": wav_paths[STK_V6_2_MODE],
                        "preserved_existing_output": True,
                    }
                    normalization_by_mode[STK_V6_2_MODE] = str(v62_meta.get("normalization", ""))
                else:
                    stems, final, meta = synthesize_v6_2_physical_routing(
                        frequency_hz=frequency_hz,
                        duration_s=duration_s,
                        sample_rate=sample_rate,
                        modal_data=modal_data,
                        sample_parameters=params,
                        audit=audit,
                        sample_id=sample_id,
                        repo_root=repo_root,
                        variant=STK_V6_2_MODE,
                    )
                    fname = f"stk_v6_2_physical_routing_alpha_{note_name}_{sample_id}.wav"
                    write_wav_int16(out_dir / fname, final, sample_rate, duration_s=duration_s)
                    wav_paths[STK_V6_2_MODE] = _rel_path(out_dir / fname, repo_root)
                    comparisons[STK_V6_2_MODE] = {
                        "balance_diagnostics": meta.get("balance_diagnostics"),
                        "norm_method": meta.get("norm_method"),
                        "wav_path": wav_paths[STK_V6_2_MODE],
                        "preserved_existing_output": False,
                    }
                    normalization_by_mode[STK_V6_2_MODE] = str(meta.get("normalization", ""))
                continue

            if comp_mode in V6_2_1_VARIANTS:
                stems, final, meta = synthesize_v6_2_physical_routing(
                    frequency_hz=frequency_hz,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    modal_data=modal_data,
                    sample_parameters=params,
                    audit=audit,
                    sample_id=sample_id,
                    repo_root=repo_root,
                    variant=comp_mode,
                )
                prefix = f"stk_v6_2_1_{comp_mode.replace('stk_v6_2_1_', '')}_{note_name}_{sample_id}"
                stem_paths: Dict[str, str] = {}
                for suffix in V621_STEM_SUFFIXES:
                    if suffix not in stems:
                        continue
                    stem_fname = f"{prefix}_{suffix}.wav"
                    write_wav_int16(out_dir / stem_fname, stems[suffix], sample_rate, duration_s=duration_s)
                    stem_paths[suffix] = _rel_path(out_dir / stem_fname, repo_root)
                mix_fname = f"{comp_mode}_{note_name}_{sample_id}.wav"
                write_wav_int16(out_dir / mix_fname, final, sample_rate, duration_s=duration_s)
                wav_paths[comp_mode] = _rel_path(out_dir / mix_fname, repo_root)
                metrics = metrics_for_stems_and_final(
                    stems,
                    sample_rate=sample_rate,
                    frequency_hz=frequency_hz,
                    duration_s=duration_s,
                    final_meta=meta,
                )
                comparisons[comp_mode] = {
                    "balance_diagnostics": meta.get("balance_diagnostics"),
                    "norm_method": meta.get("norm_method"),
                    "normalization": meta.get("normalization"),
                    "stem_gains": meta.get("stem_gains"),
                    "wav_path": wav_paths[comp_mode],
                    "stem_paths": stem_paths,
                    "final_mix_metrics": metrics.get("final_mix"),
                }
                normalization_by_mode[comp_mode] = str(meta.get("normalization", ""))
                continue

            comp_name = f"{comp_label}_{note_name}_{sample_id}.wav"
            comp_path = out_dir / comp_name
            synthesize_mode_to_wav(
                mode=comp_mode,
                frequency_hz=frequency_hz,
                note_name=note_name,
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                output_wav=comp_path,
                sample_parameters=params,
                repo_root=repo_root,
                sample_id=sample_id,
                experiment=comp_label,
            )
            wav_paths[comp_label] = _rel_path(comp_path, repo_root)
            comparisons[comp_label] = {"wav_path": wav_paths[comp_label]}

        v62_bal = (comparisons.get(STK_V6_2_MODE) or {}).get("balance_diagnostics") or {}
        v621_variants = {
            m: comparisons[m]
            for m in V6_2_1_VARIANTS
            if m in comparisons
        }

        note_rows.append(
            {
                "note_name": note_name,
                "frequency_hz": frequency_hz,
                "render_time_sec": round(time.perf_counter() - t0, 4),
                "wav_paths": wav_paths,
                "comparisons": comparisons,
                "v6_2_baseline_balance": v62_bal,
                "v6_2_1_variants": v621_variants,
            }
        )

    recommended = _pick_recommended_variant(note_rows)
    listening_order = [
        f"current_final_v1_A4_{sample_id}.wav",
        f"v5_alpha_s20_b80_A4_{sample_id}.wav",
        f"stk_v6_2_physical_routing_alpha_A4_{sample_id}.wav",
        f"{recommended}_A4_{sample_id}.wav",
        f"stk_v6_2_1_balanced_tail_alpha_A4_{sample_id}_cavity_body_tail_stem.wav",
        f"stk_v6_2_1_soft_pluck_tail_alpha_A4_{sample_id}.wav",
        f"stk_v6_2_1_more_string_body_alpha_A4_{sample_id}.wav",
        f"current_final_v1_E5_{sample_id}.wav",
        f"stk_v6_2_physical_routing_alpha_E5_{sample_id}.wav",
        f"{recommended}_E5_{sample_id}.wav",
    ]

    report: Dict[str, Any] = {
        "report_version": "stk_v6_2_1_balance_repair_v1",
        "timestamp": _utc_now(),
        "status": "stk_v6_2_1_balance_repair_diagnostic_complete",
        "sample_id": sample_id,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "stk_v5_behavior_unchanged": True,
        "production_synthesis_unchanged": True,
        "diagnostic_modes": list(V6_2_1_VARIANTS.keys()),
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "output_dir": _rel_path(out_dir, repo_root),
        "changes_from_v6_2": [
            "Reduced pluck_attack_stem gain and derivative click weight",
            "Smoother attack envelope; direct_string decay extended to ~180–350 ms with HF damping",
            "Raised cavity_body_tail_stem, top_radiation_stem, soundhole_air_stem sustain gains",
            "Sustain-window RMS normalization (200–800 ms) with attack soft-limit instead of pluck-dominated peak norm",
        ],
        "recommended_variant": recommended,
        "recommendation_rationale": (
            f"Selected `{recommended}` by aggregate tail audibility, body presence, "
            "and lower drum-tap risk across A3/A4/E5."
        ),
        "normalization_by_mode": normalization_by_mode,
        "notes": note_rows,
        "listening_order": listening_order,
        "limitations": [
            "V6.2.1 does not prove multi-guitar differentiation yet.",
            "Diagnostic-only variants; website default remains stk_body_transfer_final_v1.",
            "Original V6.2 outputs preserved when already on disk.",
        ],
        "explicit_flags": {
            "website_default_unchanged": True,
            "no_fem_run": True,
            "no_rom_run": True,
            "stk_v5_behavior_unchanged": True,
            "multi_guitar_not_proven": True,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="STK V6.2.1 balance repair diagnostics")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-id", type=str, default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--notes", type=str, default=DEFAULT_NOTES)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    args = parser.parse_args()

    notes = parse_notes_arg(args.notes)
    report = build_stk_v6_2_1_diagnostics(
        repo_root=args.repo_root,
        out_dir=args.out_dir,
        sample_id=args.sample_id,
        notes=notes,
        duration_s=args.duration_s,
        audit_path=args.audit_json,
    )
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, args.md_report)
    print(f"Wrote {args.json_report}")
    print(f"Wrote {args.md_report}")
    print(f"Recommended variant: {report.get('recommended_variant')}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
