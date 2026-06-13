#!/usr/bin/env python3
"""
STK V6.2 single-guitar physical routing diagnostics (sample_000, no FEM/ROM batch).
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
from build_sample_comparison import (  # noqa: E402
    load_lhs_sample_entries,
    parse_notes_arg,
)
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v5_design_helpers import synthesize_mode_to_wav  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402
from stk_v6_2_physical_routing import (  # noqa: E402
    DEFAULT_DURATION_S,
    STK_V6_2_MODE,
    load_reference_modal_from_audit,
    metrics_for_stems_and_final,
    synthesize_v6_2_physical_routing,
)

DEFAULT_OUT = REPO / "audio" / "stk_v6_2_diagnostic_audio"
DEFAULT_JSON_REPORT = REPO / "audio" / "debug_reports" / "stk_v6_2_single_guitar_report.json"
DEFAULT_MD_REPORT = REPO / "audio" / "debug_reports" / "stk_v6_2_single_guitar_report.md"
DEFAULT_NOTES = "A3,A4,E5"
DEFAULT_SAMPLE_ID = "sample_000"

STEM_SUFFIXES = (
    "pluck_attack_stem",
    "direct_string_short_stem",
    "bridge_body_stem",
    "top_radiation_stem",
    "soundhole_air_stem",
    "cavity_body_tail_stem",
    "final_mix",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# STK V6.2 single-guitar physical routing report",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        f"**Sample:** {report.get('sample_id')}",
        "",
        "**V6.2 does not prove multi-guitar differentiation yet.**",
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## Stage 2 readiness context",
        "",
        str(report.get("readiness_note", "")),
        "",
        "## Listening order (recommended)",
        "",
    ]
    for i, item in enumerate(report.get("listening_order") or [], start=1):
        lines.append(f"{i}. `{item}`")

    lines.extend(["", "## Note results", ""])
    for row in report.get("notes") or []:
        lines.append(f"### {row.get('note_name')} ({row.get('frequency_hz')} Hz)")
        fm = row.get("final_mix_metrics") or {}
        lines.append(
            f"- string_dominance (sustain window): {fm.get('string_dominance_ratio')} | "
            f"body/string: {fm.get('body_to_string_energy_ratio')} | "
            f"E5 metallicity warning: {fm.get('e5_metallicity_warning')}"
        )
        lines.append(f"- tail 1–2.5s: {fm.get('tail_energy_1s_to_2p5s')} | "
                     f"metallicity: {fm.get('high_note_metallicity_index')}")

    lines.extend(["", "## Feature provenance", ""])
    prov = report.get("feature_provenance_used") or {}
    for name, rec in sorted(prov.items()):
        lines.append(
            f"- `{name}`: status={rec.get('status')} per_sample={rec.get('per_sample')} "
            f"used_for={rec.get('used_for')}"
        )

    lines.extend(["", "## Limitations", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_stk_v6_2_diagnostics(
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
    feature_provenance: Dict[str, Any] = {}

    for note_name, frequency_hz in notes:
        t0 = time.perf_counter()
        stems, final, meta = synthesize_v6_2_physical_routing(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            sample_parameters=params,
            audit=audit,
            sample_id=sample_id,
            repo_root=repo_root,
        )
        feature_provenance = meta.get("feature_provenance_used") or feature_provenance
        metrics = metrics_for_stems_and_final(
            stems,
            sample_rate=sample_rate,
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            final_meta=meta,
        )

        wav_paths: Dict[str, str] = {}
        for suffix in STEM_SUFFIXES:
            if suffix not in stems:
                continue
            fname = f"stk_v6_2_{note_name}_{sample_id}_{suffix}.wav"
            fpath = out_dir / fname
            write_wav_int16(fpath, stems[suffix], sample_rate, duration_s=duration_s)
            try:
                wav_paths[suffix] = str(fpath.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                wav_paths[suffix] = str(fpath)

        for comp_mode, comp_label in (
            (STK_BODY_TRANSFER_FINAL_V1, "current_final_v1"),
            ("v5_alpha_s20_b80", "v5_alpha_s20_b80"),
        ):
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
            try:
                wav_paths[comp_label] = str(comp_path.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                wav_paths[comp_label] = str(comp_path)

        v62_name = f"stk_v6_2_physical_routing_alpha_{note_name}_{sample_id}.wav"
        write_wav_int16(out_dir / v62_name, final, sample_rate, duration_s=duration_s)
        try:
            wav_paths["stk_v6_2_physical_routing_alpha"] = str(
                (out_dir / v62_name).relative_to(repo_root)
            ).replace("\\", "/")
        except ValueError:
            wav_paths["stk_v6_2_physical_routing_alpha"] = str(out_dir / v62_name)

        note_rows.append(
            {
                "note_name": note_name,
                "frequency_hz": frequency_hz,
                "render_time_sec": round(time.perf_counter() - t0, 4),
                "wav_paths": wav_paths,
                "stem_metrics": metrics.get("stems"),
                "final_mix_metrics": metrics.get("final_mix"),
                "v6_2_meta": {
                    k: meta[k]
                    for k in (
                        "stem_gains",
                        "stem_contribution_ratios",
                        "string_dominance_ratio_sustain_window",
                        "body_to_string_energy_ratio_sustain",
                        "pluck_params",
                        "back_side_stem",
                        "limitations",
                    )
                    if k in meta
                },
            }
        )

    listening_order = [
        f"current_final_v1_A4_{sample_id}.wav",
        f"v5_alpha_s20_b80_A4_{sample_id}.wav",
        f"stk_v6_2_physical_routing_alpha_A4_{sample_id}.wav",
        f"current_final_v1_E5_{sample_id}.wav",
        f"stk_v6_2_physical_routing_alpha_E5_{sample_id}.wav",
        f"stk_v6_2_A4_{sample_id}_pluck_attack_stem.wav",
        f"stk_v6_2_A4_{sample_id}_top_radiation_stem.wav",
        f"stk_v6_2_A4_{sample_id}_soundhole_air_stem.wav",
        f"stk_v6_2_A4_{sample_id}_cavity_body_tail_stem.wav",
    ]

    try:
        out_dir_rel = str(out_dir.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        out_dir_rel = str(out_dir)

    report: Dict[str, Any] = {
        "report_version": "stk_v6_2_single_guitar_v1",
        "timestamp": _utc_now(),
        "status": "stk_v6_2_physical_routing_alpha_diagnostic_not_solved",
        "sample_id": sample_id,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_audio_synthesis_for_audit_only": False,
        "no_fem_run": True,
        "no_rom_run": True,
        "stk_v5_behavior_unchanged": True,
        "production_synthesis_unchanged": True,
        "diagnostic_mode": STK_V6_2_MODE,
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "output_dir": out_dir_rel,
        "readiness_note": audit.get("stage2_readiness", {}).get("multi_guitar_differentiation"),
        "feature_provenance_used": feature_provenance,
        "notes": note_rows,
        "listening_order": listening_order,
        "limitations": [
            "V6.2 does not prove multi-guitar differentiation yet.",
            "Reference modal catalog used for routing (reference_shared features).",
            "back_side_stem omitted — no per-sample back radiation data.",
            "Website default remains stk_body_transfer_final_v1.",
        ],
        "explicit_flags": {
            "website_default_unchanged": True,
            "no_fem_run": True,
            "no_rom_run": True,
            "stk_v5_behavior_unchanged": True,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="STK V6.2 single-guitar routing diagnostics")
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
    report = build_stk_v6_2_diagnostics(
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
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
