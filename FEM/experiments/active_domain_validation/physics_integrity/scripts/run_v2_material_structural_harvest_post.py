#!/usr/bin/env python3
"""
Post-solve structural comparison for v2_material_structural_harvest_extension.

Report-only relative to frozen v2 physics: MAC matrices, Hungarian assignment, subspace overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_material_structural_compare import compare_baseline_to_material, case_spectrum_report
from v2_sensitivity_common import (
    HARVEST_EXT_DIAG,
    STRUCTURAL_HARVEST_HI,
    STRUCTURAL_HARVEST_LO,
    VALIDATION_STATUS_JSON,
    get_validated_reduced_u_to_W_map,
    harvest_ext_result_json,
    load_harvest_extension_manifest,
    write_json,
)

BASELINE_HARVEST_ID = "baseline_coupled_v2_material_reference"
HARVEST_REPORT_JSON = HARVEST_EXT_DIAG / "v2_material_structural_harvest_extension_report.json"
HARVEST_REPORT_MD = HARVEST_EXT_DIAG / "v2_material_structural_harvest_extension_report.md"


def _write_md(report: Dict[str, Any]) -> None:
    lines = [
        "# v2 material structural harvest extension",
        "",
        f"Suite: `{report.get('suite')}`",
        f"Harvest band (Hz): `{report.get('harvest_band_hz')}`",
        f"Baseline reference: `{report.get('baseline_reference_id')}`",
        "",
        "## Per-case spectrum",
        "",
        "| sample | top | back | n_conv | n_struct | n_acous/mixed | struct f range (Hz) | v2 conv |",
        "|--------|-----|------|--------|----------|---------------|---------------------|---------|",
    ]
    for row in report.get("case_spectrum_reports") or []:
        ma = row.get("material_assignment") or {}
        fr = row.get("structural_mode_frequency_range_hz")
        fr_s = f"{fr[0]:.2f}–{fr[1]:.2f}" if fr else "—"
        lines.append(
            f"| {row.get('sample_id')} | {ma.get('top_wood_id')} | {ma.get('back_wood_id')} | "
            f"{row.get('number_of_converged_modes')} | {row.get('number_of_structural_dominated_modes')} | "
            f"{row.get('number_of_acoustic_or_mixed_modes')} | {fr_s} | {row.get('v2_converged')} |"
        )
    lines.extend(["", "## Material structural comparison", ""])
    for comp in report.get("material_comparisons") or []:
        sid = comp.get("sample_id", "?")
        lines.append(f"### {sid}")
        if comp.get("status"):
            lines.append(f"- status: `{comp['status']}`")
            lines.append("")
            continue
        lines.append(
            f"- recommended pass: `{comp.get('recommended_material_structural_pass')}`"
        )
        lines.append(f"- coverage / families / subspace: "
                     f"`{comp.get('coverage_ok')}` / `{comp.get('families_ok')}` / `{comp.get('subspace_pass')}`")
        lines.append(f"- n_struct baseline/material: `{comp.get('n_baseline_structural')}` / `{comp.get('n_material_structural')}`")
        lines.append(f"- n_high_confidence_assigned: `{comp.get('n_high_confidence_assigned')}`")
        if comp.get("shape_preserved_large_frequency_shifts"):
            lines.append("- shape-preserved large Δf (MAC≥0.85):")
            for note in comp["shape_preserved_large_frequency_shifts"]:
                lines.append(
                    f"  - {note['f_baseline_hz']:.3f} → {note['f_material_hz']:.3f} Hz "
                    f"(Δf={note['delta_f_hz']:+.3f}, MAC={note['structural_MAC']:.4f})"
                )
        lines.append("")
    lines.extend(
        [
            "## Gate status (until explicit promotion)",
            "",
            f"- `material_structural_branch_validation_pass` = `{report.get('material_structural_branch_validation_pass')}`",
            f"- `recommended_all_materials_pass` = `{report.get('recommended_all_materials_pass')}`",
            "",
            report.get("promotion_note", ""),
        ]
    )
    HARVEST_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest extension post-analysis")
    parser.add_argument(
        "--apply-promotion",
        action="store_true",
        help="Set material_structural_branch_validation_pass=PASS if all materials meet criterion",
    )
    args = parser.parse_args()

    manifest = load_harvest_extension_manifest()
    policy = manifest.get("harvest_policy") or {}
    band_lo = float(policy.get("harvest_lo_hz", STRUCTURAL_HARVEST_LO))
    band_hi = float(policy.get("harvest_hi_hz", STRUCTURAL_HARVEST_HI))
    criterion = manifest.get("validation_criterion_after_post") or {}

    u_to_W, map_meta = get_validated_reduced_u_to_W_map()
    if u_to_W is None or not map_meta.get("valid"):
        print("[harvest_post] FATAL: validated reduced u_to_W unavailable", file=sys.stderr)
        return 2
    n_W = int(map_meta.get("vector_length", map_meta.get("n_reduced_W", 112100)))

    case_reports: List[Dict[str, Any]] = []
    for sample in manifest.get("samples") or []:
        sid = str(sample["id"])
        rp = harvest_ext_result_json(sid)
        if not rp:
            case_reports.append(
                {
                    "sample_id": sid,
                    "material_assignment": sample.get("materials"),
                    "harvest_band_hz": [band_lo, band_hi],
                    "status": "missing_solve_artifacts",
                }
            )
            continue
        solve = json.loads(rp.read_text(encoding="utf-8"))
        case_reports.append(case_spectrum_report(sample, solve, band_lo=band_lo, band_hi=band_hi))

    material_ids = [
        sid
        for sid in (manifest.get("sample_ids") or [])
        if sid != BASELINE_HARVEST_ID
    ]
    comparisons: List[Dict[str, Any]] = []
    def _sample(sid: str) -> Dict[str, Any]:
        for s in manifest.get("samples") or []:
            if str(s.get("id")) == sid:
                return s
        return {"id": sid}

    for sid in material_ids:
        sample = _sample(sid)
        comparisons.append(
            compare_baseline_to_material(
                baseline_id=BASELINE_HARVEST_ID,
                material_id=sid,
                sample=sample,
                u_to_W=u_to_W,
                n_W=n_W,
                band_lo=band_lo,
                band_hi=band_hi,
                criterion=criterion,
            )
        )

    mat_comps = [c for c in comparisons if c.get("status") is None]
    recommended_all = bool(mat_comps) and all(
        c.get("recommended_material_structural_pass") for c in mat_comps
    )
    gate_pass = "PASS" if (args.apply_promotion and recommended_all) else "Pending"

    report: Dict[str, Any] = {
        "suite": manifest.get("suite"),
        "frozen_formulation": manifest.get("frozen_formulation"),
        "baseline_reference_id": BASELINE_HARVEST_ID,
        "harvest_band_hz": [band_lo, band_hi],
        "u_to_W_source": map_meta,
        "case_spectrum_reports": case_reports,
        "material_comparisons": comparisons,
        "recommended_all_materials_pass": recommended_all,
        "material_structural_branch_validation_pass": gate_pass,
        "promotion_note": (
            "Expanded harvest post-analysis complete. Structural/production gates remain Pending "
            "unless --apply-promotion and all materials meet the documented criterion."
        ),
    }
    write_json(HARVEST_REPORT_JSON, report)
    _write_md(report)

    if VALIDATION_STATUS_JSON.is_file():
        status = json.loads(VALIDATION_STATUS_JSON.read_text(encoding="utf-8"))
        status["v2_material_structural_harvest_extension"] = {
            "report_json": str(HARVEST_REPORT_JSON),
            "recommended_all_materials_pass": recommended_all,
            "harvest_band_hz": [band_lo, band_hi],
        }
        if args.apply_promotion and recommended_all:
            status["material_structural_branch_validation_pass"] = "PASS"
            status["material_species_validation_pass"] = "PASS"
            status["production_parameter_validation_pass"] = "PASS"
            status["production_parameter_coverage_pass"] = "PASS"
            status["lhs_promotion_blocked"] = False
        write_json(VALIDATION_STATUS_JSON, status)
    print(f"[harvest_post] wrote {HARVEST_REPORT_JSON}", flush=True)
    print(f"[harvest_post] recommended_all={recommended_all} gate={gate_pass}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
