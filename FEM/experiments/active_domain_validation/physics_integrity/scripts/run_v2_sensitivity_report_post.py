#!/usr/bin/env python3
"""
Post-process v2_sensitivity_validation summary (no new solves).

Reads saved sample artifacts, reports structural_branches_in_band, evaluates
structural trends, and fills missing acoustic energy table fields.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_sensitivity_common import (
    COUPLED_BASELINE_F_HZ,
    COUPLED_BASELINE_P_FRAC,
    DIAG_DIR,
    ENERGY_ACOUSTIC_THRESHOLD,
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_SUMMARY_JSON,
    SENS_ROOT,
    SUMMARY_JSON,
    V2_ROOT,
    VALIDATION_STATUS_JSON,
    hz_result_tag,
    is_acoustic_branch,
    load_manifest,
    load_production_manifest,
    structural_branches_summary,
    write_json,
    write_validation_status,
)

ACOUSTIC_STABLE_DELTA_HZ = 2.0
STRUCT_MATCH_TOL_HZ = 8.0


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _best_result_json(case_dir: Path) -> Optional[Path]:
    results_dir = case_dir / "results"
    if not results_dir.is_dir():
        return None
    paths = sorted(results_dir.glob("result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _structural_from_in_band(in_band: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return structural_branches_summary(in_band, limit=8)


def _reload_sample_structural(sample_id: str, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing = row.get("structural_branches_in_band")
    if isinstance(existing, list) and existing:
        return list(existing)
    case_dir = SENS_ROOT / "samples" / sample_id
    result_path = _best_result_json(case_dir)
    if result_path:
        solve = _load_json(result_path) or {}
        if solve.get("structural_branches_in_band"):
            return list(solve["structural_branches_in_band"])
        in_band = solve.get("in_band_modes") or []
        if in_band:
            return _structural_from_in_band(in_band)
    energy = _load_json(case_dir / "diagnostics" / "mode_energy_summary.json")
    if energy and energy.get("modes"):
        in_band = [
            m
            for m in energy["modes"]
            if 220.0 <= float(m.get("frequency_hz", 0)) <= 300.0
        ]
        return _structural_from_in_band(in_band)
    return []


def _enrich_acoustic_energies(row: Dict[str, Any], branch: Dict[str, Any]) -> None:
    row["nearest_acoustic_branch"] = branch
    row["nearest_acoustic_f_hz"] = float(branch["frequency_hz"])
    row["p_frac_energy_phys"] = float(branch.get("p_frac_energy_phys", COUPLED_BASELINE_P_FRAC))
    row["acoustic_modal_energy_phys"] = branch.get("acoustic_modal_energy_phys")
    row["structural_modal_energy_phys"] = branch.get("structural_modal_energy_phys")
    row["mass_cross_term_phys"] = branch.get("mass_cross_term_phys")
    row["mode_class_physical_energy"] = branch.get("mode_class_physical_energy", "acoustic_dominated")
    if row.get("ingest_only"):
        row["delta_f_hz_from_coupled_baseline"] = 0.0


def _enrich_baseline_coupled(samples: Dict[str, Dict[str, Any]]) -> None:
    row = samples.setdefault(
        "baseline_coupled_v2",
        {
            "sample_id": "baseline_coupled_v2",
            "ingest_only": True,
            "status": "ok",
            "mesh_gates_skipped": True,
            "v2_converged": True,
        },
    )
    case_dir = V2_ROOT / "physical_coupling_enabled"
    for tag in (hz_result_tag(COUPLED_BASELINE_F_HZ), hz_result_tag(244.39)):
        result = _load_json(case_dir / "results" / f"result_{tag}.json")
        if not result:
            continue
        branch = result.get("nearest_acoustic_mode") or result.get("acoustic_branch_by_energy")
        if not branch and result.get("in_band_modes"):
            acoustic = [m for m in result["in_band_modes"] if is_acoustic_branch(m)]
            branch = max(acoustic, key=lambda m: float(m["p_frac_energy_phys"]), default=None)
        if branch:
            _enrich_acoustic_energies(row, branch)
            row["energy_source"] = str(case_dir / "results" / f"result_{tag}.json")
            return
    _enrich_acoustic_energies(
        row,
        {
            "frequency_hz": COUPLED_BASELINE_F_HZ,
            "p_frac_energy_phys": COUPLED_BASELINE_P_FRAC,
            "mode_class_physical_energy": "acoustic_dominated",
            "source": "manifest frozen_baseline",
        },
    )


def _enrich_hole_radius_large(samples: Dict[str, Dict[str, Any]]) -> None:
    row = samples.get("hole_radius_large")
    if not row:
        return
    case_dir = SENS_ROOT / "samples" / "hole_radius_large"
    for tag in (275000, 244390):
        result = _load_json(case_dir / "results" / f"result_{tag}.json")
        if not result:
            continue
        branch = result.get("acoustic_branch_by_energy") or result.get("nearest_acoustic_branch")
        if not branch and result.get("in_band_modes"):
            acoustic = [m for m in result["in_band_modes"] if is_acoustic_branch(m)]
            if acoustic:
                branch = max(acoustic, key=lambda m: float(m["p_frac_energy_phys"]))
        if branch and is_acoustic_branch(branch):
            _enrich_acoustic_energies(row, branch)
            row["delta_f_hz_from_coupled_baseline"] = (
                float(branch["frequency_hz"]) - COUPLED_BASELINE_F_HZ
            )
            row["energy_source"] = str(case_dir / "results" / f"result_{tag}.json")
            return


def _match_structural_pairs(
    branches_a: List[Dict[str, Any]],
    branches_b: List[Dict[str, Any]],
    *,
    match_tol_hz: float = STRUCT_MATCH_TOL_HZ,
) -> List[Dict[str, Any]]:
    """Greedy frequency matching (not raw mode index)."""
    pairs: List[Dict[str, Any]] = []
    used_b: set = set()
    for i, ba in enumerate(branches_a):
        fa = float(ba["frequency_hz"])
        best_j = None
        best_df = float("inf")
        for j, bb in enumerate(branches_b):
            if j in used_b:
                continue
            fb = float(bb["frequency_hz"])
            df = abs(fa - fb)
            if df <= match_tol_hz and df < best_df:
                best_df = df
                best_j = j
        if best_j is None:
            continue
        used_b.add(best_j)
        bb = branches_b[best_j]
        fb = float(bb["frequency_hz"])
        pairs.append(
            {
                "f_a_hz": fa,
                "f_b_hz": fb,
                "delta_f_hz": fb - fa,
                "f_b_gt_f_a": fb > fa,
                "p_frac_a": float(ba.get("p_frac_energy_phys", float("nan"))),
                "p_frac_b": float(bb.get("p_frac_energy_phys", float("nan"))),
                "E_struct_a": ba.get("structural_modal_energy_phys"),
                "E_struct_b": bb.get("structural_modal_energy_phys"),
            }
        )
    return pairs


def _structural_trend_report(
    sample_a: str,
    sample_b: str,
    branches_a: List[Dict[str, Any]],
    branches_b: List[Dict[str, Any]],
    *,
    expect_b_higher: bool,
    parameter: str,
) -> Dict[str, Any]:
    pairs = _match_structural_pairs(branches_a, branches_b)
    med_a = float("nan")
    med_b = float("nan")
    if branches_a:
        med_a = float(sorted(float(b["frequency_hz"]) for b in branches_a)[len(branches_a) // 2])
    if branches_b:
        med_b = float(sorted(float(b["frequency_hz"]) for b in branches_b)[len(branches_b) // 2])
    n_higher = sum(1 for p in pairs if p["f_b_gt_f_a"])
    n_pairs = len(pairs)
    trend_by_pairs = (
        n_pairs > 0 and n_higher >= max(1, int(math.ceil(0.5 * n_pairs)))
    ) if expect_b_higher else (
        n_pairs > 0 and n_higher < max(1, int(math.ceil(0.5 * n_pairs)))
    )
    trend_by_median = (
        math.isfinite(med_a)
        and math.isfinite(med_b)
        and ((med_b > med_a) if expect_b_higher else (med_b < med_a))
    )
    passed = trend_by_pairs or trend_by_median
    return {
        "parameter": parameter,
        "sample_a": sample_a,
        "sample_b": sample_b,
        "expect": "f_b > f_a" if expect_b_higher else "f_b < f_a",
        "structural_branches_a": branches_a,
        "structural_branches_b": branches_b,
        "matched_pairs": pairs,
        "n_matched_pairs": n_pairs,
        "n_pairs_with_f_b_gt_f_a": n_higher,
        "median_f_a_hz": med_a,
        "median_f_b_hz": med_b,
        "trend_by_matched_pairs": trend_by_pairs,
        "trend_by_median_frequency": trend_by_median,
        "structural_trend_pass": passed,
    }


def _acoustic_stable(sample: Dict[str, Any]) -> bool:
    f_hz = float(sample.get("nearest_acoustic_f_hz", float("nan")))
    p_frac = float(sample.get("p_frac_energy_phys", 0.0))
    if not math.isfinite(f_hz):
        branch = sample.get("nearest_acoustic_branch") or {}
        f_hz = float(branch.get("frequency_hz", float("nan")))
        p_frac = float(branch.get("p_frac_energy_phys", p_frac))
    return (
        math.isfinite(f_hz)
        and abs(f_hz - COUPLED_BASELINE_F_HZ) <= ACOUSTIC_STABLE_DELTA_HZ
        and p_frac >= ENERGY_ACOUSTIC_THRESHOLD
    )


def _validation_flags(samples: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    f_rs = float(samples.get("hole_radius_small", {}).get("nearest_acoustic_f_hz", float("nan")))
    f_rl = float(samples.get("hole_radius_large", {}).get("nearest_acoustic_f_hz", float("nan")))
    f_ds = float(samples.get("depth_small", {}).get("nearest_acoustic_f_hz", float("nan")))
    f_dl = float(samples.get("depth_large", {}).get("nearest_acoustic_f_hz", float("nan")))

    radius_pass = (
        math.isfinite(f_rs)
        and math.isfinite(f_rl)
        and f_rs < COUPLED_BASELINE_F_HZ < f_rl
    )
    depth_pass = (
        math.isfinite(f_ds)
        and math.isfinite(f_dl)
        and f_ds > f_dl
    )

    thick_small = _reload_sample_structural("top_thickness_small", samples.get("top_thickness_small", {}))
    thick_large = _reload_sample_structural("top_thickness_large", samples.get("top_thickness_large", {}))
    stiff_soft = _reload_sample_structural("top_stiffness_soft", samples.get("top_stiffness_soft", {}))
    stiff_stiff = _reload_sample_structural("top_stiffness_stiff", samples.get("top_stiffness_stiff", {}))

    thick_struct = _structural_trend_report(
        "top_thickness_small",
        "top_thickness_large",
        thick_small,
        thick_large,
        expect_b_higher=True,
        parameter="top_thickness",
    )
    stiff_struct = _structural_trend_report(
        "top_stiffness_soft",
        "top_stiffness_stiff",
        stiff_soft,
        stiff_stiff,
        expect_b_higher=True,
        parameter="top_plate_stiffness",
    )

    return {
        "radius_acoustic_trend_pass": radius_pass,
        "depth_acoustic_trend_pass": depth_pass,
        "thickness_acoustic_branch_stable": _acoustic_stable(samples.get("top_thickness_small", {}))
        and _acoustic_stable(samples.get("top_thickness_large", {})),
        "thickness_structural_trend_pass": thick_struct["structural_trend_pass"],
        "stiffness_acoustic_branch_stable": _acoustic_stable(samples.get("top_stiffness_soft", {}))
        and _acoustic_stable(samples.get("top_stiffness_stiff", {})),
        "stiffness_structural_trend_pass": stiff_struct["structural_trend_pass"],
        "structural_analysis": {
            "top_thickness": thick_struct,
            "top_stiffness": stiff_struct,
        },
    }


def _write_report_md(
    samples: Dict[str, Dict[str, Any]],
    flags: Dict[str, Any],
    manifest: Dict[str, Any],
    promotion: Dict[str, Any],
) -> None:
    sa = flags["structural_analysis"]["top_thickness"]
    sb = flags["structural_analysis"]["top_stiffness"]
    lines = [
        "# v2 sensitivity validation — completed report",
        "",
        "## Coupled baseline",
        f"- f_acoustic = **{COUPLED_BASELINE_F_HZ:.6f}** Hz",
        f"- p_frac_energy_phys = **{COUPLED_BASELINE_P_FRAC:.4f}**",
        "",
        "## Validation flags",
        "",
        f"| flag | pass |",
        f"|------|:----:|",
    ]
    for key in (
        "radius_acoustic_trend_pass",
        "depth_acoustic_trend_pass",
        "thickness_acoustic_branch_stable",
        "thickness_structural_trend_pass",
        "stiffness_acoustic_branch_stable",
        "stiffness_structural_trend_pass",
    ):
        lines.append(f"| `{key}` | `{flags[key]}` |")
    lines.extend(
        [
            "",
            "## Acoustic summary",
            "",
            "| sample | f Hz | Δf coupled | p_frac | E_air | E_struct | class |",
            "|--------|-----:|-----------:|-------:|------:|--------:|:------|",
        ]
    )
    order = [
        "baseline_coupled_v2",
        "hole_radius_small",
        "hole_radius_large",
        "depth_small",
        "depth_large",
        "top_thickness_small",
        "top_thickness_large",
        "top_stiffness_soft",
        "top_stiffness_stiff",
    ]
    for sid in order:
        row = samples.get(sid)
        if not row:
            continue
        f_a = float(row.get("nearest_acoustic_f_hz", float("nan")))
        d_f = 0.0 if sid == "baseline_coupled_v2" else f_a - COUPLED_BASELINE_F_HZ
        p_e = float(row.get("p_frac_energy_phys", float("nan")))
        e_a = row.get("acoustic_modal_energy_phys")
        e_s = row.get("structural_modal_energy_phys")
        e_a_s = f"{float(e_a):.3e}" if e_a is not None and math.isfinite(float(e_a)) else "—"
        e_s_s = f"{float(e_s):.3e}" if e_s is not None and math.isfinite(float(e_s)) else "—"
        cls = row.get("mode_class_physical_energy", "—")
        lines.append(
            f"| {sid} | {f_a:.3f} | {d_f:+.3f} | {p_e:.4f} | {e_a_s} | {e_s_s} | {cls} |"
        )
    lines.extend(
        [
            "",
            "## Structural branches in band",
            "",
            "### top_thickness_small",
            "",
            _format_structural_table(sa["structural_branches_a"]),
            "",
            "### top_thickness_large",
            "",
            _format_structural_table(sa["structural_branches_b"]),
            "",
            f"**Thickness trend:** {sa['structural_trend_pass']} "
            f"(matched pairs {sa['n_pairs_with_f_b_gt_f_a']}/{sa['n_matched_pairs']}, "
            f"median {sa['median_f_a_hz']:.2f} → {sa['median_f_b_hz']:.2f} Hz)",
            "",
            "### top_stiffness_soft",
            "",
            _format_structural_table(sb["structural_branches_a"]),
            "",
            "### top_stiffness_stiff",
            "",
            _format_structural_table(sb["structural_branches_b"]),
            "",
            f"**Stiffness trend:** {sb['structural_trend_pass']} "
            f"(matched pairs {sb['n_pairs_with_f_b_gt_f_a']}/{sb['n_matched_pairs']}, "
            f"median {sb['median_f_a_hz']:.2f} → {sb['median_f_b_hz']:.2f} Hz)",
            "",
            "## Promotion (staged)",
            "",
            f"- `acoustic_geometric_validation_pass` = `{promotion.get('acoustic_geometric_validation_pass')}`",
            f"- `material_species_validation_pass` = `{promotion.get('material_species_validation_pass')}`",
            f"- `production_parameter_coverage_pass` = `{promotion.get('production_parameter_coverage_pass')}`",
            f"- `mesh_convergence_pass` = `{promotion.get('mesh_convergence_pass')}`",
            f"- `lhs_promotion_blocked` = `{promotion.get('lhs_promotion_blocked')}`",
            "",
            "Exploratory only (not production material gate): `top_stiffness_soft`, `top_stiffness_stiff`.",
        ]
    )
    (DIAG_DIR / "v2_sensitivity_validation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _format_structural_table(branches: List[Dict[str, Any]]) -> str:
    if not branches:
        return "_No structural branches recorded in saved artifacts._"
    lines = [
        "| f Hz | p_frac | E_struct | E_air | class |",
        "|-----:|-------:|---------:|------:|:------|",
    ]
    for b in branches:
        lines.append(
            f"| {float(b['frequency_hz']):.3f} | "
            f"{float(b.get('p_frac_energy_phys', float('nan'))):.4f} | "
            f"{float(b.get('structural_modal_energy_phys', 0)):.3e} | "
            f"{float(b.get('acoustic_modal_energy_phys', 0)):.3e} | "
            f"{b.get('mode_class_physical_energy', '—')} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 sensitivity report post-process")
    args = parser.parse_args()

    if not SUMMARY_JSON.is_file():
        print(f"[report_post] missing {SUMMARY_JSON}", file=sys.stderr)
        print(
            "Run controlled suite on VM first, or copy "
            "v2_sensitivity_validation/diagnostics/physical_core_v2_validation_summary.json",
            file=sys.stderr,
        )
        return 1

    manifest = load_manifest()
    summary = _load_json(SUMMARY_JSON) or {}
    samples: Dict[str, Dict[str, Any]] = {
        k: dict(v) for k, v in (summary.get("samples") or {}).items()
    }

    _enrich_baseline_coupled(samples)
    _enrich_hole_radius_large(samples)

    for sid in (
        "top_thickness_small",
        "top_thickness_large",
        "top_stiffness_soft",
        "top_stiffness_stiff",
    ):
        if sid in samples:
            samples[sid]["structural_branches_in_band"] = _reload_sample_structural(
                sid, samples[sid]
            )

    flags = _validation_flags(samples)
    all_struct_present = all(
        len(flags["structural_analysis"][k]["structural_branches_a"]) > 0
        or len(flags["structural_analysis"][k]["structural_branches_b"]) > 0
        for k in ("top_thickness", "top_stiffness")
    )

    acoustic_geometric_pass = all(
        flags[k]
        for k in (
            "radius_acoustic_trend_pass",
            "depth_acoustic_trend_pass",
            "thickness_acoustic_branch_stable",
        )
    )
    production_manifest = (
        load_production_manifest() if PRODUCTION_MANIFEST_PATH.is_file() else {}
    )
    phase2_ids = list(production_manifest.get("phase2_sample_ids") or [])
    phase2_results: Dict[str, Dict[str, Any]] = {}
    if PRODUCTION_SUMMARY_JSON.is_file():
        prod = _load_json(PRODUCTION_SUMMARY_JSON) or {}
        phase2_results = {
            sid: prod.get("samples", {}).get(sid, {})
            for sid in phase2_ids
            if sid in (prod.get("samples") or {})
        }
    material_pass = (
        bool(phase2_ids)
        and all((phase2_results.get(sid) or {}).get("status") == "ok" for sid in phase2_ids)
    )
    production_coverage_pass = material_pass and all(
        (phase2_results.get(sid) or {}).get("status") == "ok"
        for sid in phase2_ids
        if sid.startswith(("length_", "width_"))
    )

    promotion = dict(summary.get("promotion") or {})
    promotion["lhs_promotion_blocked_until_suite_pass"] = True
    promotion["lhs_blocked"] = True
    promotion["lhs_promotion_blocked"] = True
    promotion["mesh_convergence_blocked"] = True
    promotion.update(flags)
    promotion["acoustic_geometric_validation_pass"] = acoustic_geometric_pass
    promotion["material_species_validation_pass"] = "PASS" if material_pass else "Pending"
    promotion["production_parameter_coverage_pass"] = (
        "PASS" if production_coverage_pass else "Pending"
    )
    promotion["mesh_convergence_pass"] = "Pending"
    promotion["exploratory_not_production_gate"] = list(
        production_manifest.get("exploratory_not_production_gate")
        or ["top_stiffness_soft", "top_stiffness_stiff"]
    )
    promotion["full_nonrandom_suite_pass"] = acoustic_geometric_pass
    promotion["note_stiffness_samples"] = (
        "top_stiffness_soft/stiff are exploratory E_L scaling only; "
        "not the production wood-material validation gate."
    )
    write_validation_status(samples, phase2_results, production_manifest=production_manifest)

    out = {
        **summary,
        "report_post_processed": True,
        "coupled_baseline_acoustic_f_hz": COUPLED_BASELINE_F_HZ,
        "coupled_baseline_p_frac_energy_phys": COUPLED_BASELINE_P_FRAC,
        "validation_flags": flags,
        "samples": samples,
        "promotion": promotion,
        "note": (
            "Post-process only; coupled_physical_core_v2 unchanged. "
            "Radius pilot = first parametric validation."
        ),
    }
    write_json(SUMMARY_JSON, out)
    write_json(DIAG_DIR / "v2_sensitivity_validation_report.json", out)
    _write_report_md(samples, flags, manifest, promotion)

    print(f"[report_post] wrote {SUMMARY_JSON}")
    print(f"[report_post] wrote {DIAG_DIR / 'v2_sensitivity_validation_report.md'}")
    if not all_struct_present:
        print(
            "[report_post] warn: some structural_branches_in_band empty in artifacts; "
            "re-run controlled suite if structural trend flags are unreliable",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
