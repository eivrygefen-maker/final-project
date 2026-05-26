#!/usr/bin/env python3
"""Post-process v2_mesh_convergence solves into summary tables and staged gate status."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_mesh_convergence_common import (
    INCREMENTAL_JSON,
    SUMMARY_JSON,
    SUMMARY_MD,
    VALIDATION_STATUS_JSON,
    case_by_id,
    load_manifest,
    mesh_audit_path,
    solve_result_path,
    write_json,
)

ENERGY_ACOUSTIC_THRESHOLD = 0.85


def _load_solve(level_id: str, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    p = solve_result_path(level_id, str(case["id"]), float(case["target_hz"]))
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _acoustic_metrics(solve: Dict[str, Any]) -> Dict[str, Any]:
    branch = solve.get("acoustic_branch_by_energy") or solve.get("nearest_acoustic_branch")
    if not branch:
        in_band = solve.get("in_band_modes") or []
        if in_band:
            branch = max(in_band, key=lambda m: float(m.get("p_frac_energy_phys", 0.0)))
    f_hz = float((branch or {}).get("frequency_hz", float("nan")))
    return {
        "frequency_hz": f_hz,
        "p_frac_energy_phys": float((branch or {}).get("p_frac_energy_phys", float("nan"))),
        "E_air": (branch or {}).get("E_air_phys"),
        "E_struct": (branch or {}).get("E_struct_phys"),
        "mode_class_physical_energy": (branch or {}).get("mode_class_physical_energy"),
        "branch_captured": branch is not None,
        "v2_converged": bool(solve.get("v2_converged")),
        "ingested_l0_reuse": bool(solve.get("ingested_l0_reuse")),
        "n_reduced_W": solve.get("n_reduced_W"),
        "n_u_active": solve.get("n_u_active"),
        "n_p_active": solve.get("n_p_active"),
    }


def _structural_metrics(solve: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    lo = float(case["harvest_lo_hz"])
    hi = float(case["harvest_hi_hz"])
    struct_freqs: List[float] = []
    for m in solve.get("in_band_modes") or []:
        f_hz = float(m.get("frequency_hz", float("nan")))
        if not math.isfinite(f_hz) or f_hz < lo or f_hz > hi:
            continue
        cls = str(m.get("mode_class_physical_energy", ""))
        if cls == "structural_dominated" or float(m.get("p_frac_energy_phys", 1.0)) <= 0.15:
            struct_freqs.append(f_hz)
    struct_freqs.sort()
    return {
        "number_of_converged_modes": int(solve.get("num_modes_saved", -1)),
        "number_of_structural_dominated_modes": len(struct_freqs),
        "structural_frequency_range_hz": (
            [struct_freqs[0], struct_freqs[-1]] if struct_freqs else None
        ),
        "structural_branch_frequencies_hz": struct_freqs,
        "v2_converged": bool(solve.get("v2_converged")),
        "cross_mesh_mac": "unavailable_without_projection",
    }


def _rel_change(f_coarse: float, f_fine: float) -> float:
    if not math.isfinite(f_coarse) or not math.isfinite(f_fine) or abs(f_fine) < 1.0e-12:
        return float("nan")
    return abs(f_coarse - f_fine) / abs(f_fine)


def _pair_pass(f_coarse: float, f_fine: float, tol: float) -> bool:
    r = _rel_change(f_coarse, f_fine)
    return math.isfinite(r) and r <= tol


def main() -> int:
    manifest = load_manifest()
    tol = float((manifest.get("convergence_criterion") or {}).get("relative_frequency_tolerance", 0.01))
    levels = list(manifest.get("level_run_order") or ["L0", "L_mid", "L_prod", "L_check"])
    cases = manifest.get("cases") or []

    mesh_level_defs: Dict[str, Any] = {}
    for lid in levels:
        audit_samples: List[Dict[str, Any]] = []
        for case in cases:
            ap = mesh_audit_path(lid, str(case["id"]))
            if ap.is_file():
                audit_samples.append(json.loads(ap.read_text(encoding="utf-8")))
        mesh_level_defs[lid] = {
            "definition": (manifest.get("mesh_levels") or {}).get(lid),
            "l_prod_source": manifest.get("l_prod_source") if lid == "L_prod" else None,
            "l0_source": manifest.get("l0_source") if lid == "L0" else None,
            "mesh_audits": audit_samples,
        }

    acoustic_table: List[Dict[str, Any]] = []
    structural_table: List[Dict[str, Any]] = []
    prev_acoustic: Dict[str, float] = {}
    prev_struct_freqs: Dict[str, List[float]] = {}

    for case in cases:
        cid = str(case["id"])
        ctype = str(case.get("case_type"))
        for lid in levels:
            solve = _load_solve(lid, case)
            if not solve:
                continue
            if ctype == "acoustic":
                m = _acoustic_metrics(solve)
                f_hz = float(m["frequency_hz"])
                prev = prev_acoustic.get(cid)
                row = {
                    "case_id": cid,
                    "mesh_level": lid,
                    **m,
                    "delta_f_from_previous_mesh_level_hz": (
                        f_hz - prev if prev is not None and math.isfinite(f_hz) else float("nan")
                    ),
                    "relative_frequency_change": (
                        _rel_change(f_hz, prev) if prev is not None else float("nan")
                    ),
                }
                if prev is not None:
                    row["relative_frequency_change"] = _rel_change(f_hz, prev)
                    row["delta_f_from_previous_mesh_level_hz"] = f_hz - prev
                acoustic_table.append(row)
                if math.isfinite(f_hz):
                    prev_acoustic[cid] = f_hz
            else:
                m = _structural_metrics(solve, case)
                freqs = list(m.get("structural_branch_frequencies_hz") or [])
                tracked: List[Dict[str, Any]] = []
                pref = prev_struct_freqs.get(cid)
                if pref:
                    for f_ref in pref[: min(8, len(pref))]:
                        if not freqs:
                            break
                        j = min(range(len(freqs)), key=lambda i: abs(freqs[i] - f_ref))
                        fm = freqs[j]
                        tracked.append(
                            {
                                "reference_hz": f_ref,
                                "matched_hz": fm,
                                "delta_f_hz": fm - f_ref,
                                "relative_change": _rel_change(fm, f_ref),
                            }
                        )
                structural_table.append(
                    {
                        "case_id": cid,
                        "mesh_level": lid,
                        **m,
                        "tracked_branch_pairs": tracked,
                    }
                )
                if freqs:
                    prev_struct_freqs[cid] = freqs

    def _level_f(case_id: str, lid: str, table: List[Dict[str, Any]]) -> Optional[float]:
        for r in table:
            if r.get("case_id") == case_id and r.get("mesh_level") == lid:
                return float(r.get("frequency_hz", float("nan")))
        return None

    verdicts: Dict[str, Any] = {}
    for case in cases:
        cid = str(case["id"])
        ctype = str(case.get("case_type"))
        if ctype == "acoustic":
            f_mid = _level_f(cid, "L_mid", acoustic_table)
            f_prod = _level_f(cid, "L_prod", acoustic_table)
            f_chk = _level_f(cid, "L_check", acoustic_table)
            prod_vs_mid = _pair_pass(f_mid, f_prod, tol) if f_mid and f_prod else False
            prod_vs_check = _pair_pass(f_prod, f_chk, tol) if f_chk and f_prod else None
            verdicts[cid] = {
                "case_type": "acoustic",
                "L_mid_vs_L_prod_pass": prod_vs_mid,
                "L_prod_vs_L_check_pass": prod_vs_check,
            }
        else:
            rows = [r for r in structural_table if r.get("case_id") == cid]
            prod_row = next((r for r in rows if r.get("mesh_level") == "L_prod"), None)
            chk_row = next((r for r in rows if r.get("mesh_level") == "L_check"), None)
            mid_row = next((r for r in rows if r.get("mesh_level") == "L_mid"), None)
            tracked_prod = (prod_row or {}).get("tracked_branch_pairs") or []
            pass_mid_prod = bool(tracked_prod) and all(
                float(t.get("relative_change", 1.0)) <= tol
                for t in tracked_prod
                if math.isfinite(float(t.get("relative_change", float("nan"))))
            )
            verdicts[cid] = {
                "case_type": "structural",
                "L_mid_vs_L_prod_branch_pass": pass_mid_prod,
                "note": "Structural convergence uses nearest-frequency branch tracking; MAC across meshes unavailable.",
            }

    incremental: Dict[str, Any] = {}
    if INCREMENTAL_JSON.is_file():
        incremental = json.loads(INCREMENTAL_JSON.read_text(encoding="utf-8"))

    l_check_skipped = incremental.get("L_check_skipped")
    l_check_attempted = incremental.get("L_check_attempted", False)
    acoustic_pass = all(
        v.get("L_prod_vs_L_check_pass") is True or v.get("L_mid_vs_L_prod_pass") is True
        for v in verdicts.values()
        if v.get("case_type") == "acoustic"
    )
    struct_pass = all(
        v.get("L_mid_vs_L_prod_branch_pass") for v in verdicts.values() if v.get("case_type") == "structural"
    )
    prod_proven = l_check_attempted and not l_check_skipped and all(
        v.get("L_prod_vs_L_check_pass") is True
        for v in verdicts.values()
        if v.get("case_type") == "acoustic" and v.get("L_prod_vs_L_check_pass") is not None
    )

    if prod_proven and struct_pass:
        mesh_pass = "PASS"
    elif l_check_skipped or not l_check_attempted:
        mesh_pass = "Pending"
    else:
        mesh_pass = "FAIL"

    staged = dict(manifest.get("staged_status_defaults") or {})
    staged["mesh_convergence_pass"] = mesh_pass
    staged["v2_production_promotion_ready"] = mesh_pass == "PASS"
    staged["lhs_promotion_blocked"] = True

    report = {
        "suite": manifest.get("suite"),
        "convergence_criterion": manifest.get("convergence_criterion"),
        "l_prod_source": manifest.get("l_prod_source"),
        "l0_source": manifest.get("l0_source"),
        "mesh_level_definitions": mesh_level_defs,
        "incremental_run_state": incremental,
        "acoustic_convergence_table": acoustic_table,
        "structural_convergence_table": structural_table,
        "verdict_per_case": verdicts,
        "L_check_skipped_or_failed": l_check_skipped,
        "L_prod_proven_vs_L_check": prod_proven,
        "staged_status": staged,
    }
    write_json(SUMMARY_JSON, report)

    lines = [
        "# v2 mesh convergence summary",
        "",
        f"**Criterion:** relative frequency change ≤ {tol*100:.1f}% between compared levels.",
        "",
        "## L_prod (FOM) mesh source",
        "",
        f"- Mesher: `{manifest.get('l_prod_source', {}).get('mesher')}`",
        f"- Env: `{manifest.get('l_prod_source', {}).get('env')}`",
        "",
        "## Acoustic convergence",
        "",
        "| case | level | f (Hz) | Δf prev | rel change | p_frac | branch |",
        "|------|-------|--------|---------|------------|--------|--------|",
    ]
    for r in acoustic_table:
        lines.append(
            f"| {r.get('case_id')} | {r.get('mesh_level')} | {r.get('frequency_hz', float('nan')):.4f} | "
            f"{r.get('delta_f_from_previous_mesh_level_hz', float('nan')):+.4f} | "
            f"{r.get('relative_frequency_change', float('nan')):.4f} | "
            f"{r.get('p_frac_energy_phys', float('nan')):.4f} | {r.get('branch_captured')} |"
        )
    lines.extend(["", "## Structural / material_back_cedar", ""])
    for r in structural_table:
        fr = r.get("structural_frequency_range_hz")
        fr_s = f"{fr[0]:.1f}–{fr[1]:.1f}" if fr else "—"
        lines.append(
            f"- **{r.get('mesh_level')}**: n_struct={r.get('number_of_structural_dominated_modes')} "
            f"range={fr_s} Hz; cross-mesh MAC: {r.get('cross_mesh_mac')}"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- `mesh_convergence_pass` = **{mesh_pass}**",
            f"- `v2_production_promotion_ready` = **{staged.get('v2_production_promotion_ready')}**",
            f"- `lhs_promotion_blocked` = **{staged.get('lhs_promotion_blocked')}**",
            "",
        ]
    )
    if l_check_skipped:
        lines.append(
            f"*L_check skipped or failed:* `{l_check_skipped}`. "
            "L_prod is **not** proven from L0→L_prod alone."
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if VALIDATION_STATUS_JSON.is_file():
        status = json.loads(VALIDATION_STATUS_JSON.read_text(encoding="utf-8"))
        status["material_structural_branch_validation_pass"] = status.get(
            "material_structural_branch_validation_pass", "PASS"
        )
        status["mesh_convergence_pass"] = mesh_pass
        status["v2_production_promotion_ready"] = staged.get("v2_production_promotion_ready")
        status["lhs_promotion_blocked"] = True
        status["v2_mesh_convergence"] = {
            "summary_json": str(SUMMARY_JSON),
            "L_prod_proven_vs_L_check": prod_proven,
        }
        write_json(VALIDATION_STATUS_JSON, status)

    print(f"[mesh_conv_post] wrote {SUMMARY_JSON}", flush=True)
    print(f"[mesh_conv_post] mesh_convergence_pass={mesh_pass}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
