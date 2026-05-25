#!/usr/bin/env python3
"""Aggregate physics-integrity cases into comparison report and decision gate."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PHYSICS_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mode_diag(case_dir: Path) -> List[Dict[str, Any]]:
    p = case_dir / "diagnostics" / "mode_physics_diagnostics.json"
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("modes") or []


def _nearest_freq(freqs: List[float], target: float, tol: float = 8.0) -> Optional[float]:
    best = None
    best_d = tol
    for f in freqs:
        d = abs(f - target)
        if d < best_d:
            best_d = d
            best = f
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision",
        choices=(
            "auto",
            "PASS_REFERENCE_MODEL",
            "FAIL_SCALING_METRIC",
            "FAIL_FSI_FORMULATION",
            "INCONCLUSIVE",
        ),
        default="auto",
    )
    args = parser.parse_args()

    comp = PHYSICS_ROOT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)

    nominal_dir = PHYSICS_ROOT / "coupled_nominal"
    struct_dir = PHYSICS_ROOT / "structural_only"
    acoustic_dir = PHYSICS_ROOT / "acoustic_only"
    low_dir = PHYSICS_ROOT / "coupled_low_frequency"
    audit_path = comp / "coupling_audit.json"

    nominal_modes = _load_mode_diag(nominal_dir)
    struct_modes = _load_mode_diag(struct_dir)
    acoustic_modes = _load_mode_diag(acoustic_dir)
    low_modes = _load_mode_diag(low_dir)
    audit = _load_json(audit_path) or _load_json(nominal_dir / "diagnostics" / "physics_integrity_audit.json") or {}

    struct_result = _load_json(struct_dir / "results" / "result_structural.json") or {}
    acoustic_result = _load_json(acoustic_dir / "results" / "result_acoustic.json") or {}
    struct_freqs = [float(f) for f in struct_result.get("frequencies_hz") or []]
    acoustic_freqs = [float(f) for f in acoustic_result.get("frequencies_hz") or []]

    # Frequency shift table (coupled wood modes vs structural)
    shift_rows: List[Dict[str, Any]] = []
    for m in nominal_modes:
        if m.get("mode_class") == "acoustic_dominated":
            continue
        f_c = float(m.get("frequency_hz", 0.0))
        f_s = _nearest_freq(struct_freqs, f_c, tol=12.0)
        delta = (f_c - f_s) if f_s is not None else None
        rel = (delta / f_s * 100.0) if f_s and f_s > 0 and delta is not None else None
        shift_rows.append(
            {
                "coupled_hz": f_c,
                "structural_hz": f_s,
                "delta_hz": delta,
                "delta_percent": rel,
                "p_frac_raw": m.get("p_frac_raw"),
                "p_frac_phys_gnhep": m.get("p_frac_phys_gnhep"),
                "mode_class": m.get("mode_class"),
            }
        )

    shift_csv = comp / "frequency_shift_comparison.csv"
    with shift_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "coupled_hz",
                "structural_hz",
                "delta_hz",
                "delta_percent",
                "p_frac_raw",
                "p_frac_phys_gnhep",
                "mode_class",
            ],
        )
        w.writeheader()
        for row in shift_rows:
            w.writerow(row)

    energy_csv = comp / "mode_energy_comparison.csv"
    with energy_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "case",
                "mode_index",
                "frequency_hz",
                "p_frac_raw",
                "p_frac_phys_gnhep",
                "p_frac_production",
                "wood_participation",
                "mode_class",
                "p_block_max_phys_gnhep",
            ],
        )
        w.writeheader()
        for label, modes in (
            ("coupled_nominal", nominal_modes),
            ("coupled_low_frequency", low_modes),
            ("structural_only", struct_modes),
            ("acoustic_only", acoustic_modes),
        ):
            for m in modes:
                w.writerow({"case": label, **{k: m.get(k) for k in w.fieldnames if k != "case"}})

    # Acceptance metrics
    matvec = audit.get("assembly_matvec_diag") or {}
    p_unit_u = float(matvec.get("p_load_unit_u", 0.0) or 0.0)
    wood_exc_p = float(matvec.get("wood_excitation_p_norm", 0.0) or 0.0)
    soundhole_dofs = int(audit.get("soundhole_pressure_dof_count", 0))
    iface_facets = int(audit.get("fsi_iface_facet_count", 0))
    gnhep = audit.get("gnhep_scales") or {}
    s_up = float(gnhep.get("s_couple", 1.0))

    max_p_frac_phys = max(
        (float(m.get("p_frac_phys_gnhep", 0.0)) for m in nominal_modes + low_modes),
        default=0.0,
    )
    max_p_frac_raw = max(
        (float(m.get("p_frac_raw", 0.0)) for m in nominal_modes + low_modes),
        default=0.0,
    )
    coupled_or_acoustic = any(
        m.get("mode_class") in ("coupled", "acoustic_dominated")
        and float(m.get("p_frac_phys_gnhep", 0.0)) >= 0.02
        for m in nominal_modes + low_modes
    )
    meaningful_shift = any(
        r.get("delta_hz") is not None and abs(float(r["delta_hz"])) >= 0.5
        for r in shift_rows
        if r.get("structural_hz") is not None
    )
    scaling_ratio = max_p_frac_phys / max(max_p_frac_raw, 1.0e-30)
    operator_coupling_ok = p_unit_u > 1.0e-6 or wood_exc_p > 1.0e-6
    soundhole_ok = soundhole_dofs > 0
    iface_ok = iface_facets > 0

    decision = args.decision
    reasons: List[str] = []
    if decision == "auto":
        if not nominal_modes:
            decision = "INCONCLUSIVE"
            reasons.append("coupled_nominal mode diagnostics missing (run solve or ingest+analyze_modes).")
        elif scaling_ratio > 100.0 and max_p_frac_phys >= 0.01 and not coupled_or_acoustic:
            decision = "FAIL_SCALING_METRIC"
            reasons.append(
                f"GNHEP back-transform raises p_frac by ~{scaling_ratio:.1e}x but modes still look decoupled; "
                "check metric definition before blaming FSI."
            )
        elif scaling_ratio > 50.0 and max_p_frac_raw < 1.0e-5 and max_p_frac_phys >= 0.05:
            decision = "FAIL_SCALING_METRIC"
            reasons.append(
                "Production p_frac uses GNHEP-scaled eigenvectors; physical participation is much larger."
            )
        elif not operator_coupling_ok and not iface_ok:
            decision = "FAIL_FSI_FORMULATION"
            reasons.append("Assembled coupling matvec and interface facet count indicate broken FSI.")
        elif not soundhole_ok:
            decision = "INCONCLUSIVE"
            reasons.append("Soundhole pressure-release DOF count is zero — check mesh tags.")
        elif coupled_or_acoustic or meaningful_shift:
            if max_p_frac_raw < 1.0e-4 and max_p_frac_phys < 1.0e-3 and not meaningful_shift:
                decision = "FAIL_FSI_FORMULATION"
                reasons.append("No coupled modes and no frequency shift vs structural-only.")
            else:
                decision = "PASS_REFERENCE_MODEL"
                reasons.append(
                    "Operator coupling present; at least one mode or frequency shift shows structure–air interaction."
                )
        else:
            decision = "FAIL_FSI_FORMULATION"
            reasons.append(
                "Wood-band coupled modes show negligible physical pressure and no measurable shift vs structural-only."
            )

    coupling_audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reasons": reasons,
        "metrics": {
            "max_p_frac_raw_nominal": max_p_frac_raw,
            "max_p_frac_phys_gnhep_nominal": max_p_frac_phys,
            "scaling_ratio_phys_over_raw": scaling_ratio,
            "operator_matvec_p_load_unit_u": p_unit_u,
            "operator_matvec_wood_excitation_p": wood_exc_p,
            "soundhole_pressure_dof_count": soundhole_dofs,
            "fsi_iface_facet_count": iface_facets,
            "gnhep_s_couple": s_up,
            "first_acoustic_candidate_hz": acoustic_result.get("first_acoustic_candidate_hz"),
            "coupled_or_acoustic_mode_seen": coupled_or_acoustic,
            "meaningful_structural_frequency_shift": meaningful_shift,
        },
        "audit": audit,
    }
    (comp / "coupling_audit.json").write_text(
        json.dumps(coupling_audit, indent=2), encoding="utf-8"
    )

    lines = [
        "# Physics integrity report",
        "",
        f"Generated: {coupling_audit['generated_utc']}",
        "",
        "## Tests",
        "",
        "| Test | Status |",
        "|------|--------|",
        f"| TEST 1 coupled nominal @ 202 Hz | {'OK' if nominal_modes else 'MISSING'} |",
        f"| TEST 2 structural-only | {'OK' if struct_modes else 'MISSING'} |",
        f"| TEST 3 acoustic-only | {'OK' if acoustic_modes else 'MISSING (path exists in solver)' if acoustic_result else 'MISSING'} |",
        f"| TEST 4 coupled low-frequency | {'OK' if low_modes else 'MISSING'} |",
        "",
        "## Scaling / post-processing audit",
        "",
        f"- Max **p_frac_raw** (production-style, GNHEP-scaled eigenvector): `{max_p_frac_raw:.6e}`",
        f"- Max **p_frac_phys_gnhep** (undo s_uu/s_pp on blocks): `{max_p_frac_phys:.6e}`",
        f"- Ratio phys/raw: `{scaling_ratio:.6e}`",
        "",
        "Production harvest computes `p_frac` from the SLEPc vector **without** multiplying u/p blocks by "
        "`s_uu`/`s_pp`. That is not the same as cavity pressure in the original unscaled UFL assembly.",
        "",
        "## FSI operator audit",
        "",
        f"- FSI interface facets: `{iface_facets}`",
        f"- Soundhole pressure-release DOFs: `{soundhole_dofs}`",
        f"- Matvec ‖(A·e_u)_p‖ (wood excitation): `{wood_exc_p:.6e}`",
        f"- Matvec unit pressure load on u: `{p_unit_u:.6e}`",
        f"- GNHEP s_couple: `{s_up:.6e}`",
        "",
        "## Frequency shift (coupled vs structural, wood band)",
        "",
        "See `frequency_shift_comparison.csv`.",
        "",
        "## Mode energy / participation",
        "",
        "See `mode_energy_comparison.csv` and per-case `diagnostics/mode_physics_diagnostics.json`.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
    ]
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")

    (comp / "physics_integrity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] decision={decision} → {comp / 'physics_integrity_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
