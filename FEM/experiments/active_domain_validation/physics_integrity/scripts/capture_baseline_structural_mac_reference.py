#!/usr/bin/env python3
"""
Post-process baseline structural modes/maps for same-mesh material MAC (no material re-solve).

Replays reduced operators and saved mode vectors under coupled_physical_core_v2/
physical_coupling_enabled. Does not modify v2 formulation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_sensitivity_common import (
    BASELINE_STRUCTURAL_MAC_REF_JSON,
    COUPLED_BASELINE_F_HZ,
    N_REDUCED_W_VALIDATION,
    V2_CONFIG,
    V2_ROOT,
    is_acoustic_branch,
    write_json,
)

BAND_LO = 220.0
BAND_HI = 300.0
TARGET_HZ = 244.39


def _classify_structural(mode: Dict[str, Any]) -> bool:
    if str(mode.get("mode_class_physical_energy")) == "structural_dominated":
        return True
    return float(mode.get("p_frac_energy_phys", 1.0)) <= 0.15


def capture_from_replay() -> Dict[str, Any]:
    from physical_core_v2_post import (
        _assemble_reduced_v2_operator,
        _replay_subcase_energy,
    )

    cfg_base = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    subcase = "physical_coupling_enabled"
    replay = _replay_subcase_energy(
        cfg_base,
        V2_CONFIG,
        subcase=subcase,
        coupling_enabled=True,
        target_hz=TARGET_HZ,
    )
    _A, _M, cfg, u_to_W, p_to_W, restr = _assemble_reduced_v2_operator(
        cfg_base,
        V2_CONFIG,
        subcase=subcase,
        coupling_enabled=True,
        apply_gnhep_normalize=True,
    )
    try:
        _A.destroy()
        _M.destroy()
    except Exception:
        pass

    in_band = list(replay.get("in_band_modes_physical_energy") or [])
    structural = [m for m in in_band if _classify_structural(m) and not is_acoustic_branch(m)]
    structural.sort(
        key=lambda m: float(m.get("structural_modal_energy_phys", 0.0)),
        reverse=True,
    )
    n_W = int(restr.get("n_reduced_W", replay.get("n_reduced_W", N_REDUCED_W_VALIDATION)))
    u_list = np.asarray(u_to_W, dtype=np.int32).ravel().tolist()
    p_list = np.asarray(p_to_W, dtype=np.int32).ravel().tolist()

    out_modes: List[Dict[str, Any]] = []
    for m in structural[:12]:
        out_modes.append(
            {
                "frequency_hz": float(m["frequency_hz"]),
                "mode_index": int(m.get("mode_index", -1)),
                "vector_path": m.get("vector_path"),
                "vector_absolute_path": m.get("vector_absolute_path"),
                "p_frac_energy_phys": float(m.get("p_frac_energy_phys", float("nan"))),
                "structural_modal_energy_phys": float(
                    m.get("structural_modal_energy_phys", float("nan"))
                ),
                "acoustic_modal_energy_phys": float(
                    m.get("acoustic_modal_energy_phys", float("nan"))
                ),
                "mode_class_physical_energy": m.get("mode_class_physical_energy"),
            }
        )

    return {
        "ready": bool(out_modes) and bool(u_list),
        "capture_method": "physical_core_v2_post_replay_no_eigensolve",
        "subcase": subcase,
        "target_hz": TARGET_HZ,
        "coupled_baseline_f_hz": COUPLED_BASELINE_F_HZ,
        "harvest_band_hz": [BAND_LO, BAND_HI],
        "n_reduced_W": n_W,
        "u_to_W": u_list,
        "p_to_W": p_list,
        "n_u_active": int(restr.get("n_u_active", len(u_list))),
        "n_p_active": int(restr.get("n_p_active", len(p_list))),
        "structural_reference_modes": out_modes,
        "n_structural_reference_modes": len(out_modes),
        "replay_n_in_band": len(in_band),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture baseline structural MAC reference")
    parser.add_argument("--force", action="store_true", help="Overwrite existing reference file")
    args = parser.parse_args()

    if BASELINE_STRUCTURAL_MAC_REF_JSON.is_file() and not args.force:
        existing = json.loads(BASELINE_STRUCTURAL_MAC_REF_JSON.read_text(encoding="utf-8"))
        if existing.get("ready"):
            print(f"[baseline_mac] reuse {BASELINE_STRUCTURAL_MAC_REF_JSON}", flush=True)
            return 0

    case_modes = V2_ROOT / "physical_coupling_enabled" / "modes"
    if not list(case_modes.glob("mode_*.smx.npz")):
        print(
            f"[baseline_mac] no mode vectors under {case_modes}; "
            "run coupled_physical_core_v2 validation on VM first.",
            file=sys.stderr,
        )
        return 2

    payload = capture_from_replay()
    if not payload.get("ready"):
        payload["ready"] = False
        payload["error"] = (
            "replay found no structural-dominated in-band modes; "
            "baseline reference capture incomplete"
        )
        write_json(BASELINE_STRUCTURAL_MAC_REF_JSON, payload)
        print(f"[baseline_mac] FAILED: {payload.get('error')}", file=sys.stderr)
        return 1

    write_json(BASELINE_STRUCTURAL_MAC_REF_JSON, payload)
    print(
        f"[baseline_mac] wrote {BASELINE_STRUCTURAL_MAC_REF_JSON} "
        f"n_structural={payload['n_structural_reference_modes']} n_reduced_W={payload['n_reduced_W']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
