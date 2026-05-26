#!/usr/bin/env python3
"""
Report-only replay recertification of saved EPS modes (no eigensolve).

Recomputes Rayleigh λ, f, and residual on the unshifted physical GNHEP for each
discovered case with saved mode vectors.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_eps_mapping_audit_lib import discover_replay_targets
from v2_mesh_convergence_common import CONV_DIAG, write_json
from v2_unreg_offset_report_evaluator import assemble_replay_operators, _load_sample_spec

OUT_JSON = CONV_DIAG / "v2_existing_pass_replay_recertification.json"
OUT_MD = CONV_DIAG / "v2_existing_pass_replay_recertification.md"

PHYSICS_ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
VALIDATION_MESH = (
    REPO_ROOT / "FEM/experiments/active_domain_validation/mesh/validation_tiny_guitar_3d.msh"
).resolve()


def _resolve_mesh_for_case(case_dir: Path) -> Optional[Path]:
    for rp in sorted((case_dir / "results").glob("result_*.json")) if (case_dir / "results").is_dir() else []:
        data = json.loads(rp.read_text(encoding="utf-8"))
        mf = data.get("mesh_file")
        if mf:
            p = Path(mf)
            if p.is_file():
                return p
    if VALIDATION_MESH.is_file():
        return VALIDATION_MESH
    return None


def _sample_from_case(case_dir: Path) -> Dict[str, Any]:
    spec = case_dir / "sample_spec.json"
    if spec.is_file():
        return json.loads(spec.read_text(encoding="utf-8"))
    return json.loads(V2_CONFIG.read_text(encoding="utf-8"))


def _recertify_case(case_dir: Path) -> Dict[str, Any]:
    from physical_fsi_seed_residual_audit import _rayleigh_metrics, _block_residual_contributions
    from fem_mode_array_utils import load_mode_column_any

    case_dir = case_dir.resolve()
    mesh = _resolve_mesh_for_case(case_dir)
    out: Dict[str, Any] = {
        "case_dir": str(case_dir),
        "mesh_file": str(mesh) if mesh else None,
        "status": "skipped",
        "modes": [],
    }
    summary_path = case_dir / "diagnostics" / "mode_energy_summary.json"
    if not summary_path.is_file() or mesh is None:
        out["status"] = "artifacts_insufficient"
        return out

    sample, _ = _load_sample_spec(case_dir, _sample_from_case(case_dir))
    sort_tag = case_dir.name.replace("/", "_")[:40]
    A, M, u_to_W, p_to_W, asm = assemble_replay_operators(
        mesh, sample, out_dir=case_dir / f"sorting_replay_recert_{sort_tag}"
    )
    modes_meta = json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []
    rows: List[Dict[str, Any]] = []
    try:
        for m in modes_meta:
            rel = str(m.get("vector_path", ""))
            vpath = (case_dir / rel).resolve()
            row: Dict[str, Any] = {
                "mode_index": m.get("mode_index"),
                "vector_file": str(vpath),
                "reported_frequency_hz": float(m.get("frequency_hz", float("nan"))),
            }
            if not vpath.is_file():
                row["replay_status"] = "vector_missing"
                rows.append(row)
                continue
            try:
                vec = np.asarray(load_mode_column_any(vpath).toarray(), dtype=np.float64).ravel()
                ray = _rayleigh_metrics(A, M, vec, seed_f_hz=row["reported_frequency_hz"])
                lam = float(ray["rayleigh_lambda"])
                replay_f = float(ray["rayleigh_f_hz"])
                res = _block_residual_contributions(
                    A, M, vec, lam0=lam, u_idx=u_to_W, p_idx=p_to_W
                )
                rel_res = float(res["relative_residual"])
                df = abs(replay_f - row["reported_frequency_hz"])
                d_frac = df / replay_f if replay_f > 0 else float("inf")
                row.update(
                    {
                        "replay_rayleigh_lambda": lam,
                        "replay_rayleigh_frequency_hz": replay_f,
                        "replay_relative_residual": rel_res,
                        "frequency_label_delta_hz": df,
                        "frequency_label_delta_fraction": d_frac,
                        "replay_status": "ok",
                        "label_vs_replay_consistent_1pct": bool(
                            math.isfinite(d_frac) and d_frac <= 0.01
                        ),
                        "physical_mode_replay_supported": bool(
                            math.isfinite(lam)
                            and lam > 0
                            and math.isfinite(replay_f)
                            and rel_res <= 0.05
                        ),
                    }
                )
            except Exception as exc:
                row["replay_status"] = f"exception:{type(exc).__name__}:{exc}"
            rows.append(row)
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    ok_rows = [r for r in rows if r.get("replay_status") == "ok"]
    consistent = [r for r in ok_rows if r.get("label_vs_replay_consistent_1pct")]
    physical = [r for r in ok_rows if r.get("physical_mode_replay_supported")]
    out["modes"] = rows
    out["status"] = "completed" if ok_rows else "replay_failed"
    out["summary"] = {
        "num_modes": len(rows),
        "num_replay_ok": len(ok_rows),
        "num_label_consistent_within_1pct": len(consistent),
        "num_physical_replay_supported": len(physical),
    }
    if ok_rows and len(consistent) == len(ok_rows):
        out["recertification_conclusion"] = "prior_reported_frequencies_are_replay_consistent_labels"
    elif physical and len(consistent) < len(ok_rows):
        out["recertification_conclusion"] = "some_modes_valid_physics_wrong_frequency_labels_only"
    elif ok_rows and not physical:
        out["recertification_conclusion"] = "replay_inconsistent_physical_modes"
    else:
        out["recertification_conclusion"] = "insufficient_or_failed_replay"
    return out


def main() -> int:
    targets = discover_replay_targets(REPO_ROOT)
    cases = [_recertify_case(Path(t["case_dir"])) for t in targets if t.get("has_replay_inputs")]

    supported = sum(
        1
        for c in cases
        if c.get("recertification_conclusion")
        in (
            "prior_reported_frequencies_are_replay_consistent_labels",
            "some_modes_valid_physics_wrong_frequency_labels_only",
        )
    )
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "VM_runtime_report_only_replay_recertification",
        "mapping_fix_applied_in_code_not_in_rerun": True,
        "cases_recertified": cases,
        "answers": {
            "which_prior_PASS_conclusions_remain_supported": (
                "Cases with recertification_conclusion="
                "prior_reported_frequencies_are_replay_consistent_labels or "
                "some_modes_valid_physics_wrong_frequency_labels_only"
            ),
            "which_were_wrong_labels_only": (
                "Rows with physical_mode_replay_supported and not label_vs_replay_consistent_1pct"
            ),
            "which_require_later_rerun": (
                "Cases with replay_failed or replay_inconsistent_physical_modes; "
                "or exposure inventory status potentially_exposed_artifacts_insufficient"
            ),
        },
        "summary": {
            "num_cases_attempted": len(cases),
            "num_cases_supported_or_label_only": supported,
        },
        "prior_PASS_auto_invalidated": False,
    }
    write_json(OUT_JSON, report)
    lines = [
        "# Existing PASS replay recertification",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        f"Cases attempted: {report['summary']['num_cases_attempted']}",
        "",
    ]
    for c in cases[:20]:
        lines.append(
            f"- `{Path(c['case_dir']).name}`: {c.get('recertification_conclusion')} "
            f"({(c.get('summary') or {}).get('num_replay_ok', 0)} modes)"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[replay_recert] cases={len(cases)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
