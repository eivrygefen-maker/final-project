#!/usr/bin/env python3
"""Compare baseline vs active-domain validation runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
COMPARISON_DIR = EXPERIMENT_ROOT / "comparison"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_modes(variant_dir: Path, candidates: List[Dict[str, Any]]) -> List[Tuple[float, np.ndarray, str]]:
    from scipy import sparse

    from fem_mode_array_utils import csr_u_slice, load_mode_column_any

    modes: List[Tuple[float, np.ndarray, str]] = []
    for c in candidates:
        rel = c.get("vector_path")
        if not rel:
            continue
        path = variant_dir / str(rel)
        if not path.is_file():
            path = variant_dir / "modes" / Path(str(rel)).name
        if not path.is_file():
            continue
        mat = load_mode_column_any(path)
        n_u = int(c.get("n_u_collapsed", mat.shape[0]))
        u = csr_u_slice(mat, n_u).toarray().ravel().astype(np.float64)
        modes.append((float(c["hz"]), u, str(path.name)))
    modes.sort(key=lambda t: t[0])
    return modes


def _mac(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        return float("nan")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(abs(np.dot(a, b)) / (na * nb))


def _match_modes(
    base_modes: List[Tuple[float, np.ndarray, str]],
    cand_modes: List[Tuple[float, np.ndarray, str]],
    *,
    freq_tol_hz: float = 2.0,
) -> List[Dict[str, Any]]:
    used_c = set()
    rows: List[Dict[str, Any]] = []
    for i, (fb, ub, nb) in enumerate(base_modes):
        best_j = None
        best_df = float("inf")
        for j, (fc, uc, nc) in enumerate(cand_modes):
            if j in used_c:
                continue
            df = abs(fc - fb)
            if df < best_df:
                best_df = df
                best_j = j
        if best_j is None or best_df > freq_tol_hz:
            rows.append(
                {
                    "baseline_index": i,
                    "baseline_hz": fb,
                    "candidate_hz": None,
                    "delta_hz": None,
                    "rel_freq_error": None,
                    "mac_u": None,
                    "matched": False,
                }
            )
            continue
        fc, uc, nc = cand_modes[best_j]
        used_c.add(best_j)
        rel_err = abs(fc - fb) / max(abs(fb), 1.0e-9)
        mac = _mac(ub, uc)
        rows.append(
            {
                "baseline_index": i,
                "baseline_hz": fb,
                "candidate_hz": fc,
                "delta_hz": fc - fb,
                "rel_freq_error": rel_err,
                "mac_u": mac,
                "matched": True,
                "baseline_mode": nb,
                "candidate_mode": nc,
            }
        )
    return rows


def _verdict(freq_rows: List[Dict[str, Any]], *, freq_tol: float = 0.005, mac_tol: float = 0.98) -> str:
    matched = [r for r in freq_rows if r.get("matched")]
    if not matched:
        return "INCONCLUSIVE"
    for r in matched:
        rel = r.get("rel_freq_error")
        mac = r.get("mac_u")
        if rel is None or rel > freq_tol:
            return "FAIL"
        if mac is not None and not math.isnan(mac) and mac < mac_tol:
            return "FAIL"
    return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-hz", type=float, default=202.0)
    args = parser.parse_args()
    hz_tag = int(round(float(args.target_hz) * 1000))

    mesh_audit = _load_json(EXPERIMENT_ROOT / "mesh" / "mesh_audit.json")
    base_result = _load_json(EXPERIMENT_ROOT / "baseline" / "results" / f"result_{hz_tag}.json")
    cand_result = _load_json(EXPERIMENT_ROOT / "active_domain" / "results" / f"result_{hz_tag}.json")

    base_op = base_result.get("operator_meta", {})
    cand_op = cand_result.get("operator_meta", {})
    base_ad = base_op.get("active_domain") or {}
    cand_ad = cand_op.get("active_domain") or {}

    dof_rows = [
        {
            "metric": "n_mixed_global",
            "baseline": base_op.get("n_mixed_global"),
            "candidate": cand_op.get("n_mixed_global"),
        },
        {
            "metric": "n_u_collapsed",
            "baseline": base_op.get("n_u_collapsed"),
            "candidate": cand_op.get("n_u_collapsed"),
        },
        {
            "metric": "n_p_collapsed",
            "baseline": base_op.get("n_p_collapsed"),
            "candidate": cand_op.get("n_p_collapsed"),
        },
        {
            "metric": "n_active_mixed",
            "baseline": base_ad.get("n_active", base_op.get("n_mixed_global")),
            "candidate": cand_ad.get("n_active"),
        },
        {
            "metric": "n_u_active",
            "baseline": base_ad.get("n_u_active", base_op.get("n_u_collapsed")),
            "candidate": cand_ad.get("n_u_active"),
        },
        {
            "metric": "n_p_active",
            "baseline": base_ad.get("n_p_active", base_op.get("n_p_collapsed")),
            "candidate": cand_ad.get("n_p_active"),
        },
        {
            "metric": "elapsed_s",
            "baseline": base_op.get("elapsed_s"),
            "candidate": cand_op.get("elapsed_s"),
        },
    ]
    n_base = float(dof_rows[3]["baseline"] or 0)
    n_cand = float(dof_rows[3]["candidate"] or 1)
    pct_red = 100.0 * (1.0 - n_cand / max(n_base, 1.0))

    base_modes = _load_modes(EXPERIMENT_ROOT / "baseline", base_result.get("candidates", []))
    cand_modes = _load_modes(EXPERIMENT_ROOT / "active_domain", cand_result.get("candidates", []))
    freq_rows = _match_modes(base_modes, cand_modes)
    verdict = _verdict(freq_rows)

    def _time_stats(variant: str) -> Dict[str, Any]:
        p = EXPERIMENT_ROOT / variant / "timing" / "time_stats.json"
        return _load_json(p) if p.is_file() else {}

    summary = {
        "mesh_sha256": mesh_audit.get("sha256"),
        "baseline_mesh_sha256_result": base_result.get("mesh_sha256"),
        "candidate_mesh_sha256_result": cand_result.get("mesh_sha256"),
        "baseline_time_stats": _time_stats("baseline"),
        "candidate_time_stats": _time_stats("active_domain"),
        "mesh_n_nodes": mesh_audit.get("n_nodes"),
        "target_hz": args.target_hz,
        "soundhole_bc_baseline": base_result.get("soundhole_bc"),
        "soundhole_bc_candidate": cand_result.get("soundhole_bc"),
        "pressure_gauge_baseline": base_result.get("pressure_gauge"),
        "pressure_gauge_candidate": cand_result.get("pressure_gauge"),
        "candidate_method": cand_ad.get("method", "unknown"),
        "percent_reduction_n_active": pct_red,
        "verdict": verdict,
        "acceptance": {
            "rel_freq_error_max": 0.005,
            "mac_u_min": 0.98,
        },
        "frequency_pairs": freq_rows,
        "dof_comparison": dof_rows,
    }

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARISON_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with open(COMPARISON_DIR / "frequency_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "baseline_index",
                "baseline_hz",
                "candidate_hz",
                "delta_hz",
                "rel_freq_error",
                "mac_u",
                "matched",
            ],
        )
        w.writeheader()
        for row in freq_rows:
            w.writerow({k: row.get(k) for k in w.fieldnames})

    with open(COMPARISON_DIR / "dof_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "baseline", "candidate"])
        w.writeheader()
        for row in dof_rows:
            w.writerow(row)

    report = [
        "# Active-domain validation report",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "## Mesh",
        f"- SHA256: `{mesh_audit.get('sha256', 'n/a')}`",
        f"- Nodes: {mesh_audit.get('n_nodes')}",
        f"- Tetrahedra: {mesh_audit.get('n_tetrahedra')}",
        "",
        "## BC policy",
        f"- Baseline: soundhole_bc={base_result.get('soundhole_bc')!r}, "
        f"pressure_gauge={base_result.get('pressure_gauge')!r}",
        f"- Candidate: soundhole_bc={cand_result.get('soundhole_bc')!r}, "
        f"pressure_gauge={cand_result.get('pressure_gauge')!r}",
        "",
        "## DOF reduction",
        f"- Active mixed DOFs: baseline {dof_rows[3]['baseline']} → candidate {dof_rows[3]['candidate']} "
        f"({pct_red:.1f}% reduction)",
        f"- Active u: {dof_rows[4]['baseline']} → {dof_rows[4]['candidate']}",
        f"- Active p: {dof_rows[5]['baseline']} → {dof_rows[5]['candidate']}",
        "",
        "## Candidate method",
        f"- `{cand_ad.get('method', 'n/a')}` (algebraic restriction on parent mesh assembled operators)",
        "",
        "## Frequency pairs (matched within 2 Hz)",
    ]
    for row in freq_rows:
        if not row.get("matched"):
            report.append(f"- baseline {row['baseline_hz']:.4f} Hz: **no match**")
            continue
        report.append(
            f"- {row['baseline_hz']:.4f} Hz → {row['candidate_hz']:.4f} Hz "
            f"(rel err {100.0 * float(row['rel_freq_error']):.3f}%, MAC_u={row['mac_u']:.4f})"
        )
    report.extend(
        [
            "",
            "## Runtime (informational only)",
            f"- Baseline elapsed: {base_op.get('elapsed_s')} s",
            f"- Candidate elapsed: {cand_op.get('elapsed_s')} s",
            "",
            "## Decision",
        ]
    )
    if verdict == "PASS":
        report.append(
            "Active-domain formulation is a **valid candidate** for production testing "
            "(subject to production-mesh validation)."
        )
    elif verdict == "FAIL":
        report.append("Active-domain formulation **changes** the modeled result beyond tolerance.")
    else:
        report.append("More validation required (missing modes or MAC data).")

    (COMPARISON_DIR / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if verdict == "PASS" else (1 if verdict == "FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
