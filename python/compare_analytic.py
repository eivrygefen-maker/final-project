#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------
# Partner-consistent formulas
# -----------------------------
def compute_D_ij(E_L: float, E_T: float, G_LT: float, nu_LT: float, h: float) -> Tuple[float, float, float, float]:
    """
    Same engineering-constants -> D_ij as in your FEM solver (and consistent with partner spec):
      nu_TL = nu_LT * (E_T/E_L)
      denom = 1 - nu_LT*nu_TL
      D11 = E_L*h^3/(12*denom)
      D22 = E_T*h^3/(12*denom)
      D12 = nu_TL*E_L*h^3/(12*denom)  (== nu_LT*E_T*h^3/(12*denom))
      D66 = G_LT*h^3/12
    """
    nu_TL = nu_LT * (E_T / E_L)
    denom = 1.0 - nu_LT * nu_TL
    if abs(denom) < 1e-14:
        raise ValueError("Invalid material constants: 1 - nu_LT*nu_TL is ~0.")

    h3_over_12 = (h**3) / 12.0
    D11 = (E_L * h3_over_12) / denom
    D22 = (E_T * h3_over_12) / denom
    D12 = (nu_TL * E_L * h3_over_12) / denom
    D66 = G_LT * h3_over_12
    return D11, D22, D12, D66


def analytic_f_mn(
    *,
    D11: float,
    D22: float,
    D12: float,
    D66: float,
    rho: float,
    h: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> float:
    """
    Partner formula (Hz):
      f_mn = (pi / (2*sqrt(rho*h))) * sqrt( D11*(m/a)^4 + 2*(D12+2D66)*(m/a)^2*(n/b)^2 + D22*(n/b)^4 )
    """
    Dxy = D12 + 2.0 * D66
    term = D11 * (m / a) ** 4 + 2.0 * Dxy * (m / a) ** 2 * (n / b) ** 2 + D22 * (n / b) ** 4
    return (math.pi / (2.0 * math.sqrt(rho * h))) * math.sqrt(term)


# -----------------------------
# IO helpers (match your repo)
# -----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_result_modes(result_json: Dict[str, Any]) -> List[float]:
    """
    Your outputs look like:
      { "meta": {...}, "result": { "f0_hz": ..., "modes_hz": [ ... ] } }
    """
    if "result" in result_json and isinstance(result_json["result"], dict):
        r = result_json["result"]
        modes = r.get("modes_hz") or r.get("modes") or []
        return [float(x) for x in modes]
    # fallback: older format
    modes = result_json.get("modes_hz") or result_json.get("modes") or []
    return [float(x) for x in modes]


@dataclass
class CaseData:
    a: float
    b: float
    h: float
    rho: float
    E_L: float
    E_T: float
    G_LT: float
    nu_LT: float
    D11: float
    D22: float
    D12: float
    D66: float


def load_case_from_config(config_path: Path) -> CaseData:
    cfg = load_json(config_path)

    geom = cfg["geometry_2d"]
    a = float(geom["a"])
    b = float(geom["b"])
    h = float(geom["h"])

    mref = cfg["material_ref"]
    lib_path = Path(mref["library"])
    name = mref["name"]

    mats = load_json(lib_path)
    mat = mats[name]

    rho = float(mat["rho"])
    E_L = float(mat["E_L"])
    E_T = float(mat["E_T"])
    G_LT = float(mat["G_LT"])
    nu_LT = float(mat["nu_LT"])

    D11, D22, D12, D66 = compute_D_ij(E_L, E_T, G_LT, nu_LT, h)

    return CaseData(a, b, h, rho, E_L, E_T, G_LT, nu_LT, D11, D22, D12, D66)


def build_analytic_catalog(case: CaseData, mmax: int, nmax: int) -> List[Tuple[int, int, float]]:
    cat: List[Tuple[int, int, float]] = []
    for m in range(1, mmax + 1):
        for n in range(1, nmax + 1):
            f = analytic_f_mn(
                D11=case.D11,
                D22=case.D22,
                D12=case.D12,
                D66=case.D66,
                rho=case.rho,
                h=case.h,
                a=case.a,
                b=case.b,
                m=m,
                n=n,
            )
            cat.append((m, n, f))
    cat.sort(key=lambda t: t[2])
    return cat


def greedy_match_numeric_to_analytic(
    numeric: List[float],
    analytic: List[Tuple[int, int, float]],
    top: int,
) -> List[Dict[str, Any]]:
    """
    Greedy one-to-one matching by nearest frequency (helps if ordering swaps).
    """
    used = set()
    rows = []
    for i, fnum in enumerate(numeric[:top], start=1):
        best_j = None
        best = None
        for j, (m, n, fan) in enumerate(analytic):
            if j in used:
                continue
            diff = abs(fan - fnum)
            if best is None or diff < best:
                best = diff
                best_j = j
        assert best_j is not None
        used.add(best_j)
        m, n, fan = analytic[best_j]
        err_pct = 100.0 * (fnum - fan) / fan
        rows.append(
            {
                "mode_index": i,
                "f_num_hz": fnum,
                "m": m,
                "n": n,
                "f_analytic_hz": fan,
                "err_pct": err_pct,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to FEM config JSON, e.g. FEM/configs/rect_plate_mahogany_ref.json")
    ap.add_argument("--result", required=True, help="Path to FEM result JSON, e.g. FEM/outputs/rect_plate_mahogany_ref_result.json")
    ap.add_argument("--mmax", type=int, default=8, help="Max m for analytic catalog")
    ap.add_argument("--nmax", type=int, default=8, help="Max n for analytic catalog")
    ap.add_argument("--top", type=int, default=20, help="How many lowest numeric modes to compare")
    ap.add_argument("--out", default="", help="Optional: write comparison JSON to this path")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    res_path = Path(args.result)

    case = load_case_from_config(cfg_path)
    res = load_json(res_path)
    numeric = find_result_modes(res)

    analytic = build_analytic_catalog(case, args.mmax, args.nmax)

    # Print header (also useful to show partner calc consistency)
    print("=== Inputs (from JSON) ===")
    print(f"a={case.a}  b={case.b}  h={case.h}")
    print(f"rho={case.rho}  rho*h={case.rho*case.h}")
    print(f"E_L={case.E_L}  E_T={case.E_T}  G_LT={case.G_LT}  nu_LT={case.nu_LT}")
    print("=== Derived D_ij (same formulas as partner spec) ===")
    print(f"D11={case.D11:.6f}  D22={case.D22:.6f}  D12={case.D12:.6f}  D66={case.D66:.6f}")
    print()

    # Show first few analytic modes by true ordering
    print("=== Lowest analytic modes (sorted by frequency) ===")
    for k, (m, n, f) in enumerate(analytic[:max(10, min(args.top, 20))], start=1):
        print(f"{k:02d}: (m,n)=({m},{n})  f_an={f:.6f} Hz")
    print()

    # Compare: greedy matching (robust to ordering swaps)
    top = min(args.top, len(numeric))
    rows = greedy_match_numeric_to_analytic(numeric, analytic, top)

    print("=== Numeric -> Nearest Analytic (greedy one-to-one) ===")
    for r in rows:
        print(
            f"{r['mode_index']:02d}: f_num={r['f_num_hz']:.6f}  "
            f"~({r['m']},{r['n']})  f_an={r['f_analytic_hz']:.6f}  err={r['err_pct']:+.3f}%"
        )

    # Summary stats
    abs_err = [abs(r["err_pct"]) for r in rows]
    print()
    print("=== Summary ===")
    for N in [4, 10, 20]:
        N = min(N, len(abs_err))
        mean_abs = sum(abs_err[:N]) / N if N else 0.0
        max_abs = max(abs_err[:N]) if N else 0.0
        print(f"First {N:02d}: mean(|err|)={mean_abs:.3f}%   max(|err|)={max_abs:.3f}%")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "inputs": {
                "a": case.a, "b": case.b, "h": case.h,
                "rho": case.rho, "rho_h": case.rho * case.h,
                "E_L": case.E_L, "E_T": case.E_T, "G_LT": case.G_LT, "nu_LT": case.nu_LT,
                "D11": case.D11, "D22": case.D22, "D12": case.D12, "D66": case.D66,
                "mmax": args.mmax, "nmax": args.nmax, "top": top,
            },
            "comparison": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
