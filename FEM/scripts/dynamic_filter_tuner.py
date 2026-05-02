#!/usr/bin/env python3
"""
Interactive Stage-2 metadata tuner (no FEM solve).

Reads FEM/SORTING/candidates_log.json, applies hard veto gates (wood floor and
uniqueness anti-echo), then Maximal Marginal Relevance (MMR) with Gaussian
frequency similarity. Plots MMR-selected vs all rejected candidates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# =============================
# Tuning parameters (edit here)
# =============================
W = 1.0
U = 0.5
LAMBDA_VAL = 0.7
SIGMA_HZ = 8.0
# Wood gate here is separate from ``fem_master_dynamic`` zone floors (tuner is stage-2 MMR).
WOOD_FILTER_MIN = 0.0005
UNIQUENESS_VETO_MIN = 0.1
DEFAULT_QUOTA = 100


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_candidates_path() -> Path:
    return _project_root() / "FEM" / "SORTING" / "candidates_log.json"


def _default_selection_plot_path() -> Path:
    return _project_root() / "FEM" / "SORTING" / "selection_plot.png"


def _load_candidates(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Candidates log not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        candidates = list(data.get("candidates", []))
    elif isinstance(data, list):
        candidates = list(data)
    else:
        raise ValueError("Unsupported JSON format. Expected dict with 'candidates' or a list.")
    out: List[Dict] = []
    for c in candidates:
        try:
            out.append(
                {
                    "id": int(c.get("id")),
                    "hz": float(c.get("hz")),
                    "wood_participation": float(c.get("wood_participation", 0.0)),
                    "uniqueness": float(c.get("uniqueness", 0.0)),
                    "tag1_ratio": float(c.get("tag1_ratio", 0.0)),
                    "tag3_ratio": float(c.get("tag3_ratio", 0.0)),
                }
            )
        except Exception:
            continue
    return out


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-15:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _similarity_gaussian(freq_i: float, freq_j: float, sigma: float) -> float:
    d = float(freq_i) - float(freq_j)
    return math.exp(-(d * d) / (2.0 * sigma * sigma))


def _passes_veto_gates(c: Dict) -> bool:
    """Hard vetoes before MMR: air-noise floor and geometric echo / duplicate shapes."""
    w = float(c["wood_participation"])
    u = float(c["uniqueness"])
    return w >= WOOD_FILTER_MIN and u >= UNIQUENESS_VETO_MIN


def mmr_select(candidates: List[Dict], quota: int) -> Tuple[List[Dict], List[Dict]]:
    """
    1) Veto gates (not eligible for MMR): wood < WOOD_FILTER_MIN or uniqueness < UNIQUENESS_VETO_MIN.
    2) On survivors: min-max w, u; Q_i = W * w_norm_i + U * u_norm_i.
    3) MMR: Penalty_k = max_j S(k,j), S Gaussian in Hz, sigma = SIGMA_HZ;
       MMR_k = lambda * Q_k - (1-lambda) * Penalty_k. First pick: argmax Q_i.
    Rejected = every candidate not in the final selected set (vetoes + MMR overflow).
    """
    filtered = [c for c in candidates if _passes_veto_gates(c)]
    if not filtered:
        return [], list(candidates)

    w = np.array([float(c["wood_participation"]) for c in filtered], dtype=np.float64)
    u = np.array([float(c["uniqueness"]) for c in filtered], dtype=np.float64)
    w_norm = _minmax_norm(w)
    u_norm = _minmax_norm(u)

    pool: List[Dict] = []
    for i, c in enumerate(filtered):
        q_i = W * float(w_norm[i]) + U * float(u_norm[i])
        cc = dict(c)
        cc["_Q"] = float(q_i)
        pool.append(cc)

    selected: List[Dict] = []
    first = max(pool, key=lambda c: float(c["_Q"]))
    pool.remove(first)
    selected.append(first)

    while pool and len(selected) < quota:
        best: Dict | None = None
        best_mmr = -float("inf")
        for k in pool:
            fk = float(k["hz"])
            penalty = max(_similarity_gaussian(fk, float(sj["hz"]), SIGMA_HZ) for sj in selected)
            mmr_k = (LAMBDA_VAL * float(k["_Q"])) - ((1.0 - LAMBDA_VAL) * penalty)
            if mmr_k > best_mmr:
                best_mmr = mmr_k
                best = k
        if best is None:
            break
        pool.remove(best)
        selected.append(best)

    selected_ids = {int(s["id"]) for s in selected}
    rejected = [c for c in candidates if int(c["id"]) not in selected_ids]
    return selected, rejected


def _plot_selection(
    selected: List[Dict],
    rejected: List[Dict],
    title: str,
    *,
    headless: bool,
    save_path: Path,
) -> None:
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    if rejected:
        rx = np.array([float(c["hz"]) for c in rejected], dtype=np.float64)
        ry = np.array([float(c["wood_participation"]) for c in rejected], dtype=np.float64)
        ax.scatter(
            rx,
            ry,
            marker="x",
            s=22,
            c="red",
            alpha=0.55,
            label="Rejected (veto gates + not MMR-selected)",
        )

    if selected:
        sx = np.array([float(c["hz"]) for c in selected], dtype=np.float64)
        sy = np.array([float(c["wood_participation"]) for c in selected], dtype=np.float64)
        ax.scatter(
            sx,
            sy,
            marker="o",
            s=85,
            c="green",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            label="MMR selected",
        )
        for c in selected:
            ax.annotate(
                str(c["id"]),
                (float(c["hz"]), float(c["wood_participation"])),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=7,
                color="darkgreen",
            )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Wood participation (raw)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    if headless:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _write_selected_text(selected: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,hz,wood_participation,uniqueness,tag1_ratio,tag3_ratio,Q_mmr_base"]
    for c in sorted(selected, key=lambda x: float(x["hz"])):
        lines.append(
            f'{int(c["id"])},{float(c["hz"]):.6f},{float(c["wood_participation"]):.6f},'
            f'{float(c["uniqueness"]):.6f},{float(c["tag1_ratio"]):.6f},{float(c["tag3_ratio"]):.6f},'
            f'{float(c.get("_Q", 0.0)):.6f}'
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="MMR-based Stage-2 filter tuner over candidates_log.json")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=_default_candidates_path(),
        help="Path to candidates_log.json (default: FEM/SORTING/candidates_log.json)",
    )
    parser.add_argument("--quota", type=int, default=DEFAULT_QUOTA, help="Number of modes to select (default 100)")
    parser.add_argument(
        "--export",
        type=Path,
        default=_project_root() / "FEM" / "SORTING" / "selected_modes.csv",
        help="CSV export of selected modes",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Non-interactive: save plot to PNG instead of plt.show(), then exit.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=_default_selection_plot_path(),
        help="Output path for MMR plot when --headless (default: FEM/SORTING/selection_plot.png)",
    )
    args = parser.parse_args()

    candidates = _load_candidates(args.candidates)
    if not candidates:
        print(f"No valid candidates found in: {args.candidates}")
        return 1

    quota = max(1, int(args.quota))
    selected, rejected = mmr_select(candidates, quota=quota)

    print(f"Selected {len(selected)} modes. Exporting to text...")
    _write_selected_text(selected, args.export)
    print(f"Exported: {args.export}")

    title = (
        f"MMR tuner | selected={len(selected)} rejected={len(rejected)} | "
        f"W={W}, U={U}, λ={LAMBDA_VAL}, σ={SIGMA_HZ} Hz | "
        f"vetoes: wood≥{WOOD_FILTER_MIN}, uniqueness≥{UNIQUENESS_VETO_MIN}"
    )
    _plot_selection(
        selected,
        rejected,
        title,
        headless=bool(args.headless),
        save_path=args.plot_out.resolve(),
    )
    if args.headless:
        print(f"Saved selection plot: {args.plot_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
