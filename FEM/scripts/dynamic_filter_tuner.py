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
from typing import Dict, List, Optional, Tuple

import numpy as np

from paths import DEFAULT_SHAPE_NAME, resolve_plot_output_path, shared_plot_path

# =============================
# Tuning parameters (edit here)
# =============================
W = 1.0
U = 0.5
LAMBDA_VAL = 0.4
SIGMA_HZ = 15.0
# Wood gate here is separate from ``fem_master_dynamic`` zone floors (tuner is stage-2 MMR).
WOOD_FILTER_MIN = 0.0005
UNIQUENESS_VETO_MIN = 0.1
DEFAULT_QUOTA = 150

# ---------------------------------------------------------------------------
# Shared-host plot export — set env SHARED_HOST_DIR or edit default in paths.py
# Example: export SHARED_HOST_DIR=/media/sf_gmar
# Plots: {SHARED_HOST_DIR}/{shape}/plots/<filename>
# ---------------------------------------------------------------------------
DEFAULT_SHARED_PLOT_PATH = shared_plot_path("selection_plot.png", shape_name=DEFAULT_SHAPE_NAME)

# Y-axis scaling: cap = quantile(0.95) * scale; drop display outliers above cap (diverged runs).
Y_OUTLIER_QUANTILE = 0.95
Y_QUANTILE_LIMIT_SCALE = 1.2
# Wood participation is a sum of normalized plate-energy ratios; values above ~1 are non-physical.
PHYSICAL_WOOD_PARTICIPATION_Y_MAX = 1.0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_candidates_path() -> Path:
    return _project_root() / "FEM" / "SORTING" / "candidates_log.json"


def _default_selection_plot_path() -> Path:
    return DEFAULT_SHARED_PLOT_PATH


def _default_metadata_path() -> Path:
    return _project_root() / "FEM" / "SORTING" / "selection_metadata.json"


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


def _minmax_norm_list(x: List[float]) -> List[float]:
    if not x:
        return []
    lo = float(min(x))
    hi = float(max(x))
    if hi - lo < 1e-15:
        return [0.5 for _ in x]
    den = hi - lo
    return [(float(v) - lo) / den for v in x]


def _similarity_gaussian(freq_i: float, freq_j: float, sigma: float) -> float:
    d = float(freq_i) - float(freq_j)
    return math.exp(-(d * d) / (2.0 * sigma * sigma))


def _passes_veto_gates(c: Dict) -> bool:
    """Hard vetoes before MMR: air-noise floor and geometric echo / duplicate shapes."""
    w = float(c["wood_participation"])
    u = float(c["uniqueness"])
    return w >= WOOD_FILTER_MIN and u >= UNIQUENESS_VETO_MIN


def _passes_veto_gates_with_thresholds(c: Dict, wood_min: float, uniq_min: float) -> bool:
    w = float(c["wood_participation"])
    u = float(c["uniqueness"])
    return w >= float(wood_min) and u >= float(uniq_min)


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

    w = [float(c["wood_participation"]) for c in filtered]
    u = [float(c["uniqueness"]) for c in filtered]
    w_norm = _minmax_norm_list(w)
    u_norm = _minmax_norm_list(u)

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


def mmr_select_with_thresholds(
    candidates: List[Dict],
    quota: int,
    *,
    wood_min: float,
    uniqueness_min: float,
    selection_type: str = "primary",
    sigma_hz: float = SIGMA_HZ,
    lambda_val: float = LAMBDA_VAL,
) -> Tuple[List[Dict], List[Dict]]:
    filtered = [c for c in candidates if _passes_veto_gates_with_thresholds(c, wood_min, uniqueness_min)]
    if not filtered:
        return [], list(candidates)

    w = [float(c["wood_participation"]) for c in filtered]
    u = [float(c["uniqueness"]) for c in filtered]
    w_norm = _minmax_norm_list(w)
    u_norm = _minmax_norm_list(u)

    pool: List[Dict] = []
    for i, c in enumerate(filtered):
        if str(selection_type).strip().lower() == "coverage_anchor":
            # Coverage anchors intentionally prioritize basis diversity over acoustic loudness.
            q_i = float(u_norm[i])
        else:
            q_i = W * float(w_norm[i]) + U * float(u_norm[i])
        cc = dict(c)
        cc["_Q"] = float(q_i)
        cc["selection_type"] = str(selection_type)
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
            penalty = max(_similarity_gaussian(fk, float(sj["hz"]), sigma_hz) for sj in selected)
            mmr_k = (float(lambda_val) * float(k["_Q"])) - ((1.0 - float(lambda_val)) * penalty)
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


def _filter_frequency_window(candidates: List[Dict], hz_min: Optional[float], hz_max: Optional[float]) -> List[Dict]:
    if hz_min is None and hz_max is None:
        return list(candidates)
    out: List[Dict] = []
    for c in candidates:
        hz = float(c["hz"])
        if hz_min is not None and hz < float(hz_min):
            continue
        if hz_max is not None and hz > float(hz_max):
            continue
        out.append(c)
    return out


def _wood_participation_values(candidates: List[Dict]) -> List[float]:
    out: List[float] = []
    for c in candidates:
        try:
            y = float(c.get("wood_participation", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(y) and y >= 0.0:
            out.append(y)
    return out


def _robust_wood_y_upper(all_y: List[float]) -> float:
    """Cap Y-axis at min(quantile(0.95)*1.2, physical ceiling) for readable wood-participation scale."""
    if not all_y:
        return PHYSICAL_WOOD_PARTICIPATION_Y_MAX
    arr = np.asarray(all_y, dtype=np.float64)
    y_top = float(np.quantile(arr, Y_OUTLIER_QUANTILE)) * Y_QUANTILE_LIMIT_SCALE
    y_top = min(y_top, PHYSICAL_WOOD_PARTICIPATION_Y_MAX)
    return max(y_top, 1e-6)


def _filter_plot_candidates_by_y(candidates: List[Dict], y_max: float) -> Tuple[List[Dict], int]:
    """Drop non-finite or extreme Y outliers before scatter (keeps scale readable)."""
    kept: List[Dict] = []
    dropped = 0
    for c in candidates:
        try:
            y = float(c.get("wood_participation", 0.0) or 0.0)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not math.isfinite(y) or y < 0.0 or y > y_max:
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped


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

    all_y = _wood_participation_values(selected) + _wood_participation_values(rejected)
    y_upper = _robust_wood_y_upper(all_y)
    plot_rejected, n_drop_rej = _filter_plot_candidates_by_y(rejected, y_upper)
    plot_selected, n_drop_sel = _filter_plot_candidates_by_y(selected, y_upper)
    n_outliers = n_drop_rej + n_drop_sel

    fig, ax = plt.subplots(figsize=(12, 6))

    if plot_rejected:
        rx = [float(c["hz"]) for c in plot_rejected]
        ry = [float(c["wood_participation"]) for c in plot_rejected]
        ax.scatter(
            rx,
            ry,
            marker="x",
            s=22,
            c="red",
            alpha=0.55,
            label="Rejected (veto gates + not MMR-selected)",
        )

    if plot_selected:
        sx = [float(c["hz"]) for c in plot_selected]
        sy = [float(c["wood_participation"]) for c in plot_selected]
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
        for c in plot_selected:
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
    ax.set_ylim(0.0, y_upper)
    if n_outliers:
        ax.text(
            0.01,
            0.99,
            f"{n_outliers} outlier(s) above Y cap omitted",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="gray",
        )
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    if headless:
        out = resolve_plot_output_path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _write_selected_text(selected: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,hz,wood_participation,uniqueness,tag1_ratio,tag3_ratio,Q_mmr_base,selection_type"]
    for c in sorted(selected, key=lambda x: float(x["hz"])):
        lines.append(
            f'{int(c["id"])},{float(c["hz"]):.6f},{float(c["wood_participation"]):.6f},'
            f'{float(c["uniqueness"]):.6f},{float(c["tag1_ratio"]):.6f},{float(c["tag3_ratio"]):.6f},'
            f'{float(c.get("_Q", 0.0)):.6f},{str(c.get("selection_type", "primary"))}'
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_selection_metadata(
    selected: List[Dict],
    out_path: Path,
    *,
    selection_type: str,
    candidate_count: int,
    selected_count: int,
    min_selected: int,
    wood_threshold_used: float,
    uniqueness_threshold_used: float,
    window_min: Optional[float],
    window_max: Optional[float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "selection_type": str(selection_type),
        "candidate_count": int(candidate_count),
        "selected_count": int(selected_count),
        "min_selected_target": int(min_selected),
        "wood_threshold_used": float(wood_threshold_used),
        "uniqueness_threshold_used": float(uniqueness_threshold_used),
        "window_min_hz": None if window_min is None else float(window_min),
        "window_max_hz": None if window_max is None else float(window_max),
        "selected_candidates": [
            {
                "id": int(c["id"]),
                "hz": float(c["hz"]),
                "selection_type": str(c.get("selection_type", selection_type)),
            }
            for c in sorted(selected, key=lambda x: float(x["hz"]))
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
        help=(
            "Output path for MMR plot when --headless "
            f"(default: shared host {DEFAULT_SHARED_PLOT_PATH.as_posix()})"
        ),
    )
    parser.add_argument("--window-min", type=float, default=None, help="Optional minimum frequency for selection window.")
    parser.add_argument("--window-max", type=float, default=None, help="Optional maximum frequency for selection window.")
    parser.add_argument(
        "--min-selected",
        type=int,
        default=0,
        help="Minimum selected modes target; if unmet and --adaptive-veto is enabled, veto thresholds are relaxed.",
    )
    parser.add_argument(
        "--adaptive-veto",
        action="store_true",
        help="Adaptively lower wood/uniqueness veto thresholds until --min-selected is reached or limits are hit.",
    )
    parser.add_argument(
        "--adaptive-steps",
        type=int,
        default=8,
        help="Number of adaptive-veto relaxation steps (default: 8).",
    )
    parser.add_argument(
        "--wood-floor-min",
        type=float,
        default=0.0,
        help="Lowest allowed wood veto floor during adaptive relaxation (default: 0.0).",
    )
    parser.add_argument(
        "--uniqueness-floor-min",
        type=float,
        default=0.0,
        help="Lowest allowed uniqueness veto floor during adaptive relaxation (default: 0.0).",
    )
    parser.add_argument(
        "--selection-type",
        type=str,
        default="primary",
        help='Selection mode label. Use "coverage_anchor" to prioritize uniqueness/frequency spacing over wood loudness.',
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=_default_metadata_path(),
        help="Path to JSON metadata that records selected candidates and selection_type.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA_HZ,
        help=f"Gaussian frequency similarity sigma in Hz (default: {SIGMA_HZ}).",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_val",
        type=float,
        default=LAMBDA_VAL,
        help=f"MMR trade-off lambda in [0,1] (default: {LAMBDA_VAL}).",
    )
    args = parser.parse_args()

    candidates = _load_candidates(args.candidates)
    if not candidates:
        print(f"No valid candidates found in: {args.candidates}")
        return 1

    candidates = _filter_frequency_window(candidates, args.window_min, args.window_max)
    if not candidates:
        print(
            f"No valid candidates in requested window [{args.window_min}, {args.window_max}] from: {args.candidates}"
        )
        return 1

    quota = max(1, int(args.quota))
    sigma_hz = max(1e-6, float(args.sigma))
    lambda_val = min(1.0, max(0.0, float(args.lambda_val)))
    min_selected_target = max(0, int(args.min_selected))
    adaptive_steps = max(1, int(args.adaptive_steps))
    base_wood = float(WOOD_FILTER_MIN)
    base_uniq = float(UNIQUENESS_VETO_MIN)
    min_wood = max(0.0, float(args.wood_floor_min))
    min_uniq = max(0.0, float(args.uniqueness_floor_min))
    selection_type = str(args.selection_type).strip() or "primary"
    if selection_type.lower() == "coverage_anchor":
        min_wood = 0.0
        min_selected_target = max(5, min_selected_target)
        quota = max(quota, min_selected_target)

    chosen_wood = base_wood
    chosen_uniq = base_uniq
    if args.adaptive_veto:
        best_selected: List[Dict] = []
        best_rejected: List[Dict] = list(candidates)
        for i in range(adaptive_steps + 1):
            t = float(i) / float(adaptive_steps)
            wood_thr = base_wood - (base_wood - min_wood) * t
            uniq_thr = base_uniq - (base_uniq - min_uniq) * t
            sel_i, rej_i = mmr_select_with_thresholds(
                candidates,
                quota=quota,
                wood_min=max(min_wood, wood_thr),
                uniqueness_min=max(min_uniq, uniq_thr),
                selection_type=selection_type,
                sigma_hz=sigma_hz,
                lambda_val=lambda_val,
            )
            if len(sel_i) > len(best_selected):
                best_selected = sel_i
                best_rejected = rej_i
                chosen_wood = max(min_wood, wood_thr)
                chosen_uniq = max(min_uniq, uniq_thr)
            if min_selected_target > 0 and len(sel_i) >= min_selected_target:
                best_selected = sel_i
                best_rejected = rej_i
                chosen_wood = max(min_wood, wood_thr)
                chosen_uniq = max(min_uniq, uniq_thr)
                break
        selected, rejected = best_selected, best_rejected
        print(
            f"Adaptive veto thresholds used: wood>={chosen_wood:.6f}, uniqueness>={chosen_uniq:.6f}. "
            f"selected={len(selected)} target={min_selected_target or quota}"
        )
    else:
        selected, rejected = mmr_select_with_thresholds(
            candidates,
            quota=quota,
            wood_min=base_wood,
            uniqueness_min=base_uniq,
            selection_type=selection_type,
            sigma_hz=sigma_hz,
            lambda_val=lambda_val,
        )

    print(f"Selected {len(selected)} modes. Exporting to text...")
    _write_selected_text(selected, args.export)
    _write_selection_metadata(
        selected,
        args.metadata_out,
        selection_type=selection_type,
        candidate_count=len(candidates),
        selected_count=len(selected),
        min_selected=min_selected_target,
        wood_threshold_used=chosen_wood,
        uniqueness_threshold_used=chosen_uniq,
        window_min=args.window_min,
        window_max=args.window_max,
    )
    print(f"Exported: {args.export}")
    print(f"Metadata: {args.metadata_out}")

    title = (
        f"MMR tuner | selected={len(selected)} rejected={len(rejected)} | "
        f"W={W}, U={U}, λ={lambda_val}, σ={sigma_hz} Hz | "
        f"vetoes: wood≥{WOOD_FILTER_MIN}, uniqueness≥{UNIQUENESS_VETO_MIN}"
    )
    plot_dest = resolve_plot_output_path(args.plot_out)
    _plot_selection(
        selected,
        rejected,
        title,
        headless=bool(args.headless),
        save_path=plot_dest,
    )
    if args.headless:
        print(f"Saved selection plot: {plot_dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
