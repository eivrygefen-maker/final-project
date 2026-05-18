#!/usr/bin/env python3
"""
Stage-2 metadata tuner (no FEM solve).

Reads ``candidates_log.json``, applies a **split-quota** MMR strategy aligned with
guitar acoustics: top plate (tag 1) drives the basis; back/sides (tag 3) add color.

Default selection (150 modes total):
  * **120** modes from ``tag1_ratio >= TAG1_RATIO_MIN`` (MMR merit = normalized tag1)
  * **30** modes from ``tag3_ratio >= TAG3_RATIO_MIN``, excluding top picks (MMR merit = tag3)

Legacy combined ``wood_participation`` MMR is available via ``--legacy-combined``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from paths import DEFAULT_SHAPE_NAME, resolve_plot_output_path, shared_plot_path

# =============================
# Tuning parameters (edit here)
# =============================
LAMBDA_VAL = 0.4
# Gaussian spectral penalty bandwidth (60–480 Hz band): ~5 Hz minimum spacing between picks.
SIGMA_HZ = 5.0
UNIQUENESS_VETO_MIN = 0.1

# Split-quota (80% top / 20% back)
DEFAULT_QUOTA = 150
TOP_PLATE_QUOTA = 120
BACK_PLATE_QUOTA = 30
TAG1_RATIO_MIN = 0.0001
TAG3_RATIO_MIN = 0.0005

# Legacy combined-wood MMR (``--legacy-combined`` only)
W = 1.0
U = 0.5
WOOD_FILTER_MIN = 0.0005

# ---------------------------------------------------------------------------
# Shared-host plot export — set env SHARED_HOST_DIR or edit default in paths.py
# ---------------------------------------------------------------------------
DEFAULT_SHARED_PLOT_PATH = shared_plot_path("selection_plot.png", shape_name=DEFAULT_SHAPE_NAME)

Y_OUTLIER_QUANTILE = 0.95
Y_QUANTILE_LIMIT_SCALE = 1.2
PHYSICAL_RATIO_Y_MAX = 1.0

SELECTION_TOP = "top_plate"
SELECTION_BACK = "back_plate"
SELECTION_LEGACY = "primary"


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
            t1 = float(c.get("tag1_ratio", 0.0) or 0.0)
            t3 = float(c.get("tag3_ratio", 0.0) or 0.0)
            out.append(
                {
                    "id": int(c.get("id")),
                    "hz": float(c.get("hz")),
                    "wood_participation": float(c.get("wood_participation", t1 + t3)),
                    "uniqueness": float(c.get("uniqueness", 0.0)),
                    "tag1_ratio": t1,
                    "tag3_ratio": t3,
                }
            )
        except Exception:
            continue
    return out


def _sync_wood_participation(c: Dict) -> None:
    c["wood_participation"] = float(c["tag1_ratio"]) + float(c["tag3_ratio"])


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


def _passes_uniqueness_veto(c: Dict, uniqueness_min: float) -> bool:
    return float(c["uniqueness"]) >= float(uniqueness_min)


def _passes_legacy_veto_gates(c: Dict) -> bool:
    w = float(c["wood_participation"])
    u = float(c["uniqueness"])
    return w >= WOOD_FILTER_MIN and u >= UNIQUENESS_VETO_MIN


def mmr_select_by_merit(
    candidates: List[Dict],
    quota: int,
    merit_key: str,
    selection_type: str,
    *,
    uniqueness_min: float = UNIQUENESS_VETO_MIN,
    sigma_hz: float = SIGMA_HZ,
    lambda_val: float = LAMBDA_VAL,
) -> Tuple[List[Dict], List[Dict]]:
    """
  MMR on a single plate metric (``tag1_ratio`` or ``tag3_ratio``).

  Eligibility: ``uniqueness >= uniqueness_min`` (callers pre-filter plate ratio).
  Merit Q_i = min-max normalized ``merit_key``; spectral penalty unchanged.
    """
    if merit_key not in ("tag1_ratio", "tag3_ratio"):
        raise ValueError(f"merit_key must be tag1_ratio or tag3_ratio, got {merit_key!r}")

    pool_in = [c for c in candidates if _passes_uniqueness_veto(c, uniqueness_min)]
    if not pool_in:
        return [], list(candidates)

    merits = [float(c[merit_key]) for c in pool_in]
    merit_norm = _minmax_norm_list(merits)

    pool: List[Dict] = []
    for i, c in enumerate(pool_in):
        cc = dict(c)
        cc["_Q"] = float(merit_norm[i])
        cc["selection_type"] = str(selection_type)
        _sync_wood_participation(cc)
        pool.append(cc)

    selected: List[Dict] = []
    first = max(pool, key=lambda c: float(c["_Q"]))
    pool.remove(first)
    selected.append(first)

    target = max(0, int(quota))
    while pool and len(selected) < target:
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


def split_quota_select(
    candidates: List[Dict],
    *,
    top_quota: int = TOP_PLATE_QUOTA,
    back_quota: int = BACK_PLATE_QUOTA,
    tag1_min: float = TAG1_RATIO_MIN,
    tag3_min: float = TAG3_RATIO_MIN,
    uniqueness_min: float = UNIQUENESS_VETO_MIN,
    sigma_hz: float = SIGMA_HZ,
    lambda_val: float = LAMBDA_VAL,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Split-quota ROM selection: top plate then back plate (no duplicate IDs).

    Returns ``(merged_sorted, all_rejected, top_selected, back_selected)``.
    """
    uniq_pool = [c for c in candidates if _passes_uniqueness_veto(c, uniqueness_min)]

    top_pool = [c for c in uniq_pool if float(c["tag1_ratio"]) >= float(tag1_min)]
    top_selected, _ = mmr_select_by_merit(
        top_pool,
        top_quota,
        "tag1_ratio",
        SELECTION_TOP,
        uniqueness_min=uniqueness_min,
        sigma_hz=sigma_hz,
        lambda_val=lambda_val,
    )

    top_ids = {int(s["id"]) for s in top_selected}
    back_pool = [
        c
        for c in uniq_pool
        if int(c["id"]) not in top_ids and float(c["tag3_ratio"]) >= float(tag3_min)
    ]
    back_selected, _ = mmr_select_by_merit(
        back_pool,
        back_quota,
        "tag3_ratio",
        SELECTION_BACK,
        uniqueness_min=uniqueness_min,
        sigma_hz=sigma_hz,
        lambda_val=lambda_val,
    )

    merged = list(top_selected) + list(back_selected)
    for c in merged:
        _sync_wood_participation(c)

    merged.sort(key=lambda x: float(x["hz"]))
    selected_ids = {int(c["id"]) for c in merged}
    all_rejected = [c for c in candidates if int(c["id"]) not in selected_ids]
    return merged, all_rejected, top_selected, back_selected


def mmr_select(candidates: List[Dict], quota: int) -> Tuple[List[Dict], List[Dict]]:
    """Legacy combined-wood + uniqueness MMR (``--legacy-combined``)."""
    filtered = [c for c in candidates if _passes_legacy_veto_gates(c)]
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
        cc["selection_type"] = SELECTION_LEGACY
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
    selection_type: str = SELECTION_LEGACY,
    sigma_hz: float = SIGMA_HZ,
    lambda_val: float = LAMBDA_VAL,
) -> Tuple[List[Dict], List[Dict]]:
    """Legacy MMR with relaxed combined-wood floor."""
    filtered = [
        c
        for c in candidates
        if float(c["wood_participation"]) >= float(wood_min)
        and float(c["uniqueness"]) >= float(uniqueness_min)
    ]
    if not filtered:
        return [], list(candidates)

    w = [float(c["wood_participation"]) for c in filtered]
    u = [float(c["uniqueness"]) for c in filtered]
    w_norm = _minmax_norm_list(w)
    u_norm = _minmax_norm_list(u)

    pool: List[Dict] = []
    for i, c in enumerate(filtered):
        if str(selection_type).strip().lower() == "coverage_anchor":
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


def _ratio_values(candidates: Sequence[Dict], key: str) -> List[float]:
    out: List[float] = []
    for c in candidates:
        try:
            y = float(c.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(y) and y >= 0.0:
            out.append(y)
    return out


def _robust_ratio_y_upper(values: List[float]) -> float:
    """Y cap from quantile(0.95)*1.2 within the plotted ratio values only."""
    if not values:
        return PHYSICAL_RATIO_Y_MAX
    arr = np.asarray(values, dtype=np.float64)
    y_top = float(np.quantile(arr, Y_OUTLIER_QUANTILE)) * Y_QUANTILE_LIMIT_SCALE
    y_top = min(y_top, PHYSICAL_RATIO_Y_MAX)
    return max(y_top, 1e-6)


def _filter_plot_by_ratio(candidates: List[Dict], ratio_key: str, y_max: float) -> Tuple[List[Dict], int]:
    kept: List[Dict] = []
    dropped = 0
    for c in candidates:
        try:
            y = float(c.get(ratio_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not math.isfinite(y) or y < 0.0 or y > y_max:
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped


def _scatter_quota_axis(
    ax,
    *,
    ratio_key: str,
    ylabel: str,
    selected_top: List[Dict],
    selected_back: List[Dict],
    rejected: List[Dict],
    y_upper: float,
) -> int:
    """Draw one plate panel; return outlier drop count."""
    plot_rej, n_drop_rej = _filter_plot_by_ratio(rejected, ratio_key, y_upper)
    sel_top, n_drop_t = _filter_plot_by_ratio(selected_top, ratio_key, y_upper)
    sel_back, n_drop_b = _filter_plot_by_ratio(selected_back, ratio_key, y_upper)

    if plot_rej:
        ax.scatter(
            [float(c["hz"]) for c in plot_rej],
            [float(c[ratio_key]) for c in plot_rej],
            marker="x",
            s=18,
            c="#c62828",
            alpha=0.4,
            label="Not selected",
        )
    if sel_top:
        ax.scatter(
            [float(c["hz"]) for c in sel_top],
            [float(c[ratio_key]) for c in sel_top],
            marker="o",
            s=70,
            c="#2e7d32",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.92,
            label=f"Top quota ({len(sel_top)})",
        )
    if sel_back:
        ax.scatter(
            [float(c["hz"]) for c in sel_back],
            [float(c[ratio_key]) for c in sel_back],
            marker="^",
            s=70,
            c="#1565c0",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.92,
            label=f"Back quota ({len(sel_back)})",
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, y_upper)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    return n_drop_rej + n_drop_t + n_drop_b


def _plot_split_selection(
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

    selected_top = [c for c in selected if str(c.get("selection_type")) == SELECTION_TOP]
    selected_back = [c for c in selected if str(c.get("selection_type")) == SELECTION_BACK]
    all_for_scale = list(selected) + list(rejected)

    y1 = _robust_ratio_y_upper(_ratio_values(all_for_scale, "tag1_ratio"))
    y3 = _robust_ratio_y_upper(_ratio_values(all_for_scale, "tag3_ratio"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    n_out = 0
    n_out += _scatter_quota_axis(
        axes[0],
        ratio_key="tag1_ratio",
        ylabel="Top plate ratio (tag 1)",
        selected_top=selected_top,
        selected_back=[],
        rejected=rejected,
        y_upper=y1,
    )
    n_out += _scatter_quota_axis(
        axes[1],
        ratio_key="tag3_ratio",
        ylabel="Back/sides ratio (tag 3)",
        selected_top=[],
        selected_back=selected_back,
        rejected=rejected,
        y_upper=y3,
    )
    fig.suptitle(title)
    if n_out:
        fig.text(
            0.01,
            0.01,
            f"{n_out} point(s) above Y cap omitted per panel",
            fontsize=8,
            color="gray",
        )
    plt.tight_layout()
    if headless:
        out = resolve_plot_output_path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _plot_selection(
    selected: List[Dict],
    rejected: List[Dict],
    title: str,
    *,
    headless: bool,
    save_path: Path,
) -> None:
    """Compatibility wrapper: split-quota dual-panel plot."""
    _plot_split_selection(selected, rejected, title, headless=headless, save_path=save_path)


def _write_selected_text(selected: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,hz,wood_participation,uniqueness,tag1_ratio,tag3_ratio,Q_mmr_base,selection_type"]
    for c in selected:
        _sync_wood_participation(c)
    for c in sorted(selected, key=lambda x: float(x["hz"])):
        lines.append(
            f'{int(c["id"])},{float(c["hz"]):.6f},{float(c["wood_participation"]):.6f},'
            f'{float(c["uniqueness"]):.6f},{float(c["tag1_ratio"]):.6f},{float(c["tag3_ratio"]):.6f},'
            f'{float(c.get("_Q", 0.0)):.6f},{str(c.get("selection_type", SELECTION_TOP))}'
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_selection_metadata(
    selected: List[Dict],
    out_path: Path,
    *,
    strategy: str,
    candidate_count: int,
    selected_count: int,
    top_count: int,
    back_count: int,
    min_selected: int,
    uniqueness_threshold_used: float,
    tag1_min_used: float,
    tag3_min_used: float,
    top_quota_target: int,
    back_quota_target: int,
    window_min: Optional[float],
    window_max: Optional[float],
    wood_threshold_used: Optional[float] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "selection_strategy": str(strategy),
        "candidate_count": int(candidate_count),
        "selected_count": int(selected_count),
        "top_plate_selected": int(top_count),
        "back_plate_selected": int(back_count),
        "top_quota_target": int(top_quota_target),
        "back_quota_target": int(back_quota_target),
        "tag1_ratio_min": float(tag1_min_used),
        "tag3_ratio_min": float(tag3_min_used),
        "min_selected_target": int(min_selected),
        "uniqueness_threshold_used": float(uniqueness_threshold_used),
        "window_min_hz": None if window_min is None else float(window_min),
        "window_max_hz": None if window_max is None else float(window_max),
        "selected_candidates": [
            {
                "id": int(c["id"]),
                "hz": float(c["hz"]),
                "tag1_ratio": float(c["tag1_ratio"]),
                "tag3_ratio": float(c["tag3_ratio"]),
                "selection_type": str(c.get("selection_type", "")),
            }
            for c in sorted(selected, key=lambda x: float(x["hz"]))
        ],
    }
    if wood_threshold_used is not None:
        payload["wood_threshold_used"] = float(wood_threshold_used)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split-quota MMR Stage-2 tuner (120 top tag1 + 30 back tag3 = 150 modes)"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=_default_candidates_path(),
        help="Path to candidates_log.json (default: FEM/SORTING/candidates_log.json)",
    )
    parser.add_argument(
        "--quota",
        type=int,
        default=DEFAULT_QUOTA,
        help="Total modes (default 150; split 120/30 unless overridden)",
    )
    parser.add_argument("--top-quota", type=int, default=TOP_PLATE_QUOTA, help="Top plate MMR quota (default 120)")
    parser.add_argument("--back-quota", type=int, default=BACK_PLATE_QUOTA, help="Back plate MMR quota (default 30)")
    parser.add_argument(
        "--tag1-min",
        type=float,
        default=TAG1_RATIO_MIN,
        help="Minimum tag1_ratio for top pool (default 0.0001)",
    )
    parser.add_argument(
        "--tag3-min",
        type=float,
        default=TAG3_RATIO_MIN,
        help="Minimum tag3_ratio for back pool (default 0.0005)",
    )
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
            "Output path for selection plot when --headless "
            f"(default: shared host {DEFAULT_SHARED_PLOT_PATH.as_posix()})"
        ),
    )
    parser.add_argument("--window-min", type=float, default=None, help="Optional minimum frequency for selection window.")
    parser.add_argument("--window-max", type=float, default=None, help="Optional maximum frequency for selection window.")
    parser.add_argument(
        "--legacy-combined",
        action="store_true",
        help="Use legacy combined wood_participation MMR instead of split-quota selection.",
    )
    parser.add_argument(
        "--min-selected",
        type=int,
        default=0,
        help="Minimum selected modes; with --adaptive-veto relaxes uniqueness only (split mode).",
    )
    parser.add_argument(
        "--adaptive-veto",
        action="store_true",
        help="Relax uniqueness floor until --min-selected is met (split or legacy).",
    )
    parser.add_argument("--adaptive-steps", type=int, default=8, help="Adaptive uniqueness relaxation steps.")
    parser.add_argument(
        "--uniqueness-floor-min",
        type=float,
        default=0.0,
        help="Lowest uniqueness veto during adaptive relaxation.",
    )
    parser.add_argument(
        "--selection-type",
        type=str,
        default=SELECTION_LEGACY,
        help="Label for legacy-combined mode only.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=_default_metadata_path(),
        help="Path to JSON metadata for the selection run.",
    )
    parser.add_argument("--sigma", type=float, default=SIGMA_HZ, help=f"MMR sigma Hz (default {SIGMA_HZ}).")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=LAMBDA_VAL, help=f"MMR lambda (default {LAMBDA_VAL}).")
    parser.add_argument(
        "--wood-floor-min",
        type=float,
        default=0.0,
        help="Legacy combined mode: lowest wood floor under --adaptive-veto.",
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

    total_quota = max(1, int(args.quota))
    top_quota = max(0, int(args.top_quota))
    back_quota = max(0, int(args.back_quota))
    if top_quota + back_quota != total_quota and not args.legacy_combined:
        back_quota = max(0, total_quota - top_quota)
        print(
            f"Note: top_quota + back_quota adjusted to {top_quota} + {back_quota} = {top_quota + back_quota} "
            f"(--quota {total_quota})"
        )

    sigma_hz = max(1e-6, float(args.sigma))
    lambda_val = min(1.0, max(0.0, float(args.lambda_val)))
    min_selected_target = max(0, int(args.min_selected))
    adaptive_steps = max(1, int(args.adaptive_steps))
    chosen_uniq = float(UNIQUENESS_VETO_MIN)
    min_uniq = max(0.0, float(args.uniqueness_floor_min))
    tag1_min = float(args.tag1_min)
    tag3_min = float(args.tag3_min)

    top_selected: List[Dict] = []
    back_selected: List[Dict] = []
    selected: List[Dict] = []
    rejected: List[Dict] = []
    strategy = "split_quota"

    if args.legacy_combined:
        strategy = "legacy_combined"
        base_wood = float(WOOD_FILTER_MIN)
        min_wood = max(0.0, float(args.wood_floor_min))
        chosen_wood = base_wood
        selection_type = str(args.selection_type).strip() or SELECTION_LEGACY
        if args.adaptive_veto:
            best_selected: List[Dict] = []
            best_rejected: List[Dict] = list(candidates)
            for i in range(adaptive_steps + 1):
                t = float(i) / float(adaptive_steps)
                wood_thr = base_wood - (base_wood - min_wood) * t
                uniq_thr = chosen_uniq - (chosen_uniq - min_uniq) * t
                sel_i, rej_i = mmr_select_with_thresholds(
                    candidates,
                    quota=total_quota,
                    wood_min=max(min_wood, wood_thr),
                    uniqueness_min=max(min_uniq, uniq_thr),
                    selection_type=selection_type,
                    sigma_hz=sigma_hz,
                    lambda_val=lambda_val,
                )
                if len(sel_i) > len(best_selected):
                    best_selected, best_rejected = sel_i, rej_i
                    chosen_wood, chosen_uniq = max(min_wood, wood_thr), max(min_uniq, uniq_thr)
                if min_selected_target > 0 and len(sel_i) >= min_selected_target:
                    best_selected, best_rejected = sel_i, rej_i
                    chosen_wood, chosen_uniq = max(min_wood, wood_thr), max(min_uniq, uniq_thr)
                    break
            selected, rejected = best_selected, best_rejected
        else:
            selected, rejected = mmr_select_with_thresholds(
                candidates,
                quota=total_quota,
                wood_min=base_wood,
                uniqueness_min=chosen_uniq,
                selection_type=selection_type,
                sigma_hz=sigma_hz,
                lambda_val=lambda_val,
            )
        selected.sort(key=lambda x: float(x["hz"]))
        print(
            f"Legacy combined MMR: selected={len(selected)} "
            f"(wood>={chosen_wood:.6f}, uniqueness>={chosen_uniq:.6f})"
        )
        _write_selection_metadata(
            selected,
            args.metadata_out,
            strategy=strategy,
            candidate_count=len(candidates),
            selected_count=len(selected),
            top_count=0,
            back_count=0,
            min_selected=min_selected_target,
            uniqueness_threshold_used=chosen_uniq,
            tag1_min_used=tag1_min,
            tag3_min_used=tag3_min,
            top_quota_target=top_quota,
            back_quota_target=back_quota,
            window_min=args.window_min,
            window_max=args.window_max,
            wood_threshold_used=chosen_wood if args.legacy_combined else None,
        )
    else:
        if args.adaptive_veto:
            best_merged: List[Dict] = []
            best_rejected: List[Dict] = list(candidates)
            best_top: List[Dict] = []
            best_back: List[Dict] = []
            for i in range(adaptive_steps + 1):
                t = float(i) / float(adaptive_steps)
                uniq_thr = chosen_uniq - (chosen_uniq - min_uniq) * t
                merged_i, rej_i, top_i, back_i = split_quota_select(
                    candidates,
                    top_quota=top_quota,
                    back_quota=back_quota,
                    tag1_min=tag1_min,
                    tag3_min=tag3_min,
                    uniqueness_min=max(min_uniq, uniq_thr),
                    sigma_hz=sigma_hz,
                    lambda_val=lambda_val,
                )
                if len(merged_i) > len(best_merged):
                    best_merged, best_rejected = merged_i, rej_i, top_i, back_i
                    chosen_uniq = max(min_uniq, uniq_thr)
                if min_selected_target > 0 and len(merged_i) >= min_selected_target:
                    best_merged, best_rejected = merged_i, rej_i, top_i, back_i
                    chosen_uniq = max(min_uniq, uniq_thr)
                    break
            selected, rejected, top_selected, back_selected = (
                best_merged,
                best_rejected,
                best_top,
                best_back,
            )
        else:
            selected, rejected, top_selected, back_selected = split_quota_select(
                candidates,
                top_quota=top_quota,
                back_quota=back_quota,
                tag1_min=tag1_min,
                tag3_min=tag3_min,
                uniqueness_min=chosen_uniq,
                sigma_hz=sigma_hz,
                lambda_val=lambda_val,
            )

        if len(top_selected) < top_quota:
            print(
                f"Warning: top plate quota shortfall ({len(top_selected)}/{top_quota}); "
                f"pool may lack tag1_ratio >= {tag1_min:g}",
                file=sys.stderr,
            )
        if len(back_selected) < back_quota:
            print(
                f"Warning: back plate quota shortfall ({len(back_selected)}/{back_quota}); "
                f"pool may lack tag3_ratio >= {tag3_min:g} after top dedupe",
                file=sys.stderr,
            )

        print(
            f"Split-quota MMR: top={len(top_selected)}/{top_quota} (tag1>={tag1_min:g}), "
            f"back={len(back_selected)}/{back_quota} (tag3>={tag3_min:g}), "
            f"total={len(selected)}, uniqueness>={chosen_uniq:.6f}"
        )
        _write_selection_metadata(
            selected,
            args.metadata_out,
            strategy=strategy,
            candidate_count=len(candidates),
            selected_count=len(selected),
            top_count=len(top_selected),
            back_count=len(back_selected),
            min_selected=min_selected_target,
            uniqueness_threshold_used=chosen_uniq,
            tag1_min_used=tag1_min,
            tag3_min_used=tag3_min,
            top_quota_target=top_quota,
            back_quota_target=back_quota,
            window_min=args.window_min,
            window_max=args.window_max,
        )

    print(f"Selected {len(selected)} modes. Exporting to {args.export}...")
    _write_selected_text(selected, args.export)
    print(f"Exported: {args.export.resolve()}")
    print(f"Metadata: {args.metadata_out.resolve()}")

    title = (
        f"Split-quota MMR | top={len(top_selected)} back={len(back_selected)} "
        f"rejected={len(rejected)} | λ={lambda_val}, σ={sigma_hz} Hz | uniq≥{chosen_uniq:.2f}"
    )
    if args.legacy_combined:
        title = (
            f"Legacy combined MMR | selected={len(selected)} rejected={len(rejected)} | "
            f"λ={lambda_val}, σ={sigma_hz} Hz"
        )

    plot_dest = resolve_plot_output_path(args.plot_out)
    _plot_split_selection(
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
