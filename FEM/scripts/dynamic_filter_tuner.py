#!/usr/bin/env python3
"""
Stage-2 tuner for the 3D coupled ROM pipeline (no FEM solve).

Two-layer prune → unified pool selection → CSV export. ``dominant_tag`` is diagnostic only;
every selected mode keeps its full coupled displacement vector for ``package_rom.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fem_harvest_filter import HARVEST_FILTER_POLICY_VERSION
from fem_rom_postprocess import (
    DOMINANT_TAG_BACK,
    DOMINANT_TAG_TOP,
    MODAL_PRUNE_DF_HF_HZ,
    MODAL_PRUNE_DF_LF_MF_HZ,
    MODAL_PRUNE_HF_CROSSOVER_HZ,
    MODAL_PRUNE_LOG_P_FRAC_TOL_DEFAULT,
    MODAL_PRUNE_TAG_TOL_DEFAULT,
    annotate_dominant_tags,
    dominant_tag_for_row,
    plot_modal_pool_diagnostics,
    prune_modes_two_layer,
)
from paths import DEFAULT_SHAPE_NAME, resolve_plot_output_path, shared_plot_path

LAMBDA_VAL = 0.4
SIGMA_HZ = 5.0
UNIQUENESS_VETO_MIN = 0.04
DEFAULT_QUOTA = 0
MMR_WOOD_WEIGHT = 0.25
MMR_UNIQ_WEIGHT = 0.25
MMR_P_FRAC_WEIGHT = 0.5

DEFAULT_SHARED_PLOT_PATH = shared_plot_path("selection_plot.png", shape_name=DEFAULT_SHAPE_NAME)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_candidates_path() -> Path:
    return _project_root() / "FEM" / "SORTING" / "candidates_log.json"


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
            row = {
                "id": int(c.get("id")),
                "hz": float(c.get("hz")),
                "wood_participation": float(c.get("wood_participation", t1 + t3)),
                "uniqueness": float(c.get("uniqueness", 0.0)),
                "tag1_ratio": t1,
                "tag3_ratio": t3,
            }
            if c.get("source_target_hz") is not None:
                try:
                    row["source_target_hz"] = float(c.get("source_target_hz"))
                except (TypeError, ValueError):
                    pass
            if c.get("harvest_filter_policy"):
                row["harvest_filter_policy"] = str(c.get("harvest_filter_policy"))
            if c.get("harvest_class"):
                row["harvest_class"] = str(c.get("harvest_class"))
            if c.get("p_frac") is not None:
                try:
                    row["p_frac"] = float(c.get("p_frac"))
                except (TypeError, ValueError):
                    pass
            out.append(row)
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
    return [(float(v) - lo) / (hi - lo) for v in x]


def _similarity_gaussian(freq_i: float, freq_j: float, sigma: float) -> float:
    d = float(freq_i) - float(freq_j)
    return math.exp(-(d * d) / (2.0 * sigma * sigma))


def _passes_uniqueness_veto(c: Dict, uniqueness_min: float) -> bool:
    return float(c["uniqueness"]) >= float(uniqueness_min)


def _filter_frequency_window(
    candidates: List[Dict], hz_min: Optional[float], hz_max: Optional[float]
) -> List[Dict]:
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


def unified_mmr_select(
    candidates: List[Dict],
    quota: int,
    *,
    uniqueness_min: float = UNIQUENESS_VETO_MIN,
    sigma_hz: float = SIGMA_HZ,
    lambda_val: float = LAMBDA_VAL,
) -> Tuple[List[Dict], List[Dict]]:
    """Single-pool MMR; ``quota <= 0`` keeps every pruned candidate."""
    target = int(quota)
    if target <= 0:
        return annotate_dominant_tags(sorted(candidates, key=lambda c: float(c["hz"]))), []

    pool_in = [c for c in candidates if _passes_uniqueness_veto(c, uniqueness_min)]
    if not pool_in:
        return [], list(candidates)

    p_vals = [float(c.get("p_frac", 0.0) or 0.0) for c in pool_in]
    has_p = max(p_vals) > 1.0e-30
    w_norm = _minmax_norm_list([float(c["wood_participation"]) for c in pool_in])
    u_norm = _minmax_norm_list([float(c["uniqueness"]) for c in pool_in])
    p_norm = _minmax_norm_list(p_vals) if has_p else [0.0] * len(pool_in)

    pool: List[Dict] = []
    for i, c in enumerate(pool_in):
        if has_p:
            q_i = (
                MMR_P_FRAC_WEIGHT * float(p_norm[i])
                + MMR_WOOD_WEIGHT * float(w_norm[i])
                + MMR_UNIQ_WEIGHT * float(u_norm[i])
            )
        else:
            q_i = float(w_norm[i])
        cc = dict(c)
        cc["_Q"] = float(q_i)
        cc["dominant_tag"] = dominant_tag_for_row(cc)
        _sync_wood_participation(cc)
        pool.append(cc)

    selected: List[Dict] = []
    first = max(pool, key=lambda c: float(c["_Q"]))
    pool.remove(first)
    selected.append(first)

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

    selected.sort(key=lambda x: float(x["hz"]))
    selected_ids = {int(s["id"]) for s in selected}
    rejected = [c for c in candidates if int(c["id"]) not in selected_ids]
    return selected, rejected


def _write_selected_csv(selected: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "id,hz,wood_participation,uniqueness,tag1_ratio,tag3_ratio,p_frac,dominant_tag,"
        "source_target_hz,harvest_filter_policy,harvest_class,Q_mmr"
    ]
    for c in annotate_dominant_tags(selected):
        _sync_wood_participation(c)
        st_hz = c.get("source_target_hz", "")
        st_s = f"{float(st_hz):.6f}" if st_hz not in ("", None) else ""
        pf = c.get("p_frac")
        pf_s = f"{float(pf):.8g}" if pf not in ("", None) else ""
        lines.append(
            f'{int(c["id"])},{float(c["hz"]):.6f},{float(c["wood_participation"]):.6f},'
            f'{float(c["uniqueness"]):.6f},{float(c["tag1_ratio"]):.6f},{float(c["tag3_ratio"]):.6f},'
            f'{pf_s},{c["dominant_tag"]},'
            f'{st_s},{str(c.get("harvest_filter_policy", "") or "")},'
            f'{str(c.get("harvest_class", "") or "")},'
            f'{float(c.get("_Q", 0.0)):.6f}'
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_selection_metadata(
    selected: List[Dict],
    out_path: Path,
    *,
    candidate_count: int,
    pruned_count: int,
    quota_target: int,
    uniqueness_threshold_used: float,
    window_min: Optional[float],
    window_max: Optional[float],
) -> None:
    dom_top = sum(1 for c in selected if c.get("dominant_tag") == DOMINANT_TAG_TOP)
    payload: Dict[str, object] = {
        "selection_strategy": "unified_pool",
        "candidate_count": int(candidate_count),
        "selected_count": len(selected),
        "pruned_by_two_layer": int(pruned_count),
        "quota_target": int(quota_target),
        "dominant_tag_top": int(dom_top),
        "dominant_tag_back": int(len(selected) - dom_top),
        "uniqueness_threshold_used": float(uniqueness_threshold_used),
        "window_min_hz": None if window_min is None else float(window_min),
        "window_max_hz": None if window_max is None else float(window_max),
        "harvest_filter_policy": HARVEST_FILTER_POLICY_VERSION,
        "selected_candidates": [
            {
                "id": int(c["id"]),
                "hz": float(c["hz"]),
                "tag1_ratio": float(c["tag1_ratio"]),
                "tag3_ratio": float(c["tag3_ratio"]),
                "dominant_tag": str(c.get("dominant_tag", dominant_tag_for_row(c))),
                "p_frac": float(c["p_frac"]) if c.get("p_frac") is not None else None,
            }
            for c in sorted(selected, key=lambda x: float(x["hz"]))
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="3D ROM Stage-2: two-layer prune + unified pool (optional MMR cap)"
    )
    parser.add_argument("--candidates", type=Path, default=_default_candidates_path())
    parser.add_argument(
        "--quota",
        type=int,
        default=DEFAULT_QUOTA,
        help="MMR cap (0 = keep all modes after prune)",
    )
    parser.add_argument("--hf-crossover-hz", type=float, default=MODAL_PRUNE_HF_CROSSOVER_HZ)
    parser.add_argument("--df-lf-mf-hz", type=float, default=MODAL_PRUNE_DF_LF_MF_HZ)
    parser.add_argument("--df-hf-hz", type=float, default=MODAL_PRUNE_DF_HF_HZ)
    parser.add_argument("--tag-tol", type=float, default=MODAL_PRUNE_TAG_TOL_DEFAULT)
    parser.add_argument("--log-p-frac-tol", type=float, default=MODAL_PRUNE_LOG_P_FRAC_TOL_DEFAULT)
    parser.add_argument("--no-modal-prune", action="store_true")
    parser.add_argument("--export", type=Path, default=_project_root() / "FEM" / "SORTING" / "selected_modes.csv")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--plot-out", type=Path, default=DEFAULT_SHARED_PLOT_PATH)
    parser.add_argument("--window-min", type=float, default=None)
    parser.add_argument("--window-max", type=float, default=None)
    parser.add_argument("--min-selected", type=int, default=0)
    parser.add_argument("--adaptive-veto", action="store_true")
    parser.add_argument("--adaptive-steps", type=int, default=8)
    parser.add_argument("--uniqueness-floor-min", type=float, default=0.0)
    parser.add_argument("--metadata-out", type=Path, default=_default_metadata_path())
    parser.add_argument("--sigma", type=float, default=SIGMA_HZ)
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=LAMBDA_VAL)
    args = parser.parse_args()

    raw_candidates = _load_candidates(args.candidates)
    candidates = _filter_frequency_window(raw_candidates, args.window_min, args.window_max)
    if not candidates:
        print(f"No valid candidates in: {args.candidates}")
        return 1

    pruned_freq: List[Dict] = []
    if not args.no_modal_prune:
        candidates, pruned_freq = prune_modes_two_layer(
            candidates,
            hf_crossover_hz=float(args.hf_crossover_hz),
            df_lf_mf_hz=float(args.df_lf_mf_hz),
            df_hf_hz=float(args.df_hf_hz),
            tag_tol=float(args.tag_tol),
            log_p_frac_tol=float(args.log_p_frac_tol),
        )
        if pruned_freq:
            print(f"Two-layer prune: removed {len(pruned_freq)} duplicate(s).")
        if not candidates:
            print("No candidates remain after two-layer prune.")
            return 1

    total_quota = int(args.quota)
    sigma_hz = max(1e-6, float(args.sigma))
    lambda_val = min(1.0, max(0.0, float(args.lambda_val)))
    min_selected_target = max(0, int(args.min_selected))
    adaptive_steps = max(1, int(args.adaptive_steps))
    chosen_uniq = float(UNIQUENESS_VETO_MIN)
    min_uniq = max(0.0, float(args.uniqueness_floor_min))

    if args.adaptive_veto:
        best_selected: List[Dict] = []
        best_rejected: List[Dict] = list(candidates)
        for i in range(adaptive_steps + 1):
            t = float(i) / float(adaptive_steps)
            uniq_thr = chosen_uniq - (chosen_uniq - min_uniq) * t
            sel_i, rej_i = unified_mmr_select(
                candidates,
                quota=total_quota,
                uniqueness_min=max(min_uniq, uniq_thr),
                sigma_hz=sigma_hz,
                lambda_val=lambda_val,
            )
            if len(sel_i) > len(best_selected):
                best_selected, best_rejected = sel_i, rej_i
                chosen_uniq = max(min_uniq, uniq_thr)
            if min_selected_target > 0 and len(sel_i) >= min_selected_target:
                best_selected, best_rejected = sel_i, rej_i
                chosen_uniq = max(min_uniq, uniq_thr)
                break
        selected, rejected = best_selected, best_rejected
    else:
        selected, rejected = unified_mmr_select(
            candidates,
            quota=total_quota,
            uniqueness_min=chosen_uniq,
            sigma_hz=sigma_hz,
            lambda_val=lambda_val,
        )

    dom_top = sum(1 for c in selected if c.get("dominant_tag") == DOMINANT_TAG_TOP)
    quota_label = "all" if total_quota <= 0 else str(total_quota)
    print(
        f"Unified pool: selected={len(selected)} (quota={quota_label}), "
        f"dominant_tag Top={dom_top} Back={len(selected) - dom_top}"
    )

    _write_selection_metadata(
        selected,
        args.metadata_out,
        candidate_count=len(raw_candidates),
        pruned_count=len(pruned_freq),
        quota_target=total_quota,
        uniqueness_threshold_used=chosen_uniq,
        window_min=args.window_min,
        window_max=args.window_max,
    )
    _write_selected_csv(selected, args.export)
    print(f"Exported: {args.export.resolve()}")

    plot_dest = resolve_plot_output_path(args.plot_out)
    title = f"unified_pool | selected={len(selected)} pruned={len(pruned_freq)} quota={quota_label}"
    out_png = plot_modal_pool_diagnostics(
        raw_candidates,
        kept=selected,
        pruned=list(pruned_freq) + list(rejected),
        title=title,
        save_path=plot_dest,
        show=not bool(args.headless),
    )
    if args.headless:
        print(f"Saved diagnostic plot: {out_png.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
