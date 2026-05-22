#!/usr/bin/env python3
"""
ROM post-processing: two-layer modal de-duplication, dominant-tag labeling,
terminal diagnostics (no Streamlit), and σ-retry limit helpers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Legacy single Δf (Hz); prefer two-layer prune below.
MODAL_PRUNE_DF_HZ_DEFAULT = 0.5

MODAL_PRUNE_HF_CROSSOVER_HZ = 350.0
MODAL_PRUNE_DF_LF_MF_HZ = 1.5
MODAL_PRUNE_DF_HF_HZ = 0.75
MODAL_PRUNE_TAG_TOL_DEFAULT = 0.08
MODAL_PRUNE_LOG_P_FRAC_TOL_DEFAULT = 1.0

DOMINANT_TAG_TOP = "Top"
DOMINANT_TAG_BACK = "Back"

# Per scheduler shift: primary + σ-retry workers (fail-fast).
SIGMA_RETRY_MAX_ATTEMPTS_PER_SHIFT_DEFAULT = 5
ST_SIGMA_LADDER_MAX_DEFAULT = 6


def _as_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def dominant_tag_for_row(row: Mapping[str, Any]) -> str:
    """
    Visualization label only: which plate carries more modal energy on the shell.

    Does not affect ROM inclusion — every mode keeps its full coupled displacement vector.
    """
    t1 = _as_float(row, "tag1_ratio")
    t3 = _as_float(row, "tag3_ratio")
    return DOMINANT_TAG_TOP if t1 >= t3 else DOMINANT_TAG_BACK


def annotate_dominant_tags(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return shallow copies with ``dominant_tag`` set (Top / Back)."""
    out: List[Dict[str, Any]] = []
    for row in candidates:
        cc = dict(row)
        cc["dominant_tag"] = dominant_tag_for_row(cc)
        out.append(cc)
    return out


def df_hz_threshold_for_frequency(
    f_hz: float,
    *,
    hf_crossover_hz: float = MODAL_PRUNE_HF_CROSSOVER_HZ,
    df_lf_mf_hz: float = MODAL_PRUNE_DF_LF_MF_HZ,
    df_hf_hz: float = MODAL_PRUNE_DF_HF_HZ,
) -> float:
    """Band-dependent frequency merge tolerance (Hz)."""
    if float(f_hz) >= float(hf_crossover_hz):
        return max(1.0e-9, float(df_hf_hz))
    return max(1.0e-9, float(df_lf_mf_hz))


def pair_df_threshold_hz(f_a: float, f_b: float, **kwargs: Any) -> float:
    """Use the looser of the two band thresholds when frequencies straddle crossover."""
    ta = df_hz_threshold_for_frequency(f_a, **kwargs)
    tb = df_hz_threshold_for_frequency(f_b, **kwargs)
    return max(ta, tb)


def modes_physically_similar(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    tag_tol: float = MODAL_PRUNE_TAG_TOL_DEFAULT,
    log_p_frac_tol: float = MODAL_PRUNE_LOG_P_FRAC_TOL_DEFAULT,
) -> bool:
    """
    Layer-2 similarity: comparable top/back energy split and pressure participation.

    Modes with very different ``p_frac`` (orders of magnitude) are never merged — e.g. a
    strongly coupled mode at σ must not collapse onto a wood-only neighbor.
    """
    t1a, t3a = _as_float(a, "tag1_ratio"), _as_float(a, "tag3_ratio")
    t1b, t3b = _as_float(b, "tag1_ratio"), _as_float(b, "tag3_ratio")
    if abs(t1a - t1b) > float(tag_tol) or abs(t3a - t3b) > float(tag_tol):
        return False

    pa = max(_as_float(a, "p_frac"), 1.0e-30)
    pb = max(_as_float(b, "p_frac"), 1.0e-30)
    log_gap = abs(math.log10(pa) - math.log10(pb))
    if log_gap > float(log_p_frac_tol):
        return False
    return True


def _mode_merit_p_frac(row: Mapping[str, Any]) -> Tuple[float, ...]:
    """Higher is better: primary ``p_frac``, then wood, then uniqueness."""
    return (
        _as_float(row, "p_frac"),
        _as_float(row, "wood_participation"),
        _as_float(row, "uniqueness"),
    )


def prune_modes_two_layer(
    candidates: Sequence[Dict[str, Any]],
    *,
    hf_crossover_hz: float = MODAL_PRUNE_HF_CROSSOVER_HZ,
    df_lf_mf_hz: float = MODAL_PRUNE_DF_LF_MF_HZ,
    df_hf_hz: float = MODAL_PRUNE_DF_HF_HZ,
    tag_tol: float = MODAL_PRUNE_TAG_TOL_DEFAULT,
    log_p_frac_tol: float = MODAL_PRUNE_LOG_P_FRAC_TOL_DEFAULT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Two-layer de-duplication on a sorted frequency axis.

    Layer 1: chain modes while ``Δf`` is below the band-dependent threshold.
    Layer 2: within each frequency chain, merge physics-similar modes; keep highest ``p_frac``.
    """
    pool = sorted(candidates, key=lambda c: _as_float(c, "hz"))
    if not pool:
        return [], []

    df_kwargs = {
        "hf_crossover_hz": hf_crossover_hz,
        "df_lf_mf_hz": df_lf_mf_hz,
        "df_hf_hz": df_hf_hz,
    }

    freq_clusters: List[List[Dict[str, Any]]] = []
    for row in pool:
        f_hz = _as_float(row, "hz")
        if freq_clusters:
            f_last = _as_float(freq_clusters[-1][-1], "hz")
            if f_hz - f_last < pair_df_threshold_hz(f_last, f_hz, **df_kwargs):
                freq_clusters[-1].append(row)
                continue
        freq_clusters.append([row])

    kept: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []

    for chain in freq_clusters:
        survivors = list(chain)
        changed = True
        while changed and len(survivors) > 1:
            changed = False
            used = [False] * len(survivors)
            merged_into: List[Dict[str, Any]] = []
            for i in range(len(survivors)):
                if used[i]:
                    continue
                group = [survivors[i]]
                used[i] = True
                for j in range(i + 1, len(survivors)):
                    if used[j]:
                        continue
                    if modes_physically_similar(
                        survivors[i],
                        survivors[j],
                        tag_tol=tag_tol,
                        log_p_frac_tol=log_p_frac_tol,
                    ):
                        group.append(survivors[j])
                        used[j] = True
                        changed = True
                best = max(group, key=lambda r: _mode_merit_p_frac(r))
                merged_into.append(best)
                for row in group:
                    if row is not best:
                        pruned.append(row)
            survivors = merged_into

        if len(survivors) == 1:
            kept.append(survivors[0])
            continue

        # Residual chain without physics match: keep best p_frac only.
        best = max(survivors, key=lambda r: _mode_merit_p_frac(r))
        kept.append(best)
        for row in survivors:
            if row is not best:
                pruned.append(row)

    kept = annotate_dominant_tags(kept)
    return kept, pruned


def prune_near_duplicate_modes(
    candidates: Sequence[Dict[str, Any]],
    *,
    df_hz: float = MODAL_PRUNE_DF_HZ_DEFAULT,
    merit_prefer: Sequence[str] = ("p_frac", "wood_participation"),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Backward-compatible wrapper: uniform ``df_hz`` maps to LF/MF and HF layer-1 thresholds.
    """
    _ = merit_prefer
    return prune_modes_two_layer(
        candidates,
        df_lf_mf_hz=float(df_hz),
        df_hf_hz=float(df_hz),
    )


def sigma_retry_max_attempts_per_shift(
    solver_cfg: Optional[Mapping[str, Any]] = None,
) -> int:
    """Max ST solve attempts per scheduler shift (primary + σ-retry workers)."""
    default = int(SIGMA_RETRY_MAX_ATTEMPTS_PER_SHIFT_DEFAULT)
    if not solver_cfg:
        return default
    try:
        v = int(solver_cfg.get("eps_sigma_retry_max_per_shift", default))
    except (TypeError, ValueError):
        return default
    return max(1, min(12, v))


def st_sigma_ladder_max(solver_cfg: Optional[Mapping[str, Any]] = None) -> int:
    """Max σ values tried inside one worker EPS ST setup (LU fail-fast)."""
    default = int(ST_SIGMA_LADDER_MAX_DEFAULT)
    if not solver_cfg:
        return default
    try:
        v = int(solver_cfg.get("eps_st_sigma_ladder_max", default))
    except (TypeError, ValueError):
        return default
    return max(1, min(16, v))


def _load_candidates_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        raw = list(data.get("candidates") or [])
    elif isinstance(data, list):
        raw = list(data)
    else:
        raise ValueError("Expected dict with 'candidates' or a list.")
    out: List[Dict[str, Any]] = []
    for c in raw:
        try:
            row = dict(c)
            row["hz"] = float(c.get("hz"))
            row["tag1_ratio"] = float(c.get("tag1_ratio", 0.0) or 0.0)
            row["tag3_ratio"] = float(c.get("tag3_ratio", 0.0) or 0.0)
            row["wood_participation"] = float(
                c.get("wood_participation", row["tag1_ratio"] + row["tag3_ratio"])
            )
            row["uniqueness"] = float(c.get("uniqueness", 0.0) or 0.0)
            if c.get("p_frac") is not None:
                row["p_frac"] = float(c.get("p_frac"))
            out.append(row)
        except (TypeError, ValueError, KeyError):
            continue
    return out


def plot_modal_pool_diagnostics(
    candidates: Sequence[Dict[str, Any]],
    *,
    kept: Optional[Sequence[Dict[str, Any]]] = None,
    pruned: Optional[Sequence[Dict[str, Any]]] = None,
    title: str = "Modal pool (unified axis)",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Terminal diagnostic: frequency vs tag1 ratio, colored by ``dominant_tag``.

    Optional second panel: ``p_frac`` vs frequency (log scale) for coupling audit.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = annotate_dominant_tags(list(candidates))
    kept_set = {int(r["id"]) for r in (kept or []) if "id" in r}
    pruned_set = {int(r["id"]) for r in (pruned or []) if "id" in r}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, ykey, ylabel in (
        (axes[0], "tag1_ratio", "Top plate energy ratio (tag 1)"),
        (axes[1], "p_frac", "Pressure fraction (log scale)"),
    ):
        for tag, color, marker in (
            (DOMINANT_TAG_TOP, "#2e7d32", "o"),
            (DOMINANT_TAG_BACK, "#1565c0", "^"),
        ):
            subset = [r for r in all_rows if r.get("dominant_tag") == tag]
            if not subset:
                continue
            xs = [_as_float(r, "hz") for r in subset]
            ys = [_as_float(r, ykey) for r in subset]
            if ykey == "p_frac":
                ys = [max(y, 1.0e-20) for y in ys]
            ax.scatter(xs, ys, c=color, marker=marker, s=42, alpha=0.85, label=tag)

        if kept is not None:
            sel = [r for r in all_rows if int(r.get("id", -1)) in kept_set]
            if sel:
                ax.scatter(
                    [_as_float(r, "hz") for r in sel],
                    [
                        max(_as_float(r, ykey), 1.0e-20 if ykey == "p_frac" else 0.0)
                        for r in sel
                    ],
                    s=120,
                    facecolors="none",
                    edgecolors="#000000",
                    linewidths=1.2,
                    label="Kept",
                )

        if pruned is not None:
            rej = [r for r in all_rows if int(r.get("id", -1)) in pruned_set]
            if rej:
                ax.scatter(
                    [_as_float(r, "hz") for r in rej],
                    [
                        max(_as_float(r, ykey), 1.0e-20 if ykey == "p_frac" else 0.0)
                        for r in rej
                    ],
                    s=30,
                    c="#c62828",
                    alpha=0.35,
                    marker="x",
                    label="Pruned",
                )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        if ykey == "p_frac":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    n_in = len(all_rows)
    n_k = len(kept or [])
    n_p = len(pruned or [])
    fig.suptitle(f"{title} | input={n_in} kept={n_k} pruned={n_p}")
    plt.tight_layout()

    out = Path(save_path or "modal_pool_diagnostics.png").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    if show:
        plt.show()
    return out


def _cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Modal pool diagnostics and two-layer pruning (terminal plots)."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="candidates_log.json (or list JSON).",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("modal_pool_diagnostics.png"),
        help="PNG output path.",
    )
    parser.add_argument("--hf-crossover-hz", type=float, default=MODAL_PRUNE_HF_CROSSOVER_HZ)
    parser.add_argument("--df-lf-mf-hz", type=float, default=MODAL_PRUNE_DF_LF_MF_HZ)
    parser.add_argument("--df-hf-hz", type=float, default=MODAL_PRUNE_DF_HF_HZ)
    parser.add_argument("--tag-tol", type=float, default=MODAL_PRUNE_TAG_TOL_DEFAULT)
    parser.add_argument("--log-p-frac-tol", type=float, default=MODAL_PRUNE_LOG_P_FRAC_TOL_DEFAULT)
    parser.add_argument("--show", action="store_true", help="Also open interactive window.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    raw = _load_candidates_json(args.candidates.resolve())
    if not raw:
        print(f"No candidates in {args.candidates}", file=sys.stderr)
        return 1

    kept, pruned = prune_modes_two_layer(
        raw,
        hf_crossover_hz=args.hf_crossover_hz,
        df_lf_mf_hz=args.df_lf_mf_hz,
        df_hf_hz=args.df_hf_hz,
        tag_tol=args.tag_tol,
        log_p_frac_tol=args.log_p_frac_tol,
    )
    out = plot_modal_pool_diagnostics(
        raw,
        kept=kept,
        pruned=pruned,
        title="Modal pool after two-layer prune",
        save_path=args.plot_out,
        show=bool(args.show),
    )
    print(f"Candidates: {len(raw)} -> kept {len(kept)}, pruned {len(pruned)}")
    print(f"Diagnostic plot: {out}")
    top_n = sum(1 for r in kept if r.get("dominant_tag") == DOMINANT_TAG_TOP)
    back_n = len(kept) - top_n
    print(f"Dominant-tag mix (kept): Top={top_n} Back={back_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
