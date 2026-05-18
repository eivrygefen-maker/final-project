#!/usr/bin/env python3
"""
Regenerate MMR-style selection plots (Frequency vs. wood participation) from packaged
``snapshot_XXXX.npz`` files when the corresponding PNGs are missing.

The pipeline (``package_rom``) stores **MMR-selected** modes only: ``frequencies`` and
``wood_participations``. The live tuner plot also shows **rejected** candidates (red);
that set is not stored in the NPZ, so regenerated figures show **green selected points
only**—same marker style, axis labels, and colors as ``dynamic_filter_tuner._plot_selection``.

Some older FOM snapshots use ``freqs_hz`` + ``participation_ratios``. Legacy ROM snapshots may
have ``freqs_hz`` + ``sifter_stats_json`` only (no participation array); wood values are then
recovered by parsing the JSON for mode records with ``wood_participation``, ``participation``,
or ``tag1_ratio`` + ``tag3_ratio``, paired to frequencies by nearest-Hz matching.

Example::

    py FEM/scripts/reproduce_plots.py --start 1 --end 12
    py FEM/scripts/reproduce_plots.py --start 1 --end 19 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paths import DEFAULT_SHAPE_NAME, shared_plot_path


def _repo_root() -> Path:
    repo = Path(os.path.abspath(__file__)).resolve()
    while repo.name != "final-project" and repo.parent != repo:
        repo = repo.parent
    if repo.name != "final-project":
        raise RuntimeError(
            "Could not locate parent directory named 'final-project' starting from "
            f"{Path(__file__).resolve()}"
        )
    return repo


REPO_ROOT = _repo_root()

from dynamic_filter_tuner import (  # noqa: E402
    LAMBDA_VAL,
    SIGMA_HZ,
    U,
    UNIQUENESS_VETO_MIN,
    W,
    WOOD_FILTER_MIN,
    _plot_selection,
)


def _npz_utf8_str(z: Any, key: str) -> str:
    raw = z[key]
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            raw = raw.item()
        elif raw.dtype.kind in ("S", "U"):
            raw = raw.astype("U").item() if raw.size == 1 else raw.tobytes().decode("utf-8", errors="replace")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _mode_hz(m: Dict[str, Any]) -> float | None:
    for k in ("hz", "freq", "frequency", "f_hz", "f"):
        if k not in m:
            continue
        try:
            v = float(m[k])
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            continue
    return None


def _wood_from_mode_rec(m: Dict[str, Any]) -> float | None:
    if "wood_participation" in m:
        try:
            return float(m["wood_participation"])
        except (TypeError, ValueError):
            pass
    if "participation" in m:
        try:
            return float(m["participation"])
        except (TypeError, ValueError):
            pass
    t1, t3 = m.get("tag1_ratio"), m.get("tag3_ratio")
    if t1 is not None and t3 is not None:
        try:
            return max(0.0, float(t1) + float(t3))
        except (TypeError, ValueError):
            pass
    return None


def _collect_mode_dicts(obj: Any, *, _depth: int = 0) -> List[Dict[str, Any]]:
    """Depth-first collect dicts that have both a frequency and a wood metric."""
    if _depth > 16:
        return []
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        h = _mode_hz(obj)
        w = _wood_from_mode_rec(obj)
        if h is not None and w is not None:
            out.append(obj)
        for v in obj.values():
            out.extend(_collect_mode_dicts(v, _depth=_depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_mode_dicts(v, _depth=_depth + 1))
    return out


def _align_wood_to_freqs(freqs: np.ndarray, modes: List[Dict[str, Any]]) -> Tuple[np.ndarray, str]:
    """One wood value per freq row: nearest neighbor in (mode_hz, wood) space."""
    pairs: List[Tuple[float, float]] = []
    for m in modes:
        h = _mode_hz(m)
        w = _wood_from_mode_rec(m)
        if h is None or w is None:
            continue
        pairs.append((float(h), float(w)))
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    n = int(freqs.size)
    if not pairs:
        return np.zeros(n, dtype=np.float64), "sifter JSON: no mode rows with hz + wood; filled zeros"

    notes: List[str] = []
    out = np.zeros(n, dtype=np.float64)
    for i, f in enumerate(freqs):
        j = int(np.argmin([abs(p[0] - f) for p in pairs]))
        d = abs(pairs[j][0] - f)
        out[i] = pairs[j][1]
        if d > max(1.0, 1e-6 * abs(f)):
            notes.append(f"f[{i}]={f:.2f} nearestΔ={d:.2f}Hz")

    msg_parts: List[str] = []
    if len(pairs) != n:
        msg_parts.append(f"freq len={n} vs mode rows={len(pairs)}")
    if notes:
        msg_parts.append("large Δ: " + "; ".join(notes[:6]) + (" …" if len(notes) > 6 else ""))
    return out, " | ".join(msg_parts)


def _load_snapshot_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Returns (hz, wood, format_tag, warn_message). warn_message may be empty."""
    warn = ""
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if "frequencies" in files and "wood_participations" in files:
            hz = np.asarray(z["frequencies"], dtype=np.float64).ravel()
            wood = np.asarray(z["wood_participations"], dtype=np.float64).ravel()
            return hz, wood, "pipeline_package_rom", warn
        if "freqs_hz" in files and "participation_ratios" in files:
            hz = np.asarray(z["freqs_hz"], dtype=np.float64).ravel()
            wood = np.asarray(z["participation_ratios"], dtype=np.float64).ravel()
            return hz, wood, "rom_fom", warn
        if "freqs_hz" in files and "sifter_stats_json" in files and "participation_ratios" not in files:
            hz = np.asarray(z["freqs_hz"], dtype=np.float64).ravel()
            try:
                raw_js = _npz_utf8_str(z, "sifter_stats_json")
                parsed = json.loads(raw_js)
            except (json.JSONDecodeError, UnicodeError, KeyError) as exc:
                raise ValueError(f"sifter_stats_json: could not parse: {exc}") from exc
            modes = _collect_mode_dicts(parsed)
            wood, wpart = _align_wood_to_freqs(hz, modes)
            warn = wpart
            return hz, wood, "legacy_sifter_json", warn
    raise ValueError(
        f"{path.name}: unsupported keys (need pipeline ROM, rom_fom pairs, or "
        f"freqs_hz + sifter_stats_json without participation_ratios). Got: {sorted(files)}"
    )


def _to_selected_candidates(hz: np.ndarray, wood: np.ndarray) -> List[Dict[str, Any]]:
    if hz.shape != wood.shape:
        raise ValueError(f"hz shape {hz.shape} != wood shape {wood.shape}")
    out: List[Dict[str, Any]] = []
    for i in range(int(hz.size)):
        out.append(
            {
                "id": i + 1,
                "hz": float(hz[i]),
                "wood_participation": float(wood[i]),
                "uniqueness": 1.0,
                "tag1_ratio": 0.0,
                "tag3_ratio": 0.0,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate snapshot_XXXX.png from snapshot NPZ files.")
    ap.add_argument("--start", type=int, default=1, help="First snapshot index (default 1).")
    ap.add_argument("--end", type=int, default=12, help="Last snapshot index inclusive (default 12).")
    ap.add_argument(
        "--snapshots-dir",
        type=Path,
        default=REPO_ROOT / "ROM" / "classic" / "snapshots",
        help="Directory containing snapshot_XXXX.npz",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for snapshot_XXXX.png "
            f"(default: shared host {shared_plot_path('snapshot_0001.png', DEFAULT_SHAPE_NAME).parent.as_posix()})"
        ),
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="If the target snapshot_XXXX.png already exists, do not overwrite it.",
    )
    args = ap.parse_args()

    snapshots_dir = args.snapshots_dir.resolve()
    plots_dir = (
        args.plots_dir.resolve()
        if args.plots_dir is not None
        else shared_plot_path("snapshot_0001.png", shape_name=DEFAULT_SHAPE_NAME).parent
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    lo = int(args.start)
    hi = int(args.end)
    if hi < lo:
        print("error: --end must be >= --start", file=sys.stderr)
        return 2

    for idx in range(lo, hi + 1):
        snap_name = f"snapshot_{idx:04d}.npz"
        plot_name = f"snapshot_{idx:04d}.png"
        npz_path = snapshots_dir / snap_name
        out_path = plots_dir / plot_name

        if args.skip_existing and out_path.is_file():
            print(f"[skip] {plot_name} already exists")
            continue

        if not npz_path.is_file():
            print(f"[skip] missing {npz_path}")
            continue

        try:
            hz, wood, fmt, warn = _load_snapshot_arrays(npz_path)
        except (OSError, ValueError) as exc:
            print(f"[error] {npz_path}: {exc}", file=sys.stderr)
            return 1

        if warn:
            print(f"[warn] {npz_path.name}: {warn}")

        selected = _to_selected_candidates(hz, wood)
        rejected: List[Dict[str, Any]] = []

        title = (
            f"MMR tuner | selected={len(selected)} rejected=0 | "
            f"W={W}, U={U}, λ={LAMBDA_VAL}, σ={SIGMA_HZ} Hz | "
            f"vetoes: wood≥{WOOD_FILTER_MIN}, uniqueness≥{UNIQUENESS_VETO_MIN} | "
            f"{plot_name} (regenerated from {snap_name}, {fmt})"
        )

        _plot_selection(selected, rejected, title, headless=True, save_path=out_path)
        print(f"[ok] {out_path}  (N={len(selected)}, {fmt})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
