#!/usr/bin/env python3
"""
Detailed mode table from a single ROM snapshot .npz (freqs_hz, sifter_stats_json, elapsed_s).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _load_sifter_stats(data: np.lib.npyio.NpzFile) -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse sifter_stats_json into a dict; return (stats, error_message_or_none)."""
    if "sifter_stats_json" not in data:
        return {}, "no `sifter_stats_json` key in NPZ"
    raw = data["sifter_stats_json"]
    try:
        if isinstance(raw, np.ndarray):
            if raw.shape == ():
                s = raw.item()
            else:
                s = raw.reshape(-1)[0]
        else:
            s = raw
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        if not s:
            return {}, "empty `sifter_stats_json`"
        return json.loads(s), None
    except Exception as exc:
        return {}, f"failed to parse sifter_stats_json: {exc}"


def _pick_per_mode_arrays(
    stats: Dict[str, Any], n_modes: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Look for per-mode uniqueness / participation lists aligned with accepted modes.
    Returns (uniqueness, participation, notes).
    """
    notes: List[str] = []
    uniq: Optional[np.ndarray] = None
    part: Optional[np.ndarray] = None

    for key in (
        "uniqueness_scores",
        "mode_uniqueness",
        "structural_uniqueness",
        "accepted_uniqueness_scores",
    ):
        if key in stats and stats[key] is not None:
            try:
                u = np.asarray(stats[key], dtype=np.float64).reshape(-1)
                if u.size == n_modes:
                    uniq = u
                    notes.append(f"uniqueness from `{key}`")
                elif u.size > 0:
                    notes.append(f"`{key}` length {u.size} != mode count {n_modes} (ignored)")
            except Exception:
                notes.append(f"`{key}` present but not numeric array (ignored)")
            break

    for key in (
        "participation_ratios",
        "wood_participation",
        "wood_participation_ratios",
        "accepted_participation_ratios",
    ):
        if key in stats and stats[key] is not None:
            try:
                p = np.asarray(stats[key], dtype=np.float64).reshape(-1)
                if p.size == n_modes:
                    part = p
                    notes.append(f"participation from `{key}`")
                elif p.size > 0:
                    notes.append(f"`{key}` length {p.size} != mode count {n_modes} (ignored)")
            except Exception:
                notes.append(f"`{key}` present but not numeric array (ignored)")
            break

    return uniq, part, notes


def _stats_cell(
    row_idx: int,
    uniq_sorted: Optional[np.ndarray],
    part_sorted: Optional[np.ndarray],
) -> str:
    if uniq_sorted is None and part_sorted is None:
        return "N/A"
    parts: List[str] = []
    if uniq_sorted is not None:
        parts.append(f"uniq={uniq_sorted[row_idx]:.4f}")
    if part_sorted is not None:
        parts.append(f"wood={part_sorted[row_idx]:.4f}")
    return "; ".join(parts)


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    h = "| " + " | ".join(esc(x) for x in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(esc(c) for c in r) + " |" for r in rows)
    return "\n".join([h, sep, body])


def _elapsed_seconds(data: np.lib.npyio.NpzFile) -> Optional[float]:
    if "elapsed_s" not in data:
        return None
    try:
        return float(np.asarray(data["elapsed_s"]).reshape(-1)[0])
    except Exception:
        return None


def analyze_snapshot(path: Path) -> str:
    lines: List[str] = []
    lines.append(f"## Snapshot: `{path.name}`\n")

    if not path.is_file():
        return lines[0] + f"\n**Error:** file not found: `{path}`\n"

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        return lines[0] + f"\n**Error:** could not load NPZ: {exc}\n"

    try:
        if "freqs_hz" not in data:
            data.close()
            return lines[0] + "\n**Error:** missing required key `freqs_hz`.\n"
        freqs = np.asarray(data["freqs_hz"], dtype=np.float64).reshape(-1)
    except Exception as exc:
        data.close()
        return lines[0] + f"\n**Error:** reading `freqs_hz`: {exc}\n"

    n = int(freqs.size)
    if n == 0:
        data.close()
        return lines[0] + "\n*Empty `freqs_hz`.*\n"

    stats, stats_err = _load_sifter_stats(data)
    if stats_err:
        lines.append(f"*Warning:* {stats_err}\n")
    uniq_raw, part_raw, pm_notes = _pick_per_mode_arrays(stats, n)
    for note in pm_notes:
        lines.append(f"*Note:* {note}\n")

    order = np.argsort(freqs)
    f_sorted = freqs[order]
    uniq_sorted = uniq_raw[order] if uniq_raw is not None else None
    part_sorted = part_raw[order] if part_raw is not None else None

    if stats and uniq_raw is None and part_raw is None:
        lines.append(
            "*Warning:* `sifter_stats_json` has no per-mode `uniqueness_scores` / "
            "`participation_ratios` arrays (only run-level counters). "
            "Per-row column shows **N/A** until the solver saves per-mode stats.\n"
        )

    gaps = np.zeros(n, dtype=np.float64)
    rel_pct = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        gaps[i] = f_sorted[i] - f_sorted[i - 1]
        if f_sorted[i - 1] > 0.0:
            rel_pct[i] = 100.0 * gaps[i] / f_sorted[i - 1]

    gap_values = gaps[1:] if n > 1 else np.array([], dtype=np.float64)
    avg_gap = float(np.mean(gap_values)) if gap_values.size else float("nan")

    headers = [
        "Mode",
        "Frequency (Hz)",
        "Gap from prev (Hz)",
        "Rel. gap (%)",
        "Uniqueness / stats",
    ]
    table_rows: List[List[str]] = []
    for i in range(n):
        rg = rel_pct[i]
        table_rows.append(
            [
                str(i + 1),
                f"{f_sorted[i]:.6f}",
                "—" if i == 0 else f"{gaps[i]:.6f}",
                "—" if i == 0 or not np.isfinite(rg) else f"{rg:.4f}",
                _stats_cell(i, uniq_sorted, part_sorted),
            ]
        )

    lines.append(_markdown_table(headers, table_rows))
    lines.append("\n\n### Summary\n\n")
    lines.append(f"- **Mode count:** {n}\n")
    lines.append(f"- **Average gap:** {avg_gap:.6f} Hz\n" if np.isfinite(avg_gap) else "- **Average gap:** n/a\n")

    elapsed = _elapsed_seconds(data)
    if elapsed is not None:
        lines.append(f"- **Elapsed (elapsed_s):** {elapsed:.6f} s\n")
    else:
        lines.append("- **Elapsed (elapsed_s):** N/A\n")

    if stats:
        lines.append("\n### Raw sifter_stats_json (keys)\n\n")
        lines.append(f"`{', '.join(sorted(stats.keys()))}`\n")

    data.close()
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Markdown mode table from one snapshot NPZ (freqs_hz, sifter_stats_json, elapsed_s).",
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Path to snapshot .npz",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write Markdown to this file (default: stdout)",
    )
    args = parser.parse_args()

    text = analyze_snapshot(args.npz_path.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
