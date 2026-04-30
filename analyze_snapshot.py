#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
try:
    from tabulate import tabulate  # type: ignore
except Exception:  # pragma: no cover
    tabulate = None


def _pick_first(data: np.lib.npyio.NpzFile, candidates) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for key in candidates:
        if key in data:
            return data[key], key
    return None, None


def _mac_matrix(phi: np.ndarray) -> np.ndarray:
    # Treat columns as modal vectors; if rows seem to be modes, transpose.
    if phi.ndim != 2:
        raise ValueError(f"Modal matrix must be 2D, got shape={phi.shape}")
    if phi.shape[0] < phi.shape[1]:
        phi = phi.T

    gram = phi.T @ phi
    diag = np.real(np.diag(gram))
    denom = np.outer(diag, diag)
    mac = np.zeros_like(gram, dtype=np.float64)
    valid = denom > 0.0
    mac[valid] = (np.abs(gram[valid]) ** 2) / denom[valid]
    return mac


def _orient_modes(phi: np.ndarray, n_modes: int) -> np.ndarray:
    """Return mode matrix as (n_dofs, n_modes)."""
    if phi.ndim != 2:
        raise ValueError(f"Modal matrix must be 2D, got shape={phi.shape}")
    if phi.shape[1] == n_modes:
        return phi
    if phi.shape[0] == n_modes:
        return phi.T
    raise ValueError(f"Modal matrix shape {phi.shape} does not match frequency count {n_modes}")


def _print_mode_table(sorted_freqs: np.ndarray, mac_sorted: Optional[np.ndarray]) -> None:
    n = int(sorted_freqs.size)
    gaps = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        gaps[1:] = np.diff(sorted_freqs)
        min_gap_idx = int(np.nanargmin(gaps[1:]) + 1)
        max_gap_idx = int(np.nanargmax(gaps[1:]) + 1)
    else:
        min_gap_idx = -1
        max_gap_idx = -1

    headers = ["Index", "Frequency (Hz)", "Gap (Hz)", "Max MAC", "Status"]
    rows = []
    for i in range(n):
        freq = float(sorted_freqs[i])
        if i == 0:
            gap_txt = "-"
        else:
            marker = ""
            if i == min_gap_idx:
                marker = " *MIN*"
            elif i == max_gap_idx:
                marker = " *MAX*"
            gap_txt = f"{float(gaps[i]):.6f}{marker}"

        if mac_sorted is None:
            max_mac = np.nan
            status = "n/a"
            max_mac_txt = "n/a"
        else:
            row = np.asarray(mac_sorted[i], dtype=np.float64).copy()
            row[i] = -np.inf
            max_mac = float(np.max(row)) if row.size > 1 else 0.0
            max_mac_txt = f"{max_mac:.6f}"
            status = "Close" if max_mac > 0.15 else "Unique"

        rows.append([i + 1, f"{freq:.6f}", gap_txt, max_mac_txt, status])

    print("\nPer-mode table (sorted by frequency):")
    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="github", stralign="right", numalign="right"))
    else:
        widths = [len(h) for h in headers]
        for row in rows:
            for j, val in enumerate(row):
                widths[j] = max(widths[j], len(str(val)))
        fmt = " | ".join("{:>" + str(w) + "}" for w in widths)
        print(fmt.format(*headers))
        print("-+-".join("-" * w for w in widths))
        for row in rows:
            print(fmt.format(*row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze FEM snapshot .npz content")
    parser.add_argument("snapshot", type=Path, help="Path to snapshot .npz file")
    args = parser.parse_args()

    if not args.snapshot.exists():
        print(f"[ERROR] File not found: {args.snapshot}")
        return 1

    try:
        data = np.load(args.snapshot, allow_pickle=True)
    except Exception as exc:
        print(f"[ERROR] Failed to load NPZ: {exc}")
        return 1

    sorted_freqs = None
    mac_sorted = None
    order = None

    try:
        freqs, fkey = _pick_first(data, ["eigenfrequencies_hz", "freqs_hz", "frequencies_hz"])
        if freqs is None:
            raise KeyError("Missing eigenfrequency array (tried: eigenfrequencies_hz, freqs_hz, frequencies_hz)")
        freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
        if freqs.size == 0:
            raise ValueError("Frequency array is empty")

        order = np.argsort(freqs)
        sorted_freqs = freqs[order]
        print(f"file: {args.snapshot}")
        print(f"frequency_key: {fkey}")
        print(f"mode_count: {int(sorted_freqs.size)}")
        print(f"freq_range_hz: {float(np.min(sorted_freqs)):.6f} .. {float(np.max(sorted_freqs)):.6f}")

        if sorted_freqs.size > 1:
            gaps = np.diff(sorted_freqs)
            print(f"avg_gap_hz: {float(np.mean(gaps)):.6f}")
            print(f"min_gap_hz: {float(np.min(gaps)):.6f}")
            print(f"max_gap_hz: {float(np.max(gaps)):.6f}")
        else:
            print("avg_gap_hz: n/a (only one mode)")
    except Exception as exc:
        print(f"[ERROR] Frequency analysis failed: {exc}")

    try:
        phi, pkey = _pick_first(data, ["Phi", "modes", "eigvecs_real", "eigvecs"])
        if phi is None:
            raise KeyError("Missing modal matrix (tried: Phi, modes, eigvecs_real, eigvecs)")
        if sorted_freqs is None or order is None:
            raise RuntimeError("Cannot compute MAC without valid sorted frequencies")

        phi = np.asarray(phi, dtype=np.float64)
        phi_oriented = _orient_modes(phi, int(sorted_freqs.size))
        phi_sorted = phi_oriented[:, order]
        mac_sorted = _mac_matrix(phi_sorted)
        n = mac_sorted.shape[0]
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag_vals = mac_sorted[off_diag_mask]
        avg_off_diag = float(np.mean(off_diag_vals)) if off_diag_vals.size else 0.0
        print(f"modal_matrix_key: {pkey}")
        print(f"modal_matrix_shape: {tuple(phi.shape)}")
        print(f"mac_avg_offdiag: {avg_off_diag:.6f}")
    except Exception as exc:
        print(f"[WARN] MAC analysis skipped: {exc}")

    if sorted_freqs is not None:
        _print_mode_table(sorted_freqs, mac_sorted)

    try:
        runtime, rkey = _pick_first(data, ["total_runtime", "elapsed_s", "runtime_s"])
        if runtime is None:
            raise KeyError("No runtime metadata key found")
        rt = float(np.asarray(runtime).reshape(-1)[0])
        print(f"runtime_seconds ({rkey}): {rt:.6f}")
    except Exception as exc:
        print(f"[WARN] Runtime metadata unavailable: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
