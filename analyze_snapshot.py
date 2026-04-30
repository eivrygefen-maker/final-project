#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


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

    try:
        freqs, fkey = _pick_first(data, ["eigenfrequencies_hz", "freqs_hz", "frequencies_hz"])
        if freqs is None:
            raise KeyError("Missing eigenfrequency array (tried: eigenfrequencies_hz, freqs_hz, frequencies_hz)")
        freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
        if freqs.size == 0:
            raise ValueError("Frequency array is empty")

        print(f"file: {args.snapshot}")
        print(f"frequency_key: {fkey}")
        print(f"mode_count: {int(freqs.size)}")
        print(f"freq_range_hz: {float(np.min(freqs)):.6f} .. {float(np.max(freqs)):.6f}")

        if freqs.size > 1:
            gaps = np.diff(np.sort(freqs))
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
        phi = np.asarray(phi, dtype=np.float64)
        mac = _mac_matrix(phi)
        n = mac.shape[0]
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag_vals = mac[off_diag_mask]
        avg_off_diag = float(np.mean(off_diag_vals)) if off_diag_vals.size else 0.0
        print(f"modal_matrix_key: {pkey}")
        print(f"modal_matrix_shape: {tuple(phi.shape)}")
        print(f"mac_avg_offdiag: {avg_off_diag:.6f}")
    except Exception as exc:
        print(f"[WARN] MAC analysis skipped: {exc}")

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
