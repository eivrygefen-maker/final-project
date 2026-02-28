import json
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

# =========================
# EDIT ONLY THESE
# =========================
WAV_PATH = Path("/media/sf_gmar/guitar_audio/spruce_C4_cccc.wav")
FEM_JSON_PATH = Path("FEM/outputs/rect_plate_spruce_cccc_result.json")

START_S = 0.25   # analyze after attack
END_S   = 2.00

MIN_HZ = 20
MAX_HZ = 5000

TOL_PCT = 0.10   # ±10% comparison tolerance

# "Presence" threshold relative to noise floor (lower = detects more, but may include junk)
SNR_DB = 3.0     # 3 dB above median noise floor is VERY permissive

# Peak spacing (to avoid printing thousands of nearly-identical bins)
MIN_SEP_HZ = 5.0

# Optional: limit how many FEM modes to check
MAX_MODES = 20


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def load_modes_hz(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict) and "modes_hz" in data["result"]:
        return np.array(data["result"]["modes_hz"], dtype=float)
    if isinstance(data, dict) and "modes_hz" in data:
        return np.array(data["modes_hz"], dtype=float)
    raise ValueError("Could not find modes_hz in FEM JSON (expected data['result']['modes_hz'] or data['modes_hz']).")


def main():
    if not WAV_PATH.exists():
        raise FileNotFoundError(f"WAV not found: {WAV_PATH}")
    if not FEM_JSON_PATH.exists():
        raise FileNotFoundError(f"FEM JSON not found: {FEM_JSON_PATH}")

    modes = load_modes_hz(FEM_JSON_PATH)[:MAX_MODES]

    sr, x = wavfile.read(WAV_PATH)

    # mono + float normalize
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    mx = np.max(np.abs(x))
    if mx > 0:
        x /= mx

    i0 = int(max(0.0, START_S) * sr)
    i1 = int(min(END_S, len(x) / sr) * sr)
    if i1 <= i0 + 4096:
        raise ValueError("Window too short. Increase END_S or reduce START_S.")

    seg = x[i0:i1] * np.hanning(i1 - i0)

    nfft = next_pow2(len(seg))
    X = np.fft.rfft(seg, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

    mag = np.abs(X) + 1e-20
    df = freqs[1] - freqs[0]

    # band-limit
    band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    freqs_b = freqs[band]
    mag_b = mag[band]

    # convert to dB for robust thresholding
    mag_db = 20.0 * np.log10(mag_b)

    # noise floor estimate (median)
    floor_db = float(np.median(mag_db))
    thr_db = floor_db + SNR_DB

    # peak finding: permissive (we're trying to capture even weak components)
    min_dist_bins = max(1, int(round(MIN_SEP_HZ / df)))
    peaks, props = find_peaks(mag_db, height=thr_db, distance=min_dist_bins)

    detected = freqs_b[peaks]
    detected = np.unique(np.round(detected, 3))  # small cleanup
    detected.sort()

    print(f"\nWAV: {WAV_PATH}")
    print(f"Window: {START_S:.2f}..{END_S:.2f}s | df≈{df:.3f} Hz")
    print(f"Detection threshold: {thr_db:.2f} dB  (floor≈{floor_db:.2f} dB, +{SNR_DB:.1f} dB)")
    print(f"Detected freqs count: {len(detected)}\n")

    print("Detected frequencies (Hz), low -> high:")
    print(", ".join(f"{f:.3f}" for f in detected))

    # -------------------------
    # Compare to FEM modes (±10%)
    # -------------------------
    print("\n\nCompare to FEM modes (±10%):")
    print("mode(Hz) -> match(Hz) | ΔHz | Δ% | status")
    print("-" * 60)

    matched = 0
    errs = []

    for f0 in modes:
        lo = (1.0 - TOL_PCT) * f0
        hi = (1.0 + TOL_PCT) * f0

        cand = detected[(detected >= lo) & (detected <= hi)]
        if cand.size == 0:
            print(f"{f0:8.3f} -> {'-':>8} |  -  |  -  | NOT FOUND")
            continue

        # closest
        fp = float(cand[np.argmin(np.abs(cand - f0))])
        dhz = abs(fp - f0)
        dpct = 100.0 * dhz / max(f0, 1e-12)

        matched += 1
        errs.append(dpct)
        print(f"{f0:8.3f} -> {fp:8.3f} | {dhz:4.1f} | {dpct:4.2f} | OK")

    print("\nSummary:")
    print(f"Matched modes: {matched}/{len(modes)} within ±{int(TOL_PCT*100)}%")
    if errs:
        print(f"Mean Δ%: {np.mean(errs):.3f}% | Max Δ%: {np.max(errs):.3f}%")
    else:
        print("No matches. If you expect more, lower SNR_DB (e.g. 1.0) or increase END_S (e.g. 3.0).")


if __name__ == "__main__":
    main()
