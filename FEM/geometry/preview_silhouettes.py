#!/usr/bin/env python3
"""2D preview of luthier B-spline control hulls (no Gmsh)."""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1)

from generate_reference_models import (
    ACOUSTIC_LOOP,
    ACOUSTIC_TAIL_TIP,
    ACOUSTIC_TOP_HALF,
    CLASSICAL_LOOP,
    CLASSICAL_TAIL_TIP,
    CLASSICAL_TOP_HALF,
    REFERENCE_NOMINAL_WIDTHS,
    classical_guitar_perimeter,
    dreadnought_guitar_perimeter,
)

OUT = Path(__file__).resolve().parent / "models" / "silhouette_preview.png"


def _plot(ax, pts, title: str, color: str) -> None:
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.fill(xs, ys, alpha=0.2, color=color)
    ax.plot(xs, ys, "-", color=color, linewidth=1.2, label="control hull")
    ax.plot(xs, ys, "o", color=color, markersize=3)
    ax.axvline(0.0, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m) — neck at +x")
    ax.grid(True, alpha=0.3)


def main() -> int:
    cl = classical_guitar_perimeter()
    ac = dreadnought_guitar_perimeter()
    u_cl, w_cl, lo_cl = REFERENCE_NOMINAL_WIDTHS["classical"]
    u_ac, w_ac, lo_ac = REFERENCE_NOMINAL_WIDTHS["acoustic"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot(
        axes[0],
        cl,
        f"Torres / Classical\n"
        f"{len(CLASSICAL_TOP_HALF)} half pts + tip · upper/waist/lower "
        f"{u_cl:.2f}/{w_cl:.2f}/{lo_cl:.2f} m",
        "#c4a574",
    )
    _plot(
        axes[1],
        ac,
        f"Martin D-28 / Dreadnought\n"
        f"{len(ACOUSTIC_TOP_HALF)} half pts + tip · upper/waist/lower "
        f"{u_ac:.2f}/{w_ac:.2f}/{lo_ac:.2f} m",
        "#8b6914",
    )
    fig.suptitle(
        "Luthier B-spline control polygons (STEP uses occ.addBSpline)",
        fontsize=13,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")
    print(f"Classical tip {CLASSICAL_TAIL_TIP} · Acoustic tip {ACOUSTIC_TAIL_TIP}")
    if "--show" in sys.argv:
        plt.show()
    else:
        plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
