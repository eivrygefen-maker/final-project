from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Tuple


WOODS_DEFAULT = ["spruce", "mahogany", "rosewood"]


def project_root() -> Path:
    # python/mesh_convergence.py -> root is one level up from python/
    return Path(__file__).resolve().parents[1]


def venv_python(root: Path) -> Path:
    return root / ".venv" / "bin" / "python"


def read_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def extract_modes_hz(result_json_path: Path) -> List[float]:
    data = read_json(result_json_path)
    # Expected structure: {"result": {"modes_hz": [...], ...}}
    res = data.get("result", data)
    modes = res.get("modes_hz") or res.get("modes")
    if not modes or not isinstance(modes, list):
        raise RuntimeError(f"Cannot find modes_hz in: {result_json_path}")
    return [float(x) for x in modes]


def run_fem(root: Path, config_path: Path, out_path: Path) -> None:
    py = venv_python(root)
    if not py.exists():
        raise RuntimeError(f"Missing venv python: {py}")

    cmd = [str(py), "FEM/scripts/fem_main.py", "--config", str(config_path), "--out", str(out_path)]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True)


def rel_change_percent(a: float, b: float) -> float:
    # relative change from a->b, using b as reference (finer mesh)
    if abs(b) < 1e-12:
        return 0.0
    return abs(a - b) / abs(b) * 100.0


def summarize_pair(modes_a: List[float], modes_b: List[float], n_modes: int) -> Tuple[float, float]:
    k = min(n_modes, len(modes_a), len(modes_b))
    eps = [rel_change_percent(modes_a[i], modes_b[i]) for i in range(k)]
    if not eps:
        return 0.0, 0.0
    return max(eps), sum(eps) / len(eps)


def make_modes_block(
    wood: str,
    bc: str,
    meshes: List[float],
    results_for_wood: Dict[float, List[float]],
    show_k: int,
) -> List[str]:
    title = f"WOOD: {wood.upper()} | BC: {bc}"
    header = "mesh     " + "".join([f"   f{i+1:02d}" for i in range(show_k)])
    sep = "-" * len(header)
    lines = [title, header, sep]
    for h in meshes:
        modes = results_for_wood[h]
        row = f"{h:<8g}" + "".join([f"{modes[i]:8.3f}" for i in range(show_k)])
        lines.append(row)
    return lines


def make_delta_table_rows(pairs: List[Tuple[float, float]]) -> List[str]:
    # Just the left column (pair string) to align nicely
    rows = []
    for (hc, hf) in pairs:
        rows.append(f"{hc:g} -> {hf:g}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Mesh convergence runner for FEM modal frequencies.")
    ap.add_argument("--woods", nargs="+", default=["all"], help="spruce mahogany rosewood | all")
    ap.add_argument("--bc", default="CCCC", choices=["SSSS", "CCCC"], help="Boundary condition")
    ap.add_argument("--meshes", nargs="+", type=float, required=True, help="mesh sizes, e.g. 0.01 0.008 0.00625")
    ap.add_argument("--n_modes", type=int, default=20, help="Number of modes to compare (for Δ% stats)")
    ap.add_argument("--show_k", type=int, default=6, help="How many first modes to show in the screenshot tables")
    ap.add_argument("--keep_tmp", action="store_true", help="Keep temp config JSONs (otherwise deleted)")
    args = ap.parse_args()

    root = project_root()
    meshes = list(args.meshes)
    # Sort coarse -> fine (bigger -> smaller) for meaningful successive comparisons
    meshes.sort(reverse=True)

    woods = WOODS_DEFAULT if (len(args.woods) == 1 and args.woods[0].lower() == "all") else [w.lower() for w in args.woods]

    py = venv_python(root)
    if not py.exists():
        raise RuntimeError(f"Venv python missing: {py}")

    out_dir = root / "FEM" / "outputs" / "mesh_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMesh convergence test | bc={args.bc} | meshes={meshes} | n_modes={args.n_modes}\n")

    tmp_paths: List[Path] = []

    # Store all results for side-by-side printing
    all_results: Dict[str, Dict[float, List[float]]] = {}
    delta_stats: Dict[str, Dict[Tuple[float, float], Tuple[float, float]]] = {}  # wood -> (hc,hf) -> (maxΔ, meanΔ)

    for wood in woods:
        base_cfg = root / "FEM" / "configs" / f"rect_plate_{wood}_ref.json"
        if not base_cfg.exists():
            print(f"[SKIP] Missing config: {base_cfg}")
            continue

        print("=" * 72)
        print(f"RUNNING WOOD: {wood.upper()} | BC: {args.bc}")
        print("=" * 72)

        results_for_wood: Dict[float, List[float]] = {}

        for h in meshes:
            cfg = read_json(base_cfg)

            cfg.setdefault("bc", {})
            cfg["bc"]["kind"] = args.bc

            cfg.setdefault("solver", {})
            cfg["solver"]["mesh_size"] = float(h)
            cfg["solver"]["n_modes"] = int(args.n_modes)

            # Write temp config
            if args.keep_tmp:
                tmp_cfg_path = root / "FEM" / "configs" / "_tmp_meshconv" / f"{wood}_{args.bc}_h{h}.json"
                write_json(tmp_cfg_path, cfg)
            else:
                fd, tmp_name = tempfile.mkstemp(prefix=f"meshconv_{wood}_{args.bc}_h{h}_", suffix=".json")
                os.close(fd)
                tmp_cfg_path = Path(tmp_name)
                write_json(tmp_cfg_path, cfg)
                tmp_paths.append(tmp_cfg_path)

            out_path = out_dir / f"{wood}_{args.bc}_h{h}_result.json"
            run_fem(root, tmp_cfg_path, out_path)
            results_for_wood[h] = extract_modes_hz(out_path)

        all_results[wood] = results_for_wood

        # Compute per-pair delta stats for this wood
        pairs = [(meshes[i], meshes[i + 1]) for i in range(len(meshes) - 1)]
        delta_stats[wood] = {}
        for hc, hf in pairs:
            maxd, meand = summarize_pair(results_for_wood[hc], results_for_wood[hf], args.n_modes)
            delta_stats[wood][(hc, hf)] = (maxd, meand)

    # Cleanup temp files if needed
    if not args.keep_tmp:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    # -------------------------------
    # SIDE-BY-SIDE: FIRST MODES TABLE
    # -------------------------------
    present_woods = [w for w in WOODS_DEFAULT if w in all_results]
    if not present_woods:
        print("\nNo woods were processed (configs missing?).")
        return 0

    show_k = max(1, min(args.show_k, args.n_modes, 12))  # keep screenshot-friendly
    blocks: List[List[str]] = []
    for w in present_woods:
        blocks.append(make_modes_block(w, args.bc, meshes, all_results[w], show_k))

    # width per block (rough, for nice alignment)
    # "mesh     " + show_k*(8 chars) + spacing
    block_width = len("mesh     " + "".join([f"   f{i+1:02d}" for i in range(show_k)])) + 2
    gap = " " * 6

    print("\n" + "=" * 120)
    print(f"FIRST MODES (Hz) — SIDE BY SIDE | BC={args.bc} | meshes={meshes} | shown modes={show_k}")
    print("=" * 120)

    for row_parts in zip_longest(*blocks, fillvalue=""):
        print(gap.join(part.ljust(block_width) for part in row_parts))

    # ---------------------------------
    # COMPACT Δ% TABLE (for screenshot)
    # ---------------------------------
    pairs = [(meshes[i], meshes[i + 1]) for i in range(len(meshes) - 1)]
    if pairs:
        print("\n" + "=" * 120)
        print(f"Δ% BETWEEN SUCCESSIVE MESHES (relative %, using finer mesh as reference) | modes compared: 1..{args.n_modes}")
        print("=" * 120)

        # Header
        left = "pair (coarse->fine)".ljust(22)
        cols = []
        for w in present_woods:
            cols.append(f"{w.upper():<20}")
        print(left + gap.join(cols))
        print("-" * (22 + len(cols) * 20 + (len(cols) - 1) * len(gap)))

        for (hc, hf) in pairs:
            line = f"{hc:g} -> {hf:g}".ljust(22)
            parts = []
            for w in present_woods:
                maxd, meand = delta_stats[w][(hc, hf)]
                parts.append(f"max {maxd:6.3f}% | mean {meand:6.3f}%")
            print(line + gap.join(parts))

        # Optional overall coarse->finest
        hc0, hf_last = meshes[0], meshes[-1]
        print("\nOverall (coarsest->finest):")
        left2 = f"{hc0:g} -> {hf_last:g}".ljust(22)
        parts2 = []
        for w in present_woods:
            maxd, meand = summarize_pair(all_results[w][hc0], all_results[w][hf_last], args.n_modes)
            parts2.append(f"max {maxd:6.3f}% | mean {meand:6.3f}%")
        print(left2 + gap.join(parts2))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
