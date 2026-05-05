#!/usr/bin/env python3
"""
Targeted LAB sweep for specific samples and disjoint frequency windows.

This script is intentionally isolated and does NOT merge into the production/master
training set automatically. It writes all artifacts under:

    FEM/results/EXTRA_RESULTS/sample_XXX/
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_WINDOWS: Tuple[Tuple[float, float], ...] = ((80.0, 100.0), (400.0, 600.0))
DEFAULT_SAMPLE_IDS: Tuple[int, ...] = tuple(list(range(1, 8)) + list(range(12, 21)))
DEFAULT_MAX_WORKERS = 2
WINDOW_QUOTA = 30


def _repo_root() -> Path:
    repo = Path(os.path.abspath(__file__)).resolve()
    while repo.name != "final-project" and repo.parent != repo:
        repo = repo.parent
    if repo.name != "final-project":
        raise RuntimeError(
            "Could not locate a parent directory named 'final-project' starting from "
            f"{Path(__file__).resolve()}"
        )
    return repo


def _run_step(name: str, cmd: List[str], cwd: Path) -> int:
    print(f"\n{'=' * 72}\n  {name}\n  $ {' '.join(cmd)}\n{'=' * 72}")
    sys.stdout.flush()
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def _parse_samples(raw: str) -> List[int]:
    out: List[int] = []
    for part in (x.strip() for x in raw.split(",") if x.strip()):
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    dedup_sorted = sorted({int(x) for x in out if int(x) > 0})
    if not dedup_sorted:
        raise ValueError("No valid sample ids were parsed.")
    return dedup_sorted


def _parse_windows(raw: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for part in (x.strip() for x in raw.split(",") if x.strip()):
        if ":" not in part:
            raise ValueError(f"Invalid window '{part}' (expected min:max).")
        lo_s, hi_s = part.split(":", 1)
        lo = float(lo_s)
        hi = float(hi_s)
        if hi < lo:
            lo, hi = hi, lo
        out.append((lo, hi))
    if not out:
        raise ValueError("No windows parsed.")
    return out


def _pool_sample_id(n: int) -> str:
    return f"sample_{n:03d}"


def _default_pool_path(repo: Path) -> Path:
    p1 = repo / "FEM" / "configs" / "lhs_pool.json"
    if p1.is_file():
        return p1
    return repo / "ROM" / "classic" / "lhs_pool.json"


def _find_pool_entry(pool_path: Path, sample_key: str) -> Dict[str, Any]:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Pool missing 'entries' list: {pool_path}")
    for e in entries:
        if not isinstance(e, dict):
            continue
        if str(e.get("id", "")) == sample_key:
            return e
    raise ValueError(f"Sample {sample_key} not found in pool: {pool_path}")


def _apply_dotted_parameters(config: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    for key, val in parameters.items():
        if not isinstance(key, str) or "." not in key:
            continue
        cur: Dict[str, Any] = config
        parts = [p for p in key.split(".") if p]
        if not parts:
            continue
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = val


def _resolve_sample_parameters(pool_entry: Dict[str, Any]) -> Dict[str, Any]:
    params = pool_entry.get("parameters", {})
    return dict(params) if isinstance(params, dict) else {}


def _window_tag(window: Tuple[float, float]) -> str:
    lo, hi = window
    return f"{int(lo):03d}_{int(hi):03d}"


def _ensure_scripts(repo: Path) -> Tuple[Path, Path, Path]:
    master = repo / "FEM" / "scripts" / "fem_master_dynamic.py"
    tuner = repo / "FEM" / "scripts" / "dynamic_filter_tuner.py"
    packer = repo / "FEM" / "scripts" / "package_rom.py"
    for p in (master, tuner, packer):
        if not p.is_file():
            raise FileNotFoundError(f"Missing script: {p}")
    return master, tuner, packer


def _sample_dir(root: Path, sample_id: int) -> Path:
    return root / _pool_sample_id(sample_id)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _count_selected_rows(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    # First line is header.
    return max(0, len(lines) - 1)


def _load_coverage_pending(sorting_root: Path) -> bool:
    p = sorting_root / "coverage_anchor_state.json"
    if not p.is_file():
        return False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("coverage_emergency_pending", False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run targeted LAB windows for selected samples and write outputs to "
            "FEM/results/EXTRA_RESULTS/sample_XXX/. No automatic merge into master."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="Base FEM config JSON (default: FEM/configs/guitar_3d.json).")
    parser.add_argument("--pool", type=Path, default=None, help="LHS pool path (default: FEM/configs/lhs_pool.json or ROM/classic/lhs_pool.json).")
    parser.add_argument(
        "--samples",
        type=str,
        default="1-7,12-20",
        help="Comma/range sample ids, e.g. '1-7,12-20' (default: 1-7,12-20).",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="80:100,400:600",
        help="Comma-separated windows as min:max (default: 80:100,400:600).",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Forwarded to fem_master_dynamic (default: 2).")
    parser.add_argument("--mpiexec", action="store_true", help="Pass --use-mpiexec to fem_master_dynamic.")
    args = parser.parse_args()

    if int(args.max_workers) != 2:
        print(
            f"Error: this LAB targeted runner enforces --max-workers 2 for OOM safety (got {args.max_workers}).",
            file=sys.stderr,
        )
        return 1

    repo = _repo_root()
    py = sys.executable
    master, tuner, packer = _ensure_scripts(repo)
    base_config = (args.config.resolve() if args.config else (repo / "FEM" / "configs" / "guitar_3d.json").resolve())
    if not base_config.is_file():
        print(f"Error: base config not found: {base_config}", file=sys.stderr)
        return 1
    pool_path = (args.pool.resolve() if args.pool else _default_pool_path(repo))
    if not pool_path.is_file():
        print(f"Error: pool not found: {pool_path}", file=sys.stderr)
        return 1

    samples = _parse_samples(str(args.samples))
    windows = _parse_windows(str(args.windows))

    extra_root = (repo / "FEM" / "results" / "EXTRA_RESULTS").resolve()
    extra_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "samples": samples,
        "windows": [{"min_hz": lo, "max_hz": hi} for lo, hi in windows],
        "max_workers": 2,
        "window_quota": WINDOW_QUOTA,
        "note": "No automatic merge into master training set.",
    }
    _write_json(extra_root / "run_manifest.json", manifest)

    failures: List[str] = []
    for sid in samples:
        skey = _pool_sample_id(sid)
        sdir = _sample_dir(extra_root, sid)
        sdir.mkdir(parents=True, exist_ok=True)
        try:
            entry = _find_pool_entry(pool_path, skey)
        except Exception as exc:
            failures.append(f"{skey}: pool entry missing ({exc})")
            continue

        try:
            cfg = json.loads(base_config.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Error: cannot read base config {base_config}: {exc}", file=sys.stderr)
            return 1
        merged_cfg = copy.deepcopy(cfg)
        _apply_dotted_parameters(merged_cfg, _resolve_sample_parameters(entry))
        merged_cfg_path = sdir / "merged_config.json"
        _write_json(merged_cfg_path, merged_cfg)

        sample_report: Dict[str, Any] = {"sample_id": skey, "windows": []}
        low_selected_streak = 0
        emergency_for_next_window = False
        for win in windows:
            lo, hi = win
            wtag = _window_tag(win)
            wdir = sdir / f"window_{wtag}"
            sorting_root = wdir / "SORTING"
            (sorting_root / "temp_modes").mkdir(parents=True, exist_ok=True)
            (sorting_root / "temp_results").mkdir(parents=True, exist_ok=True)
            _write_json(sorting_root / "candidates_log.json", {"candidates": [], "completed_shift_targets": []})

            master_cmd = [
                py,
                str(master),
                "--config",
                str(merged_cfg_path),
                "--hz-min",
                str(float(lo)),
                "--hz-max",
                str(float(hi)),
                "--max-workers",
                "2",
                "--sorting-root",
                str(sorting_root),
            ]
            if args.mpiexec:
                master_cmd.append("--use-mpiexec")

            if _run_step(f"{skey} window {wtag} | Step A master sweep", master_cmd, repo) != 0:
                failures.append(f"{skey} window {wtag}: master failed")
                continue

            selected_csv = wdir / "selected_modes.csv"
            plot_out = wdir / "selection_plot.png"
            selection_metadata = wdir / "selection_metadata.json"
            emergency_for_this_window = bool(emergency_for_next_window or _load_coverage_pending(sorting_root))
            selection_type = "coverage_anchor" if emergency_for_this_window else "primary"
            tuner_cmd = [
                py,
                str(tuner),
                "--headless",
                "--candidates",
                str(sorting_root / "candidates_log.json"),
                "--window-min",
                str(float(lo)),
                "--window-max",
                str(float(hi)),
                "--quota",
                str(WINDOW_QUOTA),
                "--min-selected",
                str(WINDOW_QUOTA),
                "--adaptive-veto",
                "--adaptive-steps",
                "12",
                "--selection-type",
                selection_type,
                "--metadata-out",
                str(selection_metadata),
                "--export",
                str(selected_csv),
                "--plot-out",
                str(plot_out),
            ]
            if emergency_for_this_window:
                tuner_cmd.extend(
                    [
                        "--wood-floor-min",
                        "0.0",
                        "--min-selected",
                        "5",
                    ]
                )
            if _run_step(f"{skey} window {wtag} | Step B tuner", tuner_cmd, repo) != 0:
                failures.append(f"{skey} window {wtag}: tuner failed")
                continue

            selected_count = _count_selected_rows(selected_csv)
            if selected_count <= 1:
                low_selected_streak += 1
            else:
                low_selected_streak = 0
            emergency_for_next_window = low_selected_streak >= 2

            window_npz = wdir / "targeted_window_rom.npz"
            pack_cmd = [
                py,
                str(packer),
                "--csv",
                str(selected_csv),
                "--out",
                str(window_npz),
                "--sorting-root",
                str(sorting_root),
            ]
            if _run_step(f"{skey} window {wtag} | Step C package", pack_cmd, repo) != 0:
                failures.append(f"{skey} window {wtag}: package failed")
                continue

            sample_report["windows"].append(
                {
                    "window": {"min_hz": lo, "max_hz": hi},
                    "window_tag": wtag,
                    "sorting_root": str(sorting_root),
                    "selected_csv": str(selected_csv),
                    "plot": str(plot_out),
                    "selection_metadata": str(selection_metadata),
                    "selected_count": int(selected_count),
                    "selection_type": selection_type,
                    "npz": str(window_npz),
                }
            )

        _write_json(sdir / "sample_manifest.json", sample_report)

    status = {
        "ok": len(failures) == 0,
        "failures": failures,
        "root": str(extra_root),
        "manual_merge_hint": (
            "python FEM/scripts/merge_extra_results.py "
            "--extra-root FEM/results/EXTRA_RESULTS --per-zone-top-k 30"
        ),
    }
    _write_json(extra_root / "run_status.json", status)

    if failures:
        print("\nCompleted with failures:")
        for f in failures:
            print(f"  - {f}")
        print(f"\nOutputs preserved at: {extra_root}")
        print("No merge into master was performed.")
        return 1

    print(f"\nTargeted LAB sweep complete: {extra_root}")
    print("No merge into master was performed.")
    print(
        "To merge targeted selections manually, run:\n"
        "  python FEM/scripts/merge_extra_results.py --extra-root FEM/results/EXTRA_RESULTS --per-zone-top-k 30"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

