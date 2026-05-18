import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI
from scipy.stats import qmc

BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = BASE_DIR / "FEM" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))
import fem_main_3d
from wood_library import apply_lhs_parameters_to_config


def _logging_reset_handlers() -> None:
    """Detach all handlers from the root logger to avoid cross-sample log bleed."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


class ROMManager:
    def __init__(self, base_dir: Optional[Path] = None, shapes_config_path: Optional[Path] = None):
        self.comm = MPI.COMM_WORLD
        self.rank = int(self.comm.rank)
        # Always anchor paths at project root (FEM/rom/rom_manager.py -> ../../).
        self.base_dir = Path(__file__).resolve().parents[2]
        self.rom_root = self.base_dir / "ROM"
        self.rom_root.mkdir(parents=True, exist_ok=True)
        self.shapes_config_path = (
            Path(shapes_config_path)
            if shapes_config_path
            else self.base_dir / "FEM" / "configs" / "rom_shapes.json"
        )
        self.shapes = self._load_shapes_config()
        self._basis_cache: Dict[str, Dict] = {}
        self._last_collect_summary: Dict = {}

    def _load_shapes_config(self) -> Dict:
        if not self.shapes_config_path.exists():
            raise FileNotFoundError(f"ROM shapes config not found: {self.shapes_config_path}")
        with open(self.shapes_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "shapes" not in data:
            raise ValueError("rom_shapes.json must contain a top-level 'shapes' object.")
        return data["shapes"]

    def list_shapes(self) -> List[str]:
        return sorted(self.shapes.keys())

    def _shape_paths(self, shape_name: str) -> Dict[str, Path]:
        if shape_name not in self.shapes:
            raise KeyError(f"Unknown shape '{shape_name}'. Add it to {self.shapes_config_path}.")
        shape_root = (self.rom_root / shape_name).resolve()
        snapshots_dir = shape_root / "snapshots"
        basis_path = shape_root / "reduced_basis.npz"
        lhs_pool_path = shape_root / "lhs_pool.json"
        return {"root": shape_root, "snapshots": snapshots_dir, "basis": basis_path, "lhs_pool": lhs_pool_path}

    def get_lhs_pool_path(self, shape_name: str) -> Path:
        return self._shape_paths(shape_name)["lhs_pool"].resolve()

    @staticmethod
    def _apply_solver_profile_overrides(cfg: Dict, pool: Dict) -> Dict[str, float]:
        """
        Apply optional runtime sifter/gear overrides from LHS pool metadata.
        Returns the resolved values actually written for traceability logs.
        """
        solver = cfg.setdefault("solver", {})
        profile = pool.get("solver_profile", {}) if isinstance(pool, dict) else {}
        if not isinstance(profile, dict):
            profile = {}

        low_rng = profile.get("low_gear_hz", [90.0, 160.0])
        if isinstance(low_rng, (list, tuple)) and len(low_rng) >= 2:
            low_min = float(low_rng[0])
            low_max = float(low_rng[1])
        else:
            low_min = float(solver.get("sifter_low_freq_min_hz", 90.0))
            low_max = float(solver.get("sifter_low_freq_max_hz", 160.0))

        low_batch = int(
            profile.get(
                "sifter_low_target",
                profile.get("low_gear_batch", solver.get("sifter_low_target", solver.get("sifter_low_batch_modes", 90))),
            )
        )
        high_batch = int(
            profile.get(
                "sifter_high_target",
                profile.get("high_gear_batch", solver.get("sifter_high_target", solver.get("sifter_high_batch_modes", 70))),
            )
        )
        dup_gate_hz = float(
            profile.get(
                "sifter_dup_hz",
                profile.get("near_pair_filter_hz", solver.get("sifter_dup_hz", 1.5)),
            )
        )
        energy_near_hz = float(
            profile.get("sifter_energy_priority_hz", solver.get("sifter_energy_priority_hz", 2.0))
        )
        uniq_min = float(profile.get("sifter_uniqueness_min", solver.get("sifter_uniqueness_min", 0.12)))
        uniq_min_high = float(
            profile.get("sifter_uniqueness_min_high_band", solver.get("sifter_uniqueness_min_high_band", 0.06))
        )
        adaptive_break = float(profile.get("sifter_adaptive_break_hz", solver.get("sifter_adaptive_break_hz", 200.0)))
        dup_hz_high = float(profile.get("sifter_dup_hz_high_band", solver.get("sifter_dup_hz_high_band", 0.8)))
        wood_high = float(profile.get("min_wood_participation_high_band", solver.get("min_wood_participation_high_band", 0.08)))
        low_step_hz = float(profile.get("low_gear_step_hz", solver.get("sifter_low_step_hz", 15.0)))
        high_step_hz = float(profile.get("high_gear_step_hz", solver.get("sifter_high_step_hz", 25.0)))
        batch_max_it = int(profile.get("batch_max_it", solver.get("sifter_batch_max_it", 50)))
        min_wood_participation = float(profile.get("min_wood_participation", solver.get("min_wood_participation", 0.01)))
        max_acoustic_only = int(profile.get("max_acoustic_only_modes", solver.get("max_acoustic_only_modes", 3)))
        profile_name = str(profile.get("name", profile.get("profile_name", "LHS Pool Profile")))

        solver["sifter_low_freq_min_hz"] = low_min
        solver["sifter_low_freq_max_hz"] = low_max
        solver["sifter_low_target"] = low_batch
        solver["sifter_high_target"] = high_batch
        solver["sifter_low_batch_modes"] = low_batch
        solver["sifter_high_batch_modes"] = high_batch
        solver["sifter_dup_hz"] = dup_gate_hz
        solver["sifter_dup_hz_high_band"] = dup_hz_high
        solver["sifter_adaptive_break_hz"] = adaptive_break
        solver["sifter_energy_priority_hz"] = energy_near_hz
        solver["sifter_uniqueness_min"] = uniq_min
        solver["sifter_uniqueness_min_high_band"] = uniq_min_high
        solver["sifter_low_step_hz"] = low_step_hz
        solver["sifter_high_step_hz"] = high_step_hz
        solver["sifter_batch_max_it"] = batch_max_it
        solver["min_wood_participation"] = min_wood_participation
        solver["min_wood_participation_high_band"] = wood_high
        solver["max_acoustic_only_modes"] = max_acoustic_only
        solver["solver_profile_name"] = profile_name

        return {
            "solver_profile_name": profile_name,
            "sifter_low_freq_min_hz": low_min,
            "sifter_low_freq_max_hz": low_max,
            "sifter_low_target": low_batch,
            "sifter_high_target": high_batch,
            "sifter_low_batch_modes": low_batch,
            "sifter_high_batch_modes": high_batch,
            "sifter_dup_hz": dup_gate_hz,
            "sifter_dup_hz_high_band": dup_hz_high,
            "sifter_adaptive_break_hz": adaptive_break,
            "sifter_energy_priority_hz": energy_near_hz,
            "sifter_uniqueness_min": uniq_min,
            "sifter_uniqueness_min_high_band": uniq_min_high,
            "sifter_low_step_hz": low_step_hz,
            "sifter_high_step_hz": high_step_hz,
            "sifter_batch_max_it": batch_max_it,
            "min_wood_participation": min_wood_participation,
            "min_wood_participation_high_band": wood_high,
            "max_acoustic_only_modes": max_acoustic_only,
        }

    @staticmethod
    def _sync_pool_solver_profile_from_base_config(base_cfg: Dict, pool: Dict) -> bool:
        """
        Keep pool solver_profile aligned with current base config so new batches inherit
        the latest "single source of truth" settings (targets, gaps, participation gates).
        Returns True if pool was modified.
        """
        solver = base_cfg.get("solver", {}) if isinstance(base_cfg, dict) else {}
        if not isinstance(pool, dict):
            return False
        prof = pool.get("solver_profile")
        if not isinstance(prof, dict):
            prof = {}
            pool["solver_profile"] = prof

        desired = {
            "name": str(solver.get("solver_profile_name", "Quality over Quantity")),
            "low_gear_hz": [
                float(solver.get("sifter_low_freq_min_hz", 90.0)),
                float(solver.get("sifter_low_freq_max_hz", 160.0)),
            ],
            "sifter_low_target": int(solver.get("sifter_low_target", solver.get("sifter_low_batch_modes", 50))),
            "sifter_high_target": int(solver.get("sifter_high_target", solver.get("sifter_high_batch_modes", 40))),
            "low_gear_batch": int(solver.get("sifter_low_batch_modes", solver.get("sifter_low_target", 50))),
            "high_gear_batch": int(solver.get("sifter_high_batch_modes", solver.get("sifter_high_target", 40))),
            "near_pair_filter_hz": float(solver.get("sifter_dup_hz", 1.5)),
            "sifter_dup_hz": float(solver.get("sifter_dup_hz", 1.5)),
            "sifter_dup_hz_high_band": float(solver.get("sifter_dup_hz_high_band", 0.8)),
            "sifter_adaptive_break_hz": float(solver.get("sifter_adaptive_break_hz", 200.0)),
            "sifter_uniqueness_min": float(solver.get("sifter_uniqueness_min", 0.12)),
            "sifter_uniqueness_min_high_band": float(solver.get("sifter_uniqueness_min_high_band", 0.06)),
            "sifter_energy_priority_hz": float(solver.get("sifter_energy_priority_hz", 2.0)),
            "min_wood_participation": float(solver.get("min_wood_participation", 0.01)),
            "min_wood_participation_high_band": float(solver.get("min_wood_participation_high_band", 0.08)),
        }
        changed = False
        for k, v in desired.items():
            if prof.get(k) != v:
                prof[k] = v
                changed = True
        return changed

    @staticmethod
    def _shape_length_width_depth_bounds(shape_type: str) -> Dict[str, Dict[str, float]]:
        st = str(shape_type).lower()
        if "dreadnought" in st:
            return {
                "geometry.length": {"min": 0.45, "max": 0.70},
                "geometry.width": {"min": 0.30, "max": 0.55},
                "geometry.depth": {"min": 0.10, "max": 0.20},
            }
        if "box" in st:
            return {
                "geometry.length": {"min": 0.10, "max": 1.00},
                "geometry.width": {"min": 0.10, "max": 0.80},
                "geometry.depth": {"min": 0.01, "max": 0.50},
            }
        # Classical defaults (match GUI sliders).
        return {
            "geometry.length": {"min": 0.35, "max": 0.60},
            "geometry.width": {"min": 0.20, "max": 0.45},
            "geometry.depth": {"min": 0.08, "max": 0.15},
        }

    def _build_7d_lhs_sweep_spec(self, shape_name: str, sweep_cfg: Dict) -> Dict:
        """Seven-parameter LHS: L, W, D, thickness, hole radius, top wood ID, back wood ID."""
        from wood_library import ALL_WOOD_IDS

        base_cfg = self._load_shape_base_config(shape_name)
        shape_type = str(base_cfg.get("geometry", {}).get("shape_type", "Classical"))
        bounds = self._shape_length_width_depth_bounds(shape_type)
        wood_options = list(ALL_WOOD_IDS)
        spec = {
            "geometry.length": bounds["geometry.length"],
            "geometry.width": bounds["geometry.width"],
            "geometry.depth": bounds["geometry.depth"],
            "geometry.thickness": {"min": 0.002, "max": 0.006},
            "geometry.hole_radius": {"min": 0.035, "max": 0.055},
            "top_wood_id": wood_options,
            "back_wood_id": wood_options,
        }
        for key in list(spec.keys()):
            if key in sweep_cfg:
                spec[key] = sweep_cfg[key]
        return spec

    # Backward-compatible alias
    _build_5d_lhs_sweep_spec = _build_7d_lhs_sweep_spec

    def _load_shape_base_config(self, shape_name: str) -> Dict:
        shape_cfg = self.shapes[shape_name]
        config_path = self.base_dir / shape_cfg["base_config"]
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Force mesh path to project-local FEM/mesh to avoid stale external paths.
        # All other solver keys (adaptive_mode_sifter, sifter_*, shift_invert_target_hz, …) pass through unchanged.
        solver = cfg.setdefault("solver", {})
        mesh_name = Path(str(solver.get("mesh_file", "guitar_3d.msh"))).name
        solver["mesh_file"] = str((self.base_dir / "FEM" / "mesh" / mesh_name).resolve())
        return cfg

    def _save_shape_base_config(self, shape_name: str, cfg: Dict) -> None:
        shape_cfg = self.shapes[shape_name]
        config_path = self.base_dir / shape_cfg["base_config"]
        if self.rank == 0:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)

    def _rebuild_mesh(self, shape_name: str) -> None:
        cfg = self._load_shape_base_config(shape_name)
        mesh_file = Path(cfg["solver"]["mesh_file"])
        xdmf_cache = mesh_file.parent / "_xdmf_cache"
        if self.rank == 0 and xdmf_cache.exists():
            shutil.rmtree(xdmf_cache, ignore_errors=True)
        geom_script = self.base_dir / "FEM" / "geometry" / "build_3d_guitar.py"
        cmd = [sys.executable, str(geom_script), "-nopopup"]
        if self.rank == 0:
            proc = subprocess.run(cmd, cwd=str(self.base_dir), capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "Mesh regeneration failed during force-pool-rebuild.\n"
                    f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
                )
        self.comm.barrier()

    @staticmethod
    def _apply_parameters_to_config(config: Dict, parameters: Dict) -> None:
        """Merge LHS flat parameters and discrete wood IDs into a FEM config."""
        apply_lhs_parameters_to_config(config, parameters)

    @staticmethod
    def _set_nested(config: Dict, dotted_key: str, value):
        cur = config
        parts = dotted_key.split(".")
        for key in parts[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[parts[-1]] = value

    @staticmethod
    def _expand_parameter_values(spec):
        if isinstance(spec, list):
            return spec
        if isinstance(spec, dict):
            if "values" in spec:
                return list(spec["values"])
            if "min" in spec and "max" in spec:
                vmin = float(spec["min"])
                vmax = float(spec["max"])
                if "count" in spec:
                    count = int(spec["count"])
                    if count <= 1:
                        vals = [vmin]
                    else:
                        vals = np.linspace(vmin, vmax, count).tolist()
                elif "step" in spec:
                    step = float(spec["step"])
                    if step <= 0:
                        raise ValueError(f"Invalid sweep step={step}; must be > 0.")
                    vals = np.arange(vmin, vmax + 0.5 * step, step).tolist()
                else:
                    raise ValueError("Range sweep spec must include either 'count' or 'step'.")
                dtype = str(spec.get("dtype", "float")).lower()
                if dtype in ("int", "integer"):
                    vals = [int(round(v)) for v in vals]
                return vals
        raise ValueError(f"Unsupported sweep spec: {spec}")

    @classmethod
    def _grid_from_sweep(cls, sweep: Dict) -> List[Dict]:
        keys = sorted(sweep.keys())
        vals = [cls._expand_parameter_values(sweep[k]) for k in keys]
        combos = []
        for row in product(*vals):
            combos.append({k: v for k, v in zip(keys, row)})
        return combos

    @classmethod
    def _lhs_from_sweep(cls, sweep: Dict, samples: int, seed: int = 123) -> List[Dict]:
        if samples <= 0:
            raise ValueError("LHS samples must be positive.")
        rng = np.random.default_rng(seed)
        keys = sorted(sweep.keys())
        cols: Dict[str, List] = {}

        for key in keys:
            spec = sweep[key]
            if isinstance(spec, dict) and "min" in spec and "max" in spec:
                vmin = float(spec["min"])
                vmax = float(spec["max"])
                dtype = str(spec.get("dtype", "float")).lower()
                perm = rng.permutation(samples)
                u = (perm + rng.random(samples)) / samples
                vals = vmin + (vmax - vmin) * u
                if dtype in ("int", "integer"):
                    vals = np.rint(vals).astype(np.int64)
                cols[key] = vals.tolist()
            else:
                options = cls._expand_parameter_values(spec)
                picks = rng.integers(low=0, high=len(options), size=samples)
                cols[key] = [options[int(i)] for i in picks]

        out = []
        for i in range(samples):
            out.append({k: cols[k][i] for k in keys})
        return out

    @staticmethod
    def _write_json(path: Path, data: Dict, rank: int = 0, comm=None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(path)

    @staticmethod
    def _snapshot_telemetry_arrays(fom: Dict, freqs_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align uniqueness_scores and participation_ratios with freqs_hz; NaN-fill if solver omitted telemetry."""
        uniq_arr = np.asarray(fom.get("uniqueness_scores", []), dtype=np.float64).reshape(-1)
        part_arr = np.asarray(fom.get("participation_ratios", []), dtype=np.float64).reshape(-1)
        n = int(freqs_arr.size)
        if uniq_arr.size == 0 and part_arr.size == 0 and n > 0:
            return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
        if uniq_arr.size != n:
            raise ValueError(f"uniqueness_scores length ({uniq_arr.size}) != freqs_hz length ({n})")
        if part_arr.size != n:
            raise ValueError(f"participation_ratios length ({part_arr.size}) != freqs_hz length ({n})")
        return uniq_arr, part_arr

    @staticmethod
    def _sample_id(i: int) -> str:
        return f"sample_{i + 1:03d}"

    @classmethod
    def _lhs_values_for_key(cls, spec, uvals: np.ndarray) -> List:
        if isinstance(spec, dict) and "min" in spec and "max" in spec:
            vmin = float(spec["min"])
            vmax = float(spec["max"])
            vals = vmin + (vmax - vmin) * uvals
            dtype = str(spec.get("dtype", "float")).lower()
            if dtype in ("int", "integer"):
                vals = np.rint(vals).astype(np.int64)
            return vals.tolist()
        options = cls._expand_parameter_values(spec)
        if len(options) <= 0:
            raise ValueError("Sweep options list cannot be empty.")
        idx = np.floor(uvals * len(options)).astype(int)
        idx = np.clip(idx, 0, len(options) - 1)
        return [options[int(i)] for i in idx]

    def _create_lhs_pool(self, shape_name: str, sweep_cfg: Dict, total_samples: int, seed: int = 123) -> Dict:
        if total_samples <= 0:
            raise ValueError("LHS pool size must be > 0.")
        keys = sorted(sweep_cfg.keys())
        if not keys:
            raise ValueError("LHS sampling requires a non-empty parameter_sweep.")

        sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
        unit = sampler.random(n=total_samples)
        cols = {k: self._lhs_values_for_key(sweep_cfg[k], unit[:, i]) for i, k in enumerate(keys)}

        entries = []
        for i in range(total_samples):
            params = {k: cols[k][i] for k in keys}
            entries.append(
                {
                    "id": self._sample_id(i),
                    "parameters": params,
                    "status": "pending",
                    "snapshot_file": None,
                    "error": None,
                }
            )
        return {
            "shape_name": shape_name,
            "sampling": "lhs",
            "wood_assignment": "unrestricted_5x5",
            "seed": int(seed),
            "total_samples": int(total_samples),
            "mpi_world_size": int(self.comm.size),
            "entries": entries,
        }

    def _load_or_create_lhs_pool(
        self,
        shape_name: str,
        sweep_cfg: Dict,
        total_samples: int,
        seed: int = 123,
        force_rebuild: bool = False,
        pool_path: Optional[Path] = None,
    ) -> Dict:
        paths = self._shape_paths(shape_name)
        pool_path = (pool_path.resolve() if pool_path is not None else paths["lhs_pool"].resolve())
        master_pool_path = pool_path
        if self.rank == 0:
            print(f"DEBUG: Loading Master Pool from: {os.path.abspath(master_pool_path)}")
        if pool_path.exists() and not force_rebuild:
            with open(pool_path, "r", encoding="utf-8") as f:
                pool = json.load(f)
            pool_world_size = int(pool.get("mpi_world_size", 0) or 0)
            if pool_world_size not in (0, int(self.comm.size)):
                # Rank-count changes can leave stale in-progress ownership/status;
                # force a clean pool when communicator size differs.
                force_rebuild = True
                if self.rank == 0:
                    print(
                        f"[ROM] Rebuilding LHS pool due to MPI size change: "
                        f"pool={pool_world_size} current={self.comm.size}"
                    )
            else:
                return pool
        if pool_path.exists() and force_rebuild:
            pool_path.unlink(missing_ok=True)
        pool = self._create_lhs_pool(shape_name, sweep_cfg=sweep_cfg, total_samples=total_samples, seed=seed)
        self._write_json(pool_path, pool, rank=self.rank, comm=self.comm)
        return pool

    @staticmethod
    def _pool_status_counts(pool: Dict) -> Dict[str, int]:
        counts = {"pending": 0, "running": 0, "completed": 0, "error": 0}
        for entry in pool.get("entries", []):
            st = str(entry.get("status", "pending"))
            if st in counts:
                counts[st] += 1
        return counts

    def reset_lhs_errors(self, shape_name: str) -> int:
        pool_path = self.get_lhs_pool_path(shape_name)
        if not pool_path.exists():
            return 0
        with open(pool_path, "r", encoding="utf-8") as f:
            pool = json.load(f)
        changed = 0
        for entry in pool.get("entries", []):
            if str(entry.get("status", "")) == "error":
                entry["status"] = "pending"
                entry["error"] = None
                changed += 1
        if changed > 0:
            self._write_json(pool_path, pool, rank=self.rank, comm=self.comm)
        return changed

    @staticmethod
    def _next_snapshot_index(snapshots_dir: Path) -> int:
        pattern = re.compile(r"^snapshot_(\d+)\.npz$")
        max_idx = -1
        for path in snapshots_dir.glob("snapshot_*.npz"):
            m = pattern.match(path.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def collect_snapshots(
        self,
        shape_name: str,
        num_modes: int = 50,
        sampling: Optional[str] = None,
        lhs_samples: Optional[int] = None,
        pool_size: int = 500,
        max_runs: int = 50,
        seed: int = 123,
        dry_run: bool = False,
        retry_errors: bool = False,
        force_pool_rebuild: bool = False,
        force_rerun: bool = False,
    ) -> List[Path]:
        shape_cfg = self.shapes[shape_name]
        main_paths = self._shape_paths(shape_name)
        # Enforce strict structure: ROM/<shape>/lhs_pool.json and ROM/<shape>/snapshots/.
        paths = main_paths
        paths["root"].mkdir(parents=True, exist_ok=True)
        paths["snapshots"].mkdir(parents=True, exist_ok=True)
        logs_dir = self.base_dir / "runs" / "logs" / shape_name
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Full refresh path: rebuild .msh once.
        if force_pool_rebuild:
            self._rebuild_mesh(shape_name)

        sweep_cfg = shape_cfg.get("parameter_sweep", {})
        sampling_mode = str(sampling or shape_cfg.get("sampling", "structured")).lower()
        if not sweep_cfg:
            grid = [{}]
        elif sampling_mode == "lhs":
            n = int(lhs_samples if lhs_samples is not None else pool_size)
            lhs_7d_spec = self._build_7d_lhs_sweep_spec(shape_name, sweep_cfg)
            pool = self._load_or_create_lhs_pool(
                shape_name,
                sweep_cfg=lhs_7d_spec,
                total_samples=n,
                seed=seed,
                force_rebuild=force_pool_rebuild,
                pool_path=paths["lhs_pool"],
            )
            base_cfg_for_sync = self._load_shape_base_config(shape_name)
            if self._sync_pool_solver_profile_from_base_config(base_cfg_for_sync, pool):
                self._write_json(paths["lhs_pool"], pool, rank=self.rank, comm=self.comm)
            if retry_errors:
                changed = 0
                for entry in pool.get("entries", []):
                    if str(entry.get("status", "")) == "error":
                        entry["status"] = "pending"
                        entry["error"] = None
                        changed += 1
                if changed > 0:
                    self._write_json(paths["lhs_pool"], pool, rank=self.rank, comm=self.comm)
            grid = []
        else:
            grid = self._grid_from_sweep(sweep_cfg)
        start_idx = max(1, self._next_snapshot_index(paths["snapshots"]))
        out_files: List[Path] = []
        if dry_run:
            if sampling_mode == "lhs":
                pending = [e for e in pool.get("entries", []) if e.get("status") == "pending"]
                self._last_collect_summary = {
                    "shape": shape_name,
                    "sampling": sampling_mode,
                    "dry_run": True,
                    "planned_batch_size": min(len(pending), int(max_runs)),
                    "pool_counts": self._pool_status_counts(pool),
                }
                return [
                    paths["snapshots"] / f"snapshot_{start_idx + off:04d}.npz"
                    for off in range(min(len(pending), int(max_runs)))
                ]
            self._last_collect_summary = {
                "shape": shape_name,
                "sampling": sampling_mode,
                "dry_run": True,
                "planned_batch_size": len(grid),
            }
            return [paths["snapshots"] / f"snapshot_{start_idx + off:04d}.npz" for off in range(len(grid))]

        if sampling_mode == "lhs":
            pool_path = paths["lhs_pool"]
            next_idx = start_idx
            processed = 0
            completed_batch = 0
            error_batch = 0
            skipped_success = 0
            while True:
                if processed >= int(max_runs):
                    break
                if pool_path.exists():
                    with open(pool_path, "r", encoding="utf-8") as f:
                        pool = json.load(f)
                entries = pool.get("entries", [])
                selected_idx = None
                selected_status = ""
                for idx, candidate in enumerate(entries):
                    status_raw = candidate.get("status", "")
                    status_value = "" if status_raw is None else str(status_raw).lower()
                    if self.rank == 0:
                        cid = str(candidate.get("id", f"sample_{idx + 1:03d}"))
                        print(f"DEBUG: Checking {cid} | Status: {status_value}")
                    if force_rerun:
                        selected_idx = idx
                        selected_status = status_value
                        break
                    # Treat these statuses as finished and never re-select them.
                    if status_value in ["completed", "success", "processing"]:
                        if self.rank == 0:
                            cid = str(candidate.get("id", f"sample_{idx + 1:03d}"))
                            print(f"DEBUG: Skipping {cid} due to finished status='{status_value}'")
                        continue
                    # Select only clear "not finished yet" states.
                    if status_value in ("", "pending", "error", "failed"):
                        selected_idx = idx
                        selected_status = status_value
                        break
                if selected_idx is None:
                    if self.rank == 0:
                        print("All samples completed")
                    break
                entry = entries[selected_idx]
                sample_id = str(entry.get("id", f"sample_{selected_idx + 1:03d}"))
                status = str(entry.get("status", "")).lower()
                if self.rank == 0:
                    print(f"DEBUG: Selected {sample_id} because current status is '{selected_status}'")
                if (status in ("success", "completed")) and not force_rerun:
                    if self.rank == 0:
                        print(f"INFO: Skipping {sample_id} - already marked as SUCCESS.")
                    # Persist immediately so skip decisions survive interruptions.
                    self._write_json(pool_path, pool, rank=self.rank, comm=self.comm)
                    skipped_success += 1
                    continue
                if not force_rerun and status not in ("", "pending", "error", "failed"):
                    continue
                params = entry.get("parameters", {})
                entry["status"] = "running"
                entry["error"] = None
                self._write_json(pool_path, pool, rank=self.rank, comm=self.comm)

                snapshot_raw = entry.get("snapshot_file")
                if isinstance(snapshot_raw, str) and snapshot_raw.strip():
                    candidate_path = Path(snapshot_raw.strip())
                    snapshot_name = candidate_path.name if candidate_path.suffix.lower() == ".npz" else f"snapshot_{next_idx:04d}.npz"
                    snapshot_path = (paths["snapshots"] / snapshot_name).resolve()
                else:
                    snapshot_path = (paths["snapshots"] / f"snapshot_{next_idx:04d}.npz").resolve()
                if snapshot_path.name == "snapshot_0000.npz":
                    raise RuntimeError(
                        "Refusing to overwrite Gold Reference snapshot_0000.npz. "
                        "Use a separate output path under ROM/<shape>/snapshots/."
                    )
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sample_label = str(entry.get("id", f"{next_idx:04d}"))
                log_path = logs_dir / f"simulation_guitar_{sample_label}_{run_stamp}.log"
                if self.rank == 0:
                    print(f"DEBUG: Running sample {sample_label} | Targeted Output: {snapshot_path}")
                solver_profile_used = None
                try:
                    _logging_reset_handlers()
                    cfg = self._load_shape_base_config(shape_name)
                    self._apply_parameters_to_config(cfg, params)
                    solver_profile_used = self._apply_solver_profile_overrides(cfg, pool)
                    # Offline ROM FOM must be coupled; persist false so guitar_3d.json cannot drift to vacuum-only.
                    cfg.setdefault("solver", {})["structural_only_diagnosis"] = False
                    self._save_shape_base_config(shape_name, cfg)
                    t0 = time.perf_counter()
                    fom = fem_main_3d.run_fom_for_rom(cfg, num_modes=num_modes)
                    elapsed = time.perf_counter() - t0
                    if self.rank == 0:
                        freqs_arr = np.array(fom["freqs_hz"], dtype=np.float64)
                        uniq_arr, part_arr = self._snapshot_telemetry_arrays(fom, freqs_arr)
                        np.savez(
                            snapshot_path,
                            params_json=json.dumps(params),
                            shape_name=shape_name,
                            sampling_mode=sampling_mode,
                            sample_id=str(entry.get("id", "")),
                            freqs_hz=freqs_arr,
                            eigvecs_real=np.real(fom["eigvecs"]).astype(np.float64),
                            uniqueness_scores=uniq_arr,
                            participation_ratios=part_arr,
                            elapsed_s=np.array([elapsed], dtype=np.float64),
                            sifter_stats_json=json.dumps(fom.get("sifter_stats", {}), indent=2),
                        )
                        fem_main_3d.cleanup_sorting_workspace()
                        entry["status"] = "completed"
                        entry["snapshot_file"] = f"snapshots/{snapshot_path.name}"
                        entry["error"] = None
                        out_files.append(snapshot_path)
                        if not (isinstance(snapshot_raw, str) and snapshot_raw.strip()):
                            next_idx += 1
                        completed_batch += 1
                        self._write_json(pool_path, pool, rank=self.rank, comm=self.comm)
                        log_payload = {
                            "timestamp": run_stamp,
                            "guitar_id": sample_label,
                            "shape_name": shape_name,
                            "sampling_mode": sampling_mode,
                            "snapshot_file": str(snapshot_path.resolve()),
                            "elapsed_s": float(elapsed),
                            "num_modes": int(len(fom.get("freqs_hz", []))),
                            "solver_profile_used": solver_profile_used,
                        }
                        with open(log_path, "w", encoding="utf-8") as lf:
                            json.dump(log_payload, lf, indent=2)
                except Exception as exc:
                    traceback.print_exc()
                    entry["status"] = "error"
                    entry["error"] = str(exc)
                    error_batch += 1
                    if self.rank == 0:
                        with open(log_path, "w", encoding="utf-8") as lf:
                            json.dump(
                                {
                                    "timestamp": run_stamp,
                                    "guitar_id": sample_label,
                                    "shape_name": shape_name,
                                    "sampling_mode": sampling_mode,
                                    "status": "error",
                                    "error": str(exc),
                                    "solver_profile_used": solver_profile_used,
                                },
                                lf,
                                indent=2,
                            )
                finally:
                    _logging_reset_handlers()
                    processed += 1
                    self._write_json(pool_path, pool, rank=self.rank, comm=self.comm)
            self._last_collect_summary = {
                "shape": shape_name,
                "sampling": sampling_mode,
                "dry_run": False,
                "max_runs": int(max_runs),
                "processed_in_batch": processed,
                "completed_in_batch": completed_batch,
                "error_in_batch": error_batch,
                "skipped_success_in_batch": skipped_success,
                "force_rerun": bool(force_rerun),
                "pool_counts": self._pool_status_counts(pool),
            }
            return out_files

        for off, params in enumerate(grid):
            idx = start_idx + off
            try:
                _logging_reset_handlers()
                cfg = self._load_shape_base_config(shape_name)
                self._apply_parameters_to_config(cfg, params)
                cfg.setdefault("solver", {})["structural_only_diagnosis"] = False
                self._save_shape_base_config(shape_name, cfg)
                t0 = time.perf_counter()
                fom = fem_main_3d.run_fom_for_rom(cfg, num_modes=num_modes)
                elapsed = time.perf_counter() - t0
                snapshot_path = paths["snapshots"] / f"snapshot_{idx:04d}.npz"
                if snapshot_path.name == "snapshot_0000.npz":
                    raise RuntimeError(
                        "Refusing to overwrite Gold Reference snapshot_0000.npz. "
                        "Use a separate output path under ROM/<shape>/snapshots/."
                    )
                run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sample_label = f"{idx:04d}"
                log_path = logs_dir / f"simulation_guitar_{sample_label}_{run_stamp}.log"
                if self.rank == 0:
                    freqs_arr = np.array(fom["freqs_hz"], dtype=np.float64)
                    uniq_arr, part_arr = self._snapshot_telemetry_arrays(fom, freqs_arr)
                    np.savez(
                        snapshot_path,
                        params_json=json.dumps(params),
                        shape_name=shape_name,
                        sampling_mode=sampling_mode,
                        freqs_hz=freqs_arr,
                        eigvecs_real=np.real(fom["eigvecs"]).astype(np.float64),
                        uniqueness_scores=uniq_arr,
                        participation_ratios=part_arr,
                        elapsed_s=np.array([elapsed], dtype=np.float64),
                        sifter_stats_json=json.dumps(fom.get("sifter_stats", {}), indent=2),
                    )
                    fem_main_3d.cleanup_sorting_workspace()
                    out_files.append(snapshot_path)
                    with open(log_path, "w", encoding="utf-8") as lf:
                        json.dump(
                            {
                                "timestamp": run_stamp,
                                "guitar_id": sample_label,
                                "shape_name": shape_name,
                                "sampling_mode": sampling_mode,
                                "snapshot_file": str(snapshot_path.resolve()),
                                "elapsed_s": float(elapsed),
                                "num_modes": int(len(fom.get("freqs_hz", []))),
                            },
                            lf,
                            indent=2,
                        )
            finally:
                _logging_reset_handlers()
        self._last_collect_summary = {
            "shape": shape_name,
            "sampling": sampling_mode,
            "dry_run": False,
            "processed_in_batch": len(grid),
            "completed_in_batch": len(grid),
            "error_in_batch": 0,
        }
        return out_files

    def get_last_collect_summary(self) -> Dict:
        return dict(self._last_collect_summary)

    def build_basis(self, shape_name: str, energy: float = 0.999, max_rank: int = 128) -> Path:
        paths = self._shape_paths(shape_name)
        snapshots = sorted(paths["snapshots"].glob("snapshot_*.npz"))
        if not snapshots:
            raise RuntimeError(f"No snapshots found in {paths['snapshots']}. Run offline collection first.")

        columns = []
        for snap in snapshots:
            data = np.load(snap, allow_pickle=True)
            eig = data["eigvecs_real"]
            for k in range(eig.shape[1]):
                columns.append(eig[:, k])
        S = np.column_stack(columns)
        U, sigma, _ = np.linalg.svd(S, full_matrices=False)
        energy_curve = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
        r = int(np.searchsorted(energy_curve, energy) + 1)
        r = min(r, max_rank, U.shape[1])
        V = U[:, :r].astype(np.float64)

        paths["root"].mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            np.savez(
                paths["basis"],
                basis=V,
                singular_values=sigma.astype(np.float64),
                energy_curve=energy_curve.astype(np.float64),
                selected_rank=np.array([r], dtype=np.int32),
                snapshots_count=np.array([len(snapshots)], dtype=np.int32),
            )
        self._basis_cache[shape_name] = {
            "basis": V,
            "mtime": paths["basis"].stat().st_mtime,
            "path": paths["basis"],
        }
        return paths["basis"]

    @staticmethod
    def _project_petsc_mat(mat, V: np.ndarray) -> np.ndarray:
        r = V.shape[1]
        red = np.zeros((r, r), dtype=np.float64)
        vt = V.T
        x = mat.createVecRight()
        y = mat.createVecRight()
        for j in range(r):
            x.array[:] = V[:, j]
            mat.mult(x, y)
            red[:, j] = vt @ np.real(y.array)
        return red

    def _get_basis_cached(self, shape_name: str) -> np.ndarray:
        paths = self._shape_paths(shape_name)
        basis_path = paths["basis"]
        if not basis_path.exists():
            raise RuntimeError(f"Reduced basis missing: {basis_path}. Run basis generation first.")

        mtime = basis_path.stat().st_mtime
        cached = self._basis_cache.get(shape_name)
        if cached and cached.get("mtime") == mtime and cached.get("path") == basis_path:
            return cached["basis"]

        basis_data = np.load(basis_path)
        V = basis_data["basis"]
        self._basis_cache[shape_name] = {"basis": V, "mtime": mtime, "path": basis_path}
        return V

    def solve_online(self, shape_name: str, params: Dict, nev: int = 3) -> Dict:
        # Basis is cached in memory per shape for fast UI switching.
        V = self._get_basis_cached(shape_name)

        cfg = self._load_shape_base_config(shape_name)
        self._apply_parameters_to_config(cfg, params)

        t0 = time.perf_counter()
        _, _, A, M = fem_main_3d.assemble_coupled_operators_for_rom(cfg)
        Ar = self._project_petsc_mat(A, V)
        Mr = self._project_petsc_mat(M, V)
        lam, _ = np.linalg.eig(np.linalg.solve(Mr, Ar))
        elapsed = time.perf_counter() - t0

        lam = np.real(lam)
        lam = lam[np.isfinite(lam)]
        lam = lam[lam > 1e-12]
        lam.sort()
        lam = lam[:nev]
        freqs = np.sqrt(lam) / (2.0 * np.pi)
        return {"freqs_hz": freqs.tolist(), "elapsed_s": elapsed, "nev": int(len(freqs))}

    def compare(self, shape_name: str, params: Dict, nev: int = 3, fom_modes: int = 15) -> Dict:
        cfg = self._load_shape_base_config(shape_name)
        self._apply_parameters_to_config(cfg, params)
        self._save_shape_base_config(shape_name, cfg)

        t0 = time.perf_counter()
        fom = fem_main_3d.run_fom_for_rom(cfg, num_modes=max(fom_modes, nev))
        t_fom = time.perf_counter() - t0

        rom = self.solve_online(shape_name, params=params, nev=nev)
        n = min(len(fom["freqs_hz"]), len(rom["freqs_hz"]), nev)
        f_fom = np.array(fom["freqs_hz"][:n], dtype=np.float64)
        f_rom = np.array(rom["freqs_hz"][:n], dtype=np.float64)
        err_pct = np.where(np.abs(f_fom) > 1e-12, 100.0 * np.abs(f_rom - f_fom) / np.abs(f_fom), np.nan)
        return {
            "shape": shape_name,
            "params": params,
            "fom_time_s": t_fom,
            "rom_time_s": rom["elapsed_s"],
            "speedup": (t_fom / rom["elapsed_s"]) if rom["elapsed_s"] > 0 else np.inf,
            "fom_freqs_hz": f_fom.tolist(),
            "rom_freqs_hz": f_rom.tolist(),
            "error_pct": err_pct.tolist(),
        }

