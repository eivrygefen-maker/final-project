import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = BASE_DIR / "FEM" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))
import fem_main_3d


class ROMManager:
    def __init__(self, base_dir: Optional[Path] = None, shapes_config_path: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
        self.rom_root = self.base_dir / "ROM_DATA"
        self.rom_root.mkdir(parents=True, exist_ok=True)
        self.shapes_config_path = (
            Path(shapes_config_path)
            if shapes_config_path
            else self.base_dir / "FEM" / "configs" / "rom_shapes.json"
        )
        self.shapes = self._load_shapes_config()

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
        shape_root = self.rom_root / shape_name
        snapshots_dir = shape_root / "snapshots"
        basis_path = shape_root / "reduced_basis.npz"
        return {"root": shape_root, "snapshots": snapshots_dir, "basis": basis_path}

    def _load_shape_base_config(self, shape_name: str) -> Dict:
        shape_cfg = self.shapes[shape_name]
        config_path = self.base_dir / shape_cfg["base_config"]
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
    def _grid_from_sweep(sweep: Dict[str, List]) -> List[Dict]:
        keys = sorted(sweep.keys())
        vals = [sweep[k] for k in keys]
        combos = []
        for row in product(*vals):
            combos.append({k: v for k, v in zip(keys, row)})
        return combos

    def collect_snapshots(self, shape_name: str, num_modes: int = 6) -> List[Path]:
        shape_cfg = self.shapes[shape_name]
        paths = self._shape_paths(shape_name)
        paths["snapshots"].mkdir(parents=True, exist_ok=True)

        sweep_cfg = shape_cfg.get("parameter_sweep", {})
        grid = self._grid_from_sweep(sweep_cfg) if sweep_cfg else [{}]
        out_files: List[Path] = []

        for idx, params in enumerate(grid):
            cfg = self._load_shape_base_config(shape_name)
            for k, v in params.items():
                self._set_nested(cfg, k, v)
            t0 = time.perf_counter()
            fom = fem_main_3d.run_fom_for_rom(cfg, num_modes=num_modes)
            elapsed = time.perf_counter() - t0
            snapshot_path = paths["snapshots"] / f"snapshot_{idx:04d}.npz"
            np.savez(
                snapshot_path,
                params_json=json.dumps(params),
                freqs_hz=np.array(fom["freqs_hz"], dtype=np.float64),
                eigvecs_real=np.real(fom["eigvecs"]).astype(np.float64),
                elapsed_s=np.array([elapsed], dtype=np.float64),
            )
            out_files.append(snapshot_path)
        return out_files

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
        np.savez(
            paths["basis"],
            basis=V,
            singular_values=sigma.astype(np.float64),
            energy_curve=energy_curve.astype(np.float64),
            selected_rank=np.array([r], dtype=np.int32),
            snapshots_count=np.array([len(snapshots)], dtype=np.int32),
        )
        return paths["basis"]

    @staticmethod
    def _project_petsc_mat(mat, V: np.ndarray) -> np.ndarray:
        r = V.shape[1]
        red = np.zeros((r, r), dtype=np.float64)
        x = mat.createVecRight()
        y = mat.createVecRight()
        for j in range(r):
            x.array[:] = V[:, j]
            mat.mult(x, y)
            red[:, j] = V.T @ np.real(y.array)
        return red

    def solve_online(self, shape_name: str, params: Dict, nev: int = 3) -> Dict:
        paths = self._shape_paths(shape_name)
        if not paths["basis"].exists():
            raise RuntimeError(f"Reduced basis missing: {paths['basis']}. Run basis generation first.")
        basis_data = np.load(paths["basis"])
        V = basis_data["basis"]

        cfg = self._load_shape_base_config(shape_name)
        for k, v in params.items():
            self._set_nested(cfg, k, v)

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

    def compare(self, shape_name: str, params: Dict, nev: int = 3, fom_modes: int = 6) -> Dict:
        cfg = self._load_shape_base_config(shape_name)
        for k, v in params.items():
            self._set_nested(cfg, k, v)

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

