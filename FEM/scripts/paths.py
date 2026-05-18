"""
Central path configuration for repo I/O and VirtualBox shared-folder exports.

Set ``SHARED_HOST_DIR`` in the environment to override the default mount
(``/media/sf_gmar``). Large artifacts are written under a shape-scoped hierarchy::

    {SHARED_HOST_DIR}/{shape_name}/{asset_category}/...

so thousands of files stay partitioned (e.g. ``classic/plots/``, ``dreadnought/rom_data/``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_SHAPE_NAME = "classic"

# Canonical asset category folder names on the shared host.
ASSET_AUDIO = "audio"
ASSET_PLOTS = "plots"
ASSET_ROM_DATA = "rom_data"

_SHAPE_SEGMENT_RE = re.compile(r"[^a-z0-9_-]+")


def repo_root() -> Path:
    """Project root (directory named ``final-project``)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if parent.name == "final-project":
            return parent
    return here.parents[2]


REPO_ROOT = repo_root()

SHARED_HOST_DIR = Path(os.environ.get("SHARED_HOST_DIR", "/media/sf_gmar")).expanduser()

# Repo-local paths (small / scratch)
FEM_SORTING_DIR = REPO_ROOT / "FEM" / "SORTING"
FEM_RESULTS_PLOTS_DIR = REPO_ROOT / "FEM" / "results" / "plots"
FEM_LAB_RESULTS_DIR = REPO_ROOT / "FEM" / "results" / "LAB_RESULTS"
ROM_CLASSIC_SNAPSHOTS_DIR = REPO_ROOT / "ROM" / "classic" / "snapshots"


def normalize_shape_name(shape_name: str) -> str:
    """Filesystem-safe shape segment (lowercase alnum, underscore, hyphen)."""
    raw = str(shape_name or DEFAULT_SHAPE_NAME).strip().lower()
    cleaned = _SHAPE_SEGMENT_RE.sub("_", raw).strip("_")
    return cleaned or DEFAULT_SHAPE_NAME


def get_shared_dir(shape_name: str, asset_category: str) -> Path:
    """
    Resolve ``{SHARED_HOST_DIR}/{shape_name}/{asset_category}/`` and create it.

    Parameters
    ----------
    shape_name:
        ROM / guitar shape key (e.g. ``classic``, ``dreadnought``).
    asset_category:
        Subfolder for one asset class (e.g. ``audio``, ``plots``, ``rom_data``).
    """
    shape_seg = normalize_shape_name(shape_name)
    cat_seg = str(asset_category or "").strip().lower().replace("\\", "/").strip("/")
    if not cat_seg:
        raise ValueError("asset_category must be a non-empty string (e.g. 'plots', 'rom_data').")
    directory = SHARED_HOST_DIR / shape_seg / cat_seg
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def shared_asset_path(
    shape_name: str,
    asset_category: str,
    filename: str,
) -> Path:
    """Full path to a file under the shape-specific asset category directory."""
    name = Path(filename).name
    if not name:
        raise ValueError("filename must be a non-empty basename or path with a name component.")
    return get_shared_dir(shape_name, asset_category) / name


def shared_audio_path(filename: str, shape_name: str = DEFAULT_SHAPE_NAME) -> Path:
    return shared_asset_path(shape_name, ASSET_AUDIO, filename)


def shared_plot_path(filename: str, shape_name: str = DEFAULT_SHAPE_NAME) -> Path:
    return shared_asset_path(shape_name, ASSET_PLOTS, filename)


def shared_rom_csv_path(filename: str, shape_name: str = DEFAULT_SHAPE_NAME) -> Path:
    return shared_asset_path(shape_name, ASSET_ROM_DATA, filename)


def shared_rom_npz_path(filename: str, shape_name: str = DEFAULT_SHAPE_NAME) -> Path:
    """CSR ROM archives on the shared host (same ``rom_data`` tree as CSV exports)."""
    return shared_asset_path(shape_name, ASSET_ROM_DATA, filename)


def _rewrite_legacy_flat_dirs(path: Path, shape_name: str) -> Path:
    """
    Map pre-refactor flat shared paths (``guitar_audio``, ``pipeline_exports/...``)
    into ``{shape}/{category}/``.
    """
    sh = normalize_shape_name(shape_name)
    s = path.as_posix()
    replacements = (
        ("/pipeline_exports/selection_plots/", f"/{sh}/{ASSET_PLOTS}/"),
        ("/pipeline_exports/rom_csv/", f"/{sh}/{ASSET_ROM_DATA}/"),
        ("/pipeline_exports/", f"/{sh}/"),
        ("/guitar_audio/", f"/{sh}/{ASSET_AUDIO}/"),
    )
    for old, new in replacements:
        if old in s:
            s = s.replace(old, new)
    return Path(s)


def resolve_shared_path(
    path: str | Path,
    shape_name: str = DEFAULT_SHAPE_NAME,
) -> Path:
    """
    Expand ``{SHARED_HOST}`` / legacy ``/media/sf_gmar`` prefixes and normalize to the
    hierarchical layout when old flat directory names are present.
    """
    s = str(path).replace("\\", "/")
    token = "{SHARED_HOST}"
    if token in s:
        s = s.replace(token, str(SHARED_HOST_DIR.as_posix()))
    legacy_prefix = "/media/sf_gmar"
    if s.startswith(legacy_prefix):
        s = str(SHARED_HOST_DIR.as_posix()) + s[len(legacy_prefix) :]
    resolved = _rewrite_legacy_flat_dirs(Path(s), shape_name)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def infer_shape_from_pool_path(pool_path: Path | None) -> str:
    """Derive shape name from ``ROM/<shape>/lhs_pool.json`` when possible."""
    if pool_path is None:
        return DEFAULT_SHAPE_NAME
    parts = Path(pool_path).resolve().parts
    if "ROM" in parts:
        idx = parts.index("ROM")
        if idx + 1 < len(parts):
            return normalize_shape_name(parts[idx + 1])
    return DEFAULT_SHAPE_NAME
