"""
Central path configuration for repo I/O and VirtualBox shared-folder exports.

Set ``SHARED_HOST_DIR`` in the environment to override the default mount
(``/media/sf_gmar``). Large artifacts (WAV, PNG exports, merged CSV dumps) should
prefer ``shared_*`` helpers so the VM disk is not filled.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Project root (directory named ``final-project``)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if parent.name == "final-project":
            return parent
    return here.parents[2]


REPO_ROOT = repo_root()

SHARED_HOST_DIR = Path(os.environ.get("SHARED_HOST_DIR", "/media/sf_gmar")).expanduser()

# Subdirectories on the shared host mount
SHARED_AUDIO_DIR = SHARED_HOST_DIR / "guitar_audio"
SHARED_EXPORTS_DIR = SHARED_HOST_DIR / "pipeline_exports"
SHARED_PLOTS_DIR = SHARED_EXPORTS_DIR / "selection_plots"
SHARED_ROM_CSV_DIR = SHARED_EXPORTS_DIR / "rom_csv"

# Repo-local paths (small / scratch)
FEM_SORTING_DIR = REPO_ROOT / "FEM" / "SORTING"
FEM_RESULTS_PLOTS_DIR = REPO_ROOT / "FEM" / "results" / "plots"
FEM_LAB_RESULTS_DIR = REPO_ROOT / "FEM" / "results" / "LAB_RESULTS"
ROM_CLASSIC_SNAPSHOTS_DIR = REPO_ROOT / "ROM" / "classic" / "snapshots"


def ensure_shared_dirs() -> None:
    """Create standard shared export folders if the mount is writable."""
    for d in (SHARED_AUDIO_DIR, SHARED_EXPORTS_DIR, SHARED_PLOTS_DIR, SHARED_ROM_CSV_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def resolve_shared_path(path: str | Path) -> Path:
    """
    Map legacy hardcoded ``/media/sf_gmar/...`` paths and ``{SHARED_HOST}`` tokens
  to the configured ``SHARED_HOST_DIR``.
    """
    s = str(path).replace("\\", "/")
    token = "{SHARED_HOST}"
    if token in s:
        s = s.replace(token, str(SHARED_HOST_DIR.as_posix()))
    legacy_prefix = "/media/sf_gmar"
    if s.startswith(legacy_prefix):
        s = str(SHARED_HOST_DIR.as_posix()) + s[len(legacy_prefix) :]
    return Path(s)


def shared_audio_path(name: str) -> Path:
    ensure_shared_dirs()
    return SHARED_AUDIO_DIR / name


def shared_plot_path(name: str) -> Path:
    ensure_shared_dirs()
    return SHARED_PLOTS_DIR / name


def shared_rom_csv_path(name: str) -> Path:
    ensure_shared_dirs()
    return SHARED_ROM_CSV_DIR / name
