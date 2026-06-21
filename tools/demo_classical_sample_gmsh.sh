#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAMPLE_ID="sample_000"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/classic_gmsh_demo_${SAMPLE_ID}.XXXXXX")"

cleanup() {
  if [[ -n "${TMP_ROOT:-}" && -d "${TMP_ROOT}" ]]; then
    rm -rf "${TMP_ROOT}"
  fi
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

if ! command -v gmsh >/dev/null 2>&1; then
  echo "ERROR: gmsh command not found. Install GMSH or add it to PATH before running this demo." >&2
  exit 1
fi

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python3/python command not found. Python is required to run the GMSH geometry script." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
try:
    import gmsh  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ERROR: Python gmsh module is unavailable: {exc}")
PY

mkdir -p "${TMP_ROOT}/FEM/configs" "${TMP_ROOT}/FEM/mesh"
cp -R "${REPO_ROOT}/FEM/geometry" "${TMP_ROOT}/FEM/"

DEMO_CONFIG="${TMP_ROOT}/FEM/configs/guitar_3d.json"
DEMO_MESH="${TMP_ROOT}/FEM/mesh/display_mesh.msh"

REPO_ROOT="${REPO_ROOT}" TMP_ROOT="${TMP_ROOT}" SAMPLE_ID="${SAMPLE_ID}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
tmp = Path(os.environ["TMP_ROOT"])
sample_id = os.environ["SAMPLE_ID"]

lhs_path = repo / "ROM" / "classic" / "lhs_pool.json"
base_config_path = repo / "FEM" / "configs" / "guitar_3d.json"
woods_path = repo / "FEM" / "materials" / "woods_ortho.json"
out_path = tmp / "FEM" / "configs" / "guitar_3d.json"

with lhs_path.open("r", encoding="utf-8") as f:
    lhs = json.load(f)
with base_config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)
with woods_path.open("r", encoding="utf-8") as f:
    woods = json.load(f)

entry = next((e for e in lhs.get("entries", []) if e.get("id") == sample_id), None)
if entry is None:
    raise SystemExit(f"ERROR: {sample_id} not found in {lhs_path}")
if str(entry.get("status", "")).upper() != "COMPLETED":
    raise SystemExit(f"ERROR: {sample_id} is not marked COMPLETED in {lhs_path}")

params = entry.get("parameters") or {}
wood_aliases = {
    "spruce": ("spruce_sitka", "Sitka Spruce"),
    "rosewood": ("rosewood_indian", "Indian Rosewood"),
    "mahogany": ("mahogany_honduran", "Honduran Mahogany"),
    "cedar": ("cedar_western", "Western Red Cedar"),
    "maple": ("maple_hard", "Maple"),
}

def material_for(short_id):
    key, name = wood_aliases.get(str(short_id), (str(short_id), str(short_id)))
    raw = dict(woods[key])
    return {
        "name": name,
        "density": raw["rho"],
        "E_L": raw["E_L"],
        "E_T": raw["E_T"],
        "E_R": raw["E_R"],
        "nu_LT": raw["nu_LT"],
        "nu_LR": raw["nu_LR"],
        "nu_TR": raw["nu_TR"],
        "G_LT": raw["G_LT"],
        "G_LR": raw["G_LR"],
        "G_TR": raw["G_TR"],
        "q_min": raw["q_min"],
        "q_max": raw["q_max"],
        "color": raw["color"],
        "wood_id": str(short_id),
    }

length = float(params["geometry.length"])
width = float(params["geometry.width"])
depth = float(params["geometry.depth"])
top_t = float(params["geometry.top_thickness"])
hole_r = float(params["geometry.hole_radius"])
back_t = float(params.get("geometry.back_thickness", top_t * 1.1))
top_wood = str(params["top_wood_id"])
back_wood = str(params["back_wood_id"])

geometry = dict(config.get("geometry") or {})
geometry.update(
    {
        "shape_type": "Classical",
        "length": length,
        "width": width,
        "depth": depth,
        "thickness": top_t,
        "top_thickness": top_t,
        "back_thickness": back_t,
        "hole_radius": hole_r,
        "lower_bout": width,
        "upper_bout": width * 0.7567567568,
        "waist": width * 0.6216216216,
        "soundhole_y": 0.0,
        "soundhole_x": 0.0,
        "soundhole_from_neck_ratio": 0.5,
        "bridge_x": -0.12,
        "top_wood_id": top_wood,
        "back_wood_id": back_wood,
    }
)
config["geometry"] = geometry
config["materials"] = {
    "top": material_for(top_wood),
    "back": material_for(back_wood),
    "air": {"density": 1.204, "speed_of_sound": 343.0},
}
config.setdefault("solver", {})["mesh_file"] = str(tmp / "FEM" / "mesh" / "guitar_3d.msh")

out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(f"Sample: {sample_id}")
print(f"Source: {lhs_path}")
print(
    "Parameters: "
    f"L={length:.3f} W={width:.3f} D={depth:.3f} "
    f"top_t={top_t:.4f} hole_r={hole_r:.3f} "
    f"top={top_wood} back={back_wood}"
)
print(f"Temporary config: {out_path}")
PY

echo "Repository root: ${REPO_ROOT}"
echo "Temporary demo folder: ${TMP_ROOT}"
echo "Active website config is not modified: ${REPO_ROOT}/FEM/configs/guitar_3d.json"
echo "Active website display mesh is not modified: ${REPO_ROOT}/FEM/mesh/display_mesh.msh"
echo "Generating temporary GMSH display mesh..."

(
  cd "${TMP_ROOT}"
  FEM_ALLOW_DISPLAY=1 "${PYTHON_BIN}" "FEM/geometry/build_3d_guitar.py" --config "${DEMO_CONFIG}" -nopopup
)

if [[ ! -s "${DEMO_MESH}" ]]; then
  echo "ERROR: expected temporary mesh was not created: ${DEMO_MESH}" >&2
  exit 1
fi

echo "Opening GMSH GUI for ${SAMPLE_ID}: ${DEMO_MESH}"
echo "Close the GMSH window to remove temporary demo files."
gmsh "${DEMO_MESH}"
