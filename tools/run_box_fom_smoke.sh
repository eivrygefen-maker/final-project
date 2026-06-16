#!/usr/bin/env bash
# Lightweight BOX FOM/M4 pipeline smoke — no FEM solve, no STK, no WAV.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0
ok() { echo "OK  $1"; PASS=$((PASS + 1)); }
bad() { echo "FAIL $1"; FAIL=$((FAIL + 1)); }

echo "== BOX FOM pipeline smoke =="

M4_SCRIPT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py"
LHS_POOL="${REPO_ROOT}/ROM/box/lhs_pool.json"
ROM_SHAPES="${REPO_ROOT}/FEM/configs/rom_shapes.json"

for f in \
  tools/generate_box_lhs_pool.py \
  tools/run_box_fom_overnight_batch.sh \
  "${M4_SCRIPT}" \
  "${LHS_POOL}"
do
  if [[ -f "${f}" ]]; then
    ok "present ${f#${REPO_ROOT}/}"
  else
    bad "missing ${f#${REPO_ROOT}/}"
  fi
done

if grep -q '"box"' "${ROM_SHAPES}" && grep -q '"Box"' "${ROM_SHAPES}"; then
  ok "rom_shapes.json registers box shape"
else
  bad "rom_shapes.json missing box entry"
fi

if grep -q 'COUNT="${COUNT:-40}"' "${REPO_ROOT}/tools/run_box_fom_overnight_batch.sh"; then
  ok "overnight default count 40"
else
  bad "overnight default count 40"
fi

if grep -q 'WORKERS="${WORKERS:-3}"' "${REPO_ROOT}/tools/run_box_fom_overnight_batch.sh"; then
  ok "overnight default 3 FEM workers"
else
  bad "overnight default 3 FEM workers"
fi

if grep -q 'run_m4_production_pipeline.py' "${REPO_ROOT}/tools/run_box_fom_overnight_batch.sh"; then
  ok "overnight calls M4 production pipeline"
else
  bad "overnight calls M4 production pipeline"
fi

if grep -q 'build_app_stk_note_library.py' "${REPO_ROOT}/tools/run_box_fom_overnight_batch.sh"; then
  bad "overnight script must not call STK build tool"
else
  ok "overnight avoids STK build tool"
fi

if grep -q 'app_stk_note_cache' "${REPO_ROOT}/tools/run_box_fom_overnight_batch.sh"; then
  bad "overnight script must not reference audio cache"
else
  ok "overnight avoids audio cache paths"
fi

if grep -q 'BOX_FOM_ISOLATION_OK' "${REPO_ROOT}/tools/run_box_fom_overnight_batch.sh"; then
  ok "overnight isolation marker"
else
  bad "overnight isolation marker"
fi

if grep -q 'ROM/box/lhs_pool.json' "${REPO_ROOT}/tools/run_box_fom_overnight_batch.sh"; then
  ok "overnight uses ROM/box LHS"
else
  bad "overnight uses ROM/box LHS"
fi

if grep -q '/box/' "${REPO_ROOT}/tools/run_box_fom_overnight_batch.sh"; then
  ok "overnight box path segments"
else
  bad "overnight box path segments"
fi

python3 - <<'PY'
import json
import sys
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root / "tools"))
from generate_box_lhs_pool import (
    DEFAULT_COUNT,
    DEFAULT_FOM_RUN_ID_SUFFIX,
    box_fom_run_root,
    box_sample_id,
    build_pool_document,
    is_box_fom_sample_completed,
)

assert DEFAULT_COUNT == 40
assert box_sample_id(0) == "box_sample_000"
assert box_sample_id(39) == "box_sample_039"

doc = build_pool_document(count=40, seed=20260616, existing={}, force=False)
assert len(doc["entries"]) == 40
assert doc["entries"][0]["parameters"]["geometry.shape_type"] == "Box"
assert "geometry.back_thickness" in doc["entries"][0]["parameters"]

run_root = box_fom_run_root(root, "box_sample_000", DEFAULT_FOM_RUN_ID_SUFFIX)
assert "/guitars/box_sample_000/runs/box_sample_000_" in run_root.as_posix()

status = is_box_fom_sample_completed(root, "box_sample_missing", run_id_suffix=DEFAULT_FOM_RUN_ID_SUFFIX)
assert status.get("ready") is False

shapes = json.loads((root / "FEM/configs/rom_shapes.json").read_text(encoding="utf-8"))
assert "box" in shapes.get("shapes", {})

print("box_fom_smoke_ok")
PY
if [[ $? -eq 0 ]]; then ok "python FOM LHS / path logic"; else bad "python FOM LHS / path logic"; fi

if python3 tools/generate_box_lhs_pool.py --dry-run --count 40 >/dev/null 2>&1; then
  ok "generate_box_lhs_pool dry-run"
else
  bad "generate_box_lhs_pool dry-run"
fi

echo ""
echo "Passed: ${PASS}  Failed: ${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
echo "BOX FOM smoke complete."
