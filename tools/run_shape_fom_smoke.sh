#!/usr/bin/env bash
# Lightweight shape FOM/M4 smoke — no FEM solve, no STK, no WAV.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0
ok() { echo "OK  $1"; PASS=$((PASS + 1)); }
bad() { echo "FAIL $1"; FAIL=$((FAIL + 1)); }

echo "== Shape FOM pipeline smoke =="

M4_SCRIPT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py"
REGISTRY="${REPO_ROOT}/FEM/scripts/m4_shape_registry.py"
GEN_SCRIPT="${REPO_ROOT}/tools/generate_shape_lhs_pool.py"
OVERNIGHT="${REPO_ROOT}/tools/run_shape_fom_overnight_batch.sh"
ROM_SHAPES="${REPO_ROOT}/FEM/configs/rom_shapes.json"
POLICY_DOC="${REPO_ROOT}/docs/m4_acoustic_opening_policy.md"

for f in \
  "${REGISTRY}" \
  "${GEN_SCRIPT}" \
  "${OVERNIGHT}" \
  "${M4_SCRIPT}" \
  "${ROM_SHAPES}" \
  docs/classic_fom_rom_pipeline_report.md
do
  if [[ -f "${f}" ]]; then
    ok "present ${f#${REPO_ROOT}/}"
  else
    bad "missing ${f#${REPO_ROOT}/}"
  fi
done

if grep -q -- '--shape' "${M4_SCRIPT}"; then
  ok "run_m4_production_pipeline.py accepts --shape"
else
  bad "run_m4_production_pipeline.py missing --shape"
fi

python3 - <<PY
import sys
from pathlib import Path
repo = Path("${REPO_ROOT}")
sys.path.insert(0, str(repo / "FEM" / "scripts"))
from m4_shape_registry import registered_shape_keys, resolve_shape_config

for key in ("classic", "box", "acoustic"):
    cfg = resolve_shape_config(key)
    assert cfg.lhs_pool_rel == f"ROM/{key}/lhs_pool.json", (key, cfg.lhs_pool_rel)
    assert cfg.shared_export_key == key
    assert cfg.geometry_shape_type in ("Classical", "Box", "Acoustic")
    print(f"registry_ok shape={key} lhs={cfg.lhs_pool_rel} geom={cfg.geometry_shape_type}")
print("registered=", registered_shape_keys())
PY
ok "shape registry resolves classic/box/acoustic"

python3 - <<PY
import sys
from pathlib import Path
repo = Path("${REPO_ROOT}")
sys.path.insert(0, str(repo / "FEM" / "scripts"))
sys.path.insert(0, str(repo / "FEM" / "experiments" / "active_domain_validation" / "physics_integrity" / "scripts"))
from m4_shape_registry import resolve_shape_config, REGISTERED_SCOUT_DENSITY_POLICIES
from v2_b3_m4_scout_discovery_diagnostics import density_result_path, SCOUT_DISCOVERY_REL
from v2_b3_m4_scout_intrinsic_coverage import is_registered_scout_density_policy

expected = {
    "classic": "intrinsic_discovered_modes_v1",
    "box": "box_discovered_modes_v2",
    "acoustic": "acoustic_discovered_modes_v1",
}
for key, policy in expected.items():
    cfg = resolve_shape_config(key)
    assert cfg.scout_density_policy == policy, (key, cfg.scout_density_policy)
    assert is_registered_scout_density_policy(policy)
    print(f"scout_density_policy_ok shape={key} policy={policy}")
assert SCOUT_DISCOVERY_REL == "scout/discovery"
assert density_result_path(repo / "run").name == "density_result.json"
assert len(REGISTERED_SCOUT_DENSITY_POLICIES) == 3
from v2_b3_m4_scout_intrinsic_coverage import COVERAGE_POLICY_BOX_V1
assert is_registered_scout_density_policy(COVERAGE_POLICY_BOX_V1)
PY
ok "scout density policies + failure artifact path discoverable"

python3 "${GEN_SCRIPT}" --shape box --count 100 --dry-run
python3 "${GEN_SCRIPT}" --shape acoustic --count 100 --dry-run
ok "box/acoustic LHS dry-run 100 samples"

for shape in box acoustic; do
  pool="${REPO_ROOT}/ROM/${shape}/lhs_pool.json"
  if [[ ! -f "${pool}" ]]; then
    python3 "${GEN_SCRIPT}" --shape "${shape}" --count 100
    ok "generated ${pool#${REPO_ROOT}/}"
  fi
done

python3 "${GEN_SCRIPT}" --shape classic --dry-run 2>/dev/null && ok "classic LHS preserved (no regen)" || ok "classic LHS skip/regen policy"

python3 "${M4_SCRIPT}" --shape box --max-samples 1 --dry-run >/tmp/shape_fom_smoke_dry.log 2>&1 \
  && ok "M4 dry-run --shape box" \
  || bad "M4 dry-run --shape box"

python3 "${M4_SCRIPT}" --lhs-json ROM/classic/lhs_pool.json --max-samples 1 --dry-run >/tmp/shape_fom_smoke_classic_legacy.log 2>&1 \
  && ok "legacy --lhs-json classic dry-run" \
  || bad "legacy --lhs-json classic dry-run"

python3 - <<PY
import sys
from pathlib import Path
repo = Path("${REPO_ROOT}")
sys.path.insert(0, str(repo / "FEM" / "scripts"))
from m4_shape_registry import resolve_shape_config
for key in ("classic", "box", "acoustic"):
    pol = resolve_shape_config(key).acoustic_opening_policy()
    assert pol.get("requires_aperture_mask") is True
    assert pol.get("aperture_selection_method")
    print(f"acoustic_policy_ok shape={key} has_soundhole={pol.get('has_soundhole')}")
PY
ok "acoustic opening policy discoverable"

if [[ -f "${POLICY_DOC}" ]]; then
  ok "acoustic opening policy doc"
else
  bad "missing docs/m4_acoustic_opening_policy.md"
fi

# Fail only on executable STK/audio tool references — not comments or FOM field names
# (e.g. "No STK" in comments, last_audio_coupling_computed_count in LHS schema).
_stk_exec_pattern='build_app_stk_note_library|run_app_stk|app_stk_note_cache|stk_pgsm|stk_app_audio'
_stk_hits=""
for _f in "${OVERNIGHT}" "${GEN_SCRIPT}"; do
  if [[ -f "${_f}" ]]; then
    _line_hits="$(grep -nE "${_stk_exec_pattern}" "${_f}" 2>/dev/null || true)"
    if [[ -n "${_line_hits}" ]]; then
      _stk_hits="${_stk_hits}${_f}:${_line_hits}"$'\n'
    fi
  fi
done
if [[ -z "${_stk_hits}" ]]; then
  ok "no STK/audio executable refs in FOM overnight/generator scripts"
else
  bad "STK/audio executable reference in FOM scripts (see grep hits above)"
fi

GEOM_CONTRACT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_geometry_numeric_contract_test.py"
if python3 "${GEOM_CONTRACT}" >/tmp/shape_fom_smoke_geom_contract.log 2>&1; then
  ok "geometry numeric/metadata contract (box shape_type)"
else
  bad "geometry numeric contract failed (see /tmp/shape_fom_smoke_geom_contract.log)"
fi

REUSE_INTEGRITY_TEST="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_reuse_integrity_test.py"
if python3 "${REUSE_INTEGRITY_TEST}" >/tmp/shape_fom_smoke_reuse_integrity.log 2>&1; then
  ok "M4 reuse integrity (stale PASS / terminal mismatch)"
else
  bad "reuse integrity tests failed (see /tmp/shape_fom_smoke_reuse_integrity.log)"
fi

MESH_SHAPE_TEST="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_mesh_shape_contract_test.py"
SHAPE_CONTEXT_TEST="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_shape_context_test.py"
if python3 "${MESH_SHAPE_TEST}" >/tmp/shape_fom_smoke_mesh_shape.log 2>&1; then
  ok "mesh shape manifest + scout/prod consistency"
else
  bad "mesh shape contract tests failed (see /tmp/shape_fom_smoke_mesh_shape.log)"
fi
SHAPE_CONTEXT_TEST="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_shape_context_test.py"
if python3 "${SHAPE_CONTEXT_TEST}" >/tmp/shape_fom_smoke_shape_context.log 2>&1; then
  ok "unified ShapeContext"
else
  bad "shape context tests failed (see /tmp/shape_fom_smoke_shape_context.log)"
fi
STAGE_ARTIFACT_TEST="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_stage_artifact_contract_test.py"
if python3 "${STAGE_ARTIFACT_TEST}" >/tmp/shape_fom_smoke_stage_artifact.log 2>&1; then
  ok "scout stage artifact contract (preview JSON ownership)"
else
  bad "stage artifact contract tests failed (see /tmp/shape_fom_smoke_stage_artifact.log)"
fi
INSPECT_SCRIPT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/inspect_shape_mesh_aperture.py"
BOX_FOM_TEST="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_fom_validation_test.py"
if [[ -f "${INSPECT_SCRIPT}" ]]; then
  ok "present inspect_shape_mesh_aperture.py"
else
  bad "missing inspect_shape_mesh_aperture.py"
fi
if python3 "${BOX_FOM_TEST}" >/tmp/shape_fom_smoke_box_fom_validation.log 2>&1; then
  ok "BOX FOM validation (terminal promote, mesh inspect, full-clean)"
else
  bad "BOX FOM validation tests failed (see /tmp/shape_fom_smoke_box_fom_validation.log)"
fi

echo ""
echo "Smoke: ${PASS} passed, ${FAIL} failed"
[[ "${FAIL}" -eq 0 ]]
