#!/usr/bin/env bash
# M4 geometry-corrected dataset migration plan (DO NOT run automatically).
# Review each section before executing on the VM.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
cd "$REPO_ROOT"

echo "=== M4 geometry-corrected dataset reset (manual) ==="
echo "Repo: $REPO_ROOT"
echo "Dataset marker: m4_geometry_corrected_v1"
echo ""
echo "This script prints commands only unless you pass --execute"
EXECUTE="${1:-}"

if [[ "$EXECUTE" != "--execute" ]]; then
  cat <<'EOF'
Planned steps (run with --execute after review):

1) Archive root-cause reports (keep):
   mkdir -p FEM/experiments/active_domain_validation/physics_integrity/docs/archive/pre_geometryfix_v1
   cp FEM/experiments/active_domain_validation/physics_integrity/docs/M4_OPERATOR_*.json docs/archive/pre_geometryfix_v1/ 2>/dev/null || true
   cp FEM/experiments/active_domain_validation/physics_integrity/docs/M4_GEOMETRY_AUDIO_VALIDATION.json docs/archive/pre_geometryfix_v1/ 2>/dev/null || true

2) Mark lhs_pool dataset version (edit JSON manually or use jq):
   # Set top-level "dataset_version": "m4_geometry_corrected_v1"
   # Reset entries[0..35].status to "PENDING", clear last_run_id if invalid
   # Reset sample_036 entry completely to PENDING (do not resume old checkpoint)

3) Quarantine invalid ROM (prevents accidental load):
   mv ROM/classic ROM/classic_INVALID_pre_geometryfix_v1
   mkdir -p ROM/m4_geometry_corrected_v1
   # Train new ROM only after >=3 corrected samples complete

4) Delete invalid per-sample heavy artifacts (000-035, 036 partial):
   GUITARS="FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
   for sid in $(seq -f 'sample_%03g' 0 35); do
     for run in "$GUITARS/$sid/runs"/*; do
       [[ -d "$run/aggregation" ]] && rm -rf "$run/aggregation"
       [[ -d "$run/rom" ]] && rm -rf "$run/rom"
       [[ -d "$run/workers" ]] && rm -rf "$run/workers"
       # Keep lprod/checkpoint only for before/after audit if needed; delete for clean rerun:
       # rm -rf "$run/lprod" "$run/scout"
     done
   done
   # sample_036: remove entire partial run tree
   rm -rf "$GUITARS/sample_036/runs"/*

5) Verify old ROM cannot load:
   export M4_REQUIRED_ROM_DATASET_VERSION=m4_geometry_corrected_v1
   # run_m4_production_pipeline.py --run-rom-prepredict should SKIPPED until new ROM exists

EOF
  exit 0
fi

echo "Execute mode not fully automated — perform steps 1-5 manually with backups."
exit 1
