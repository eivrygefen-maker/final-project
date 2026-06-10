#!/usr/bin/env bash
set -uo pipefail

cd ~/final-project
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BASE="FEM/experiments/active_domain_validation/physics_integrity"
RUN_ID="sample_002_rom_prod_001"
RUN_ROOT="$BASE/pipeline_runs/guitars/sample_002/runs/$RUN_ID"
TARGET_PLAN="$BASE/pipeline_runs/validation_inputs/sample_sample_002_reference_0661505c893237ee/target_plan.json"
LOG_DIR="$BASE/pipeline_runs/index/rom_run_logs"
RUN_LOG="$LOG_DIR/${RUN_ID}.log"
TIME_LOG="$LOG_DIR/${RUN_ID}.time.txt"

echo "start_time=$(date -Is)"
echo "run_id=$RUN_ID"
echo "run_root=$RUN_ROOT"
echo "target_plan=$TARGET_PLAN"

/usr/bin/time -v -o "$TIME_LOG" \
python "$BASE/scripts/run_m4_production_pipeline.py" \
    --force-sample sample_002 \
    --run-id-suffix rom_prod_001 \
    --mesh-profile rom \
    --dataset-version m4_geometry_corrected_rommesh_v1 \
    --workers 3 \
    --target-plan-file "$TARGET_PLAN" \
    --execute \
    --compact-after-sample \
    2>&1 | tee "$RUN_LOG"

RC=${PIPESTATUS[0]}

echo "end_time=$(date -Is)"
echo "exit_code=$RC"
echo "run_id=$RUN_ID"
echo "run_root=$RUN_ROOT"
echo "run_log=$RUN_LOG"
echo "time_log=$TIME_LOG"

exit "$RC"
