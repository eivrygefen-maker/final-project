#!/bin/bash
# M4.4.1a dry-run command preview (solver-mkl strict env at execution)
set -euo pipefail
/home/vboxuser/solver-mkl/venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve_target_list.py --checkpoint-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/lprod/checkpoint" --targets-json "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/worker_results/sample_001_chunk_02/chunk_targets.json" --factor-solver mkl_pardiso --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/worker_results/sample_001_chunk_02" --dry-run
