# M4.5.1 batch dry-run — sample_003

- batch_id: `m4_5_first3_lhs`
- run_id: `sample_003_m45dry1`
- reuse_status: **planned_new_run**
- will_execute: **false**

## Stages (planned)

- **Stage 0 — sample/config**: PLANNED — `# resolve overlay for sample_003 (v2_b3_resolve_pilot_core_config.py pattern)`
- **Stage 1 — scout mesh**: PLANNED — `/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_v...`
- **Stage 1 — scout checkpoint**: PLANNED — `/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_v...`
- **Stage 2 — scout discovery**: PLANNED — `/home/vboxuser/solver-mkl/venv/bin/python FEM/experiments/active_domain_valid...`
- **Stage 3 — zones + L_prod plan**: PLANNED — `(planner only)`
- **Stage 4 — L_prod mesh**: PLANNED — `/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_v...`
- **Stage 4 — L_prod checkpoint**: PLANNED — `/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_v...`
- **Stage 5 — L_prod workers**: PLANNED — `# FCFS workers W0..W2 (M4.4)`
- **Stage 6 — aggregation**: PLANNED — `(planner only)`
- **Stage 6 — freeze milestone**: PLANNED — `python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_...`

No mesh build, scout solve, workers, aggregation, or freeze executed.
