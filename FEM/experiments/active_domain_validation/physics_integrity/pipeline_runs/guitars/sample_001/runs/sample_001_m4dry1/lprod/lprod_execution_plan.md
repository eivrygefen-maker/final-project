# L_prod execution plan (dry-run) — sample_001

- run_id: `sample_001_m4dry1`
- will_execute: **false**
- targets: **56**
- chunks: **11**
- workers: **3**
- input status: `SCOUT_PASS_TARGET_PLAN_READY`

## Stage 4 — L_prod mesh + checkpoint

- mesh: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/lprod/mesh/L_prod/sample_001.msh`
- checkpoint: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/lprod/checkpoint`
- lprod_mesh_status: `planned_build_required`
- lprod_checkpoint_status: `planned`

```bash
# production .venv — sample-specific L_prod mesh when geometry != baseline
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mesh_convergence.py # planned for sample_001; mesh_level=L_prod
# production .venv
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_prod --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/sample/resolved_core_config.json" --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/lprod/checkpoint"
```

## Stage 5 — workers (FCFS)

- makespan estimate (3 workers): **1900.0 s**
- serial estimate: **5320.0 s**

### Solver interface (M4.4.1a)

- Primary: `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve_target_list.py` with `--targets-json` per chunk
- Legacy (not used): `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve.py` `--targets-hz`

## Safety

- No L_prod execution in M4.4.1a.
