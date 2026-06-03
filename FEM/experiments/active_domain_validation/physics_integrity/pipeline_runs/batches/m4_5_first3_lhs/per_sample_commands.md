# Per-sample command previews — m4_5_first3_lhs

Planned execution order per guitar (not run by this dry-run).

## sample_002 — sample_002_m45dry1

### Stage 0 — sample/config
```bash
# resolve overlay for sample_002 (v2_b3_resolve_pilot_core_config.py pattern)
```

### Stage 1 — scout mesh
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_B3_scout_coarse_mesh_build.py # sample-specific geometry: sample_002 (M4.3+ may pass FEM geometry overrides)
```

### Stage 1 — scout checkpoint
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_scout_coarse --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/sample_002/resolved_core_config.json" --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_002/runs/sample_002_m45dry1/scout/checkpoint"
```

### Stage 2 — scout discovery
```bash
/home/vboxuser/solver-mkl/venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_target_density_experiment.py --checkpoint-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_002/runs/sample_002_m45dry1/scout/checkpoint" --start-hz 60.0 --stop-hz 550.0 --spacings-hz 7.5 --B3-discovery-mode --discovery-band-hz 60.0 550.0 --target-window-half-width-hz 3.75 --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_002/runs/sample_002_m45dry1/scout/discovery"
```

### Stage 4 — L_prod mesh
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mesh_convergence.py # planned L_prod build for sample sample_002; geometry-aware path TBD M4.3+
```

### Stage 4 — L_prod checkpoint
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_prod --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/sample_002/resolved_core_config.json" --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_002/runs/sample_002_m45dry1/lprod/checkpoint"
```

### Stage 5 — L_prod workers
```bash
# FCFS workers W0..W2 (M4.4)
```

### Stage 6 — freeze milestone
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_freeze_first_e2e_run.py --run-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_002/runs/sample_002_m45dry1 --force  # after AGGREGATION_PASS
```

## sample_003 — sample_003_m45dry1

### Stage 0 — sample/config
```bash
# resolve overlay for sample_003 (v2_b3_resolve_pilot_core_config.py pattern)
```

### Stage 1 — scout mesh
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_B3_scout_coarse_mesh_build.py # sample-specific geometry: sample_003 (M4.3+ may pass FEM geometry overrides)
```

### Stage 1 — scout checkpoint
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_scout_coarse --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/sample_003/resolved_core_config.json" --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_003/runs/sample_003_m45dry1/scout/checkpoint"
```

### Stage 2 — scout discovery
```bash
/home/vboxuser/solver-mkl/venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_target_density_experiment.py --checkpoint-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_003/runs/sample_003_m45dry1/scout/checkpoint" --start-hz 60.0 --stop-hz 550.0 --spacings-hz 7.5 --B3-discovery-mode --discovery-band-hz 60.0 550.0 --target-window-half-width-hz 3.75 --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_003/runs/sample_003_m45dry1/scout/discovery"
```

### Stage 4 — L_prod mesh
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mesh_convergence.py # planned L_prod build for sample sample_003; geometry-aware path TBD M4.3+
```

### Stage 4 — L_prod checkpoint
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_prod --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/sample_003/resolved_core_config.json" --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_003/runs/sample_003_m45dry1/lprod/checkpoint"
```

### Stage 5 — L_prod workers
```bash
# FCFS workers W0..W2 (M4.4)
```

### Stage 6 — freeze milestone
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_freeze_first_e2e_run.py --run-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_003/runs/sample_003_m45dry1 --force  # after AGGREGATION_PASS
```

## sample_004 — sample_004_m45dry1

### Stage 0 — sample/config
```bash
# resolve overlay for sample_004 (v2_b3_resolve_pilot_core_config.py pattern)
```

### Stage 1 — scout mesh
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_B3_scout_coarse_mesh_build.py # sample-specific geometry: sample_004 (M4.3+ may pass FEM geometry overrides)
```

### Stage 1 — scout checkpoint
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_scout_coarse --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/sample_004/resolved_core_config.json" --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_004/runs/sample_004_m45dry1/scout/checkpoint"
```

### Stage 2 — scout discovery
```bash
/home/vboxuser/solver-mkl/venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_target_density_experiment.py --checkpoint-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_004/runs/sample_004_m45dry1/scout/checkpoint" --start-hz 60.0 --stop-hz 550.0 --spacings-hz 7.5 --B3-discovery-mode --discovery-band-hz 60.0 550.0 --target-window-half-width-hz 3.75 --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_004/runs/sample_004_m45dry1/scout/discovery"
```

### Stage 4 — L_prod mesh
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mesh_convergence.py # planned L_prod build for sample sample_004; geometry-aware path TBD M4.3+
```

### Stage 4 — L_prod checkpoint
```bash
/home/vboxuser/final-project/.venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_prod --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/sample_004/resolved_core_config.json" --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_004/runs/sample_004_m45dry1/lprod/checkpoint"
```

### Stage 5 — L_prod workers
```bash
# FCFS workers W0..W2 (M4.4)
```

### Stage 6 — freeze milestone
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_freeze_first_e2e_run.py --run-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_004/runs/sample_004_m45dry1 --force  # after AGGREGATION_PASS
```
