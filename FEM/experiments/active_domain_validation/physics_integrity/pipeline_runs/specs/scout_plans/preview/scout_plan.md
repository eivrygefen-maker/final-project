# Coarse-mesh modal-density scout plan (dry-run)

- **Generated:** `2026-06-02T21:53:12Z`
- **will_execute:** `False`
- **Strategy:** coarse_fem_mesh_modal_density_scout

## L_prod verified sizing (mm)

- `wood_surface_size_m`: **7.0**
- `wood_thickness_size_m`: **1.0**
- `air_threshold_size_min_m`: **4.0**
- `air_threshold_size_max_m`: **50.0**
- **active_dim (m3exec2 ref):** 316017

## Proposed L_scout_coarse (mm)

- `wood_thickness_size_m`: **3.0**
- `wood_surface_size_m`: **8.5**
- `air_threshold_size_min_m`: **11.0**
- `air_threshold_size_max_m`: **55.0**
- `air_threshold_dist_min_m`: **15.0**
- `air_threshold_dist_max_m`: **250.0**
- **reuse L_dev_coarse:** False

## Active dimension estimate

- Point estimate: **47941**
- Rough band: **[31522, 78064]**

## Warnings

- L_prod mesh file not found on this host: C:\projects\final-project\final-project\FEM\experiments\active_domain_validation\physics_integrity\v2_mesh_convergence\mesh\L_prod\baseline_coupled_v2.msh
- Manifest has no L_scout_coarse yet; mesh build and Stage A --mesh-level require manifest + code updates.
- v2_b3_checkpoint_export ALLOWED_MESH_LEVELS does not include L_scout_coarse; extend before Stage A.

## Command previews (do not run without approval)

### Mesh build
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mesh_convergence.py --levels L_scout_coarse --cases baseline_coupled_v2
```

### Stage A
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_scout_coarse --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/lhs_pilot_001_timing/resolved_core_config.json" --output-dir "C:/projects/final-project/final-project/FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_scout_coarse_scout_m34_plan_preview"
```

### Stage B discovery
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_target_density_experiment.py --checkpoint-dir "C:/projects/final-project/final-project/FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_scout_coarse_scout_m34_plan_preview" --start-hz 60.0 --stop-hz 550.0 --spacings-hz 15.0 --B3-discovery-mode --discovery-band-hz 60.0 550.0 --target-window-half-width-hz 7.5 --output-dir "C:/projects/final-project/final-project/FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/solver_benchmarks/scout_density_discovery_60_550_step15_scout_m34_plan_preview"
```
