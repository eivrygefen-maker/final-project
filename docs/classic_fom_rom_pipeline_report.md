# Classical Guitar FOM/ROM Pipeline — Technical Report

**Date:** 2026-06-16  
**Scope:** Investigation only — documents the current working **classic** pipeline end-to-end.  
**No code or behavior was changed** to produce this report.

**Future intent (not implemented here):** Generalize the same pipeline for `classic`, `box`, and `acoustic` with shape-specific LHS, bounds, GMSH bodies, and output folders.

---

## Executive summary

The production classical FOM data path is the **B3 M4 pipeline**:

```text
ROM/classic/lhs_pool.json
  → run_m4_production_pipeline.py
  → per-sample run tree under pipeline_runs/guitars/<sample_id>/runs/<run_id>/
  → scout → L_prod mesh → checkpoint → parallel worker eigen solves → aggregation
  → freeze → shared export → lhs_pool.json bookkeeping update
  → (separate step) build_m4_rom_from_completed_fom.py → ROM/classic/m4_modal_surrogate.{json,npz}
```

The **canonical entry point** is `FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py`.  
The legacy wrapper `run_pipeline.sh` → `FEM/scripts/run_pipeline.py` is **not** the current M4 production path.

---

## 1. Current classical pipeline command

### 1.1 Canonical production command

From `run_m4_production_pipeline.py` epilog and `M4_PRODUCTION_RUNNER_OLD_NEW_AUDIT.md`:

```bash
cd ~/final-project
source .venv/bin/activate

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --max-samples 10 \
  --workers 3 \
  --execute \
  --continue-on-fail
```

### 1.2 Official ROM-mesh production batch (first 5 samples)

Prepared by `v2_b3_m4_prepare_rom_official_batch.py` (`OFFICIAL_BATCH_ID = lhs_rom_official_v1_20260610`, `OFFICIAL_RUN_ID_SUFFIX = rom_official_v1`, `OFFICIAL_MAX_SAMPLES = 5`):

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_full_lhs_pool_reset.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --run-id-suffix rom_official_v1 \
  --execute

python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --batch-id lhs_rom_official_v1_20260610 \
  --run-id-suffix rom_official_v1 \
  --max-samples 5 \
  --workers 3 \
  --mesh-profile rom \
  --dataset-version m4_geometry_corrected_rommesh_v1 \
  --strict-production \
  --compact-after-sample \
  --compact-blocking \
  --isolated-subprocess \
  --execute
```

Completed official runs use `run_id` values like `sample_000_rom_official_v1` … `sample_004_rom_official_v1`.

### 1.3 Single-sample forced run (example from VM logs)

`pipeline_runs/index/rom_run_logs/sample_002_rom_prod_001.sh`:

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --force-sample sample_002 \
  --run-id-suffix rom_prod_001 \
  --mesh-profile rom \
  --dataset-version m4_geometry_corrected_rommesh_v1 \
  --workers 3 \
  --target-plan-file FEM/.../validation_inputs/sample_sample_002_reference_0661505c893237ee/target_plan.json \
  --execute \
  --compact-after-sample
```

### 1.4 Dry-run / reconcile (no heavy solve)

```bash
# Plan only — writes batch + per-sample specs, no FEM
python FEM/.../run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --max-samples 5 \
  --workers 3 \
  --dry-run

# Scan existing trees, repair freeze/LHS bookkeeping, optional shared re-export
python FEM/.../run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --reconcile-existing-runs
```

### 1.5 ROM build (post-FOM, separate command)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_m4_rom_from_completed_fom.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --shape-name classic \
  --completed-only \
  --max-samples 16
```

Official initial-five only:

```bash
python FEM/.../build_m4_rom_from_completed_fom.py \
  --official-rom-mesh-only \
  --official-initial-only
```

### 1.6 LHS pool generation (upstream, not part of per-sample solve)

```bash
python FEM/scripts/regenerate_lhs_pool.py --shape classic --samples 500 --seed 123
```

### 1.7 Important CLI arguments (`run_m4_production_pipeline.py`)

| Argument | Default | Role |
|----------|---------|------|
| `--lhs-json` | `ROM/classic/lhs_pool.json` | LHS design pool |
| `--max-samples` | `1` | Cap on samples selected this invocation |
| `--workers` | `3` | Parallel L_prod chunk workers **inside** each sample |
| `--run-id-suffix` | `m4prod1` | Run ID = `{sample_id}_{suffix}` |
| `--batch-id` | auto `lhs_prod_m4_YYYYMMDD` | Batch spec filename |
| `--start-index` / `--end-index` | slice `entries[]` | Index range in pool |
| `--force-sample` | — | Run exactly one `sample_id` |
| `--skip-completed` / `--no-skip-completed` | skip on | Respect `COMPLETED` in pool |
| `--force` | off | Re-run despite prior pass |
| `--execute` / `--dry-run` | mutually exclusive modes | Run vs plan |
| `--continue-on-fail` | off | Don't stop batch on sample failure |
| `--mesh-profile` | `rom` | `rom` or `reference` |
| `--dataset-version` | profile default | e.g. `m4_geometry_corrected_rommesh_v1` |
| `--strict-production` | on for corrected datasets | Fail-fast gates |
| `--isolated-subprocess` | on in strict mode | Fresh Python per sample |
| `--compact-after-sample` | off | Delete heavy artifacts after pass |
| `--shared-root` | auto `/media/sf_gmar` | Shared export mount |
| `--run-rom-compare` / `--run-rom-shadow` | off | Optional ROM sidecars |

---

## 2. Pipeline stages

End-to-end flow for one LHS sample:

```text
LHS pool entry (ROM/classic/lhs_pool.json)
  → select_lhs_samples() + build_lhs_batch_spec()
  → write specs/generated/{batch_id}.json and {run_id}.json
  → run_production_batch() [v2_b3_m4_lhs_production_batch.py]
       → ensure run tree: pipeline_runs/guitars/{sample_id}/runs/{run_id}/
       → write sample/sample_input.json
       → v2_b3_m4_run_one_sample.py (stages below)
            1. scout        — coarse mesh + modal density discovery (60–550 Hz)
            2. worker_plan  — frequency zones + chunk targets from scout
            3. checkpoint   — L_prod mesh build + Stage-A checkpoint export
            4. workers      — parallel eigen solves per frequency chunk
            5. aggregate    — dedupe modes, catalog, plots, participation, audio coupling
            6. freeze       — terminal manifest + durable outputs
       → shared export (plots + summary JSON)
       → optional compaction + cleanup barrier
       → sync_lhs_pool_entry() → ROM/classic/lhs_pool.json
       → append lhs_production_runs_index.jsonl
  → batch summary JSON
```

**ROM training** is a **downstream** step: `build_m4_rom_from_completed_fom.py` reads completed `aggregation/modes_catalog.jsonl` files and writes `ROM/classic/m4_modal_surrogate.{json,npz}`.

Stage order is fixed in `v2_b3_m4_run_one_sample.py`:

```python
STAGE_ORDER = ("scout", "worker_plan", "checkpoint", "workers", "aggregate", "freeze")
```

---

## 3. Input files

| File | Role |
|------|------|
| `ROM/classic/lhs_pool.json` | Master LHS design pool + per-sample run bookkeeping |
| `FEM/configs/rom_shapes.json` | Maps shape key `classic` → base config + `geometry.shape_type: Classical` |
| `FEM/configs/guitar_3d.json` | Base FEM geometry, materials, solver defaults for classical |
| `FEM/experiments/.../configs/v2_mesh_convergence_build/` | Per-level mesh build configs |
| `FEM/experiments/.../v2_mesh_convergence/v2_mesh_convergence_manifest.json` | Mesh level definitions (`L_scout_coarse`, `L_rom_prod`, etc.) |
| `FEM/geometry/models/classic.step` | CAD reference body for classical GMSH injection |
| `FEM/geometry/build_3d_guitar.py` | GMSH mesh generator (subprocess) |
| `pipeline_runs/specs/generated/{batch_id}.json` | Auto-generated batch spec |
| `pipeline_runs/specs/generated/{run_id}.json` | Per-sample production spec |
| `pipeline_runs/index/lhs_pool_status.json` | Sidecar sample status (legacy lowercase + sync) |
| Solver venv | `solver-mkl` Python used for worker chunk solves (`petsc4py` + MKL PARDISO) |

**Scripts (orchestration):**

| Script | Role |
|--------|------|
| `run_m4_production_pipeline.py` | Top-level LHS → batch runner |
| `v2_b3_m4_lhs_pool_bridge.py` | Pool I/O, spec building, selection, bookkeeping |
| `v2_b3_m4_lhs_production_batch.py` | Sequential sample loop, subprocess isolation, export, compaction |
| `v2_b3_m4_run_one_sample.py` | Per-sample stage orchestration |
| `v2_b3_m4_pipeline_run_scout.py` | Scout stage |
| `v2_b3_m4_lprod_checkpoint_run.py` | L_prod mesh + checkpoint |
| `v2_b3_m4_worker_run_remaining.py` | Worker stage (parallel chunks) |
| `v2_b3_m4_aggregate_worker_results.py` | Aggregation + catalog |
| `v2_b3_m4_shared_export.py` | Shared-folder plots/summaries |
| `build_m4_rom_from_completed_fom.py` | ROM surrogate training from catalogs |
| `FEM/scripts/regenerate_lhs_pool.py` | Regenerate LHS pool from `rom_shapes.json` |

---

## 4. LHS schema (`ROM/classic/lhs_pool.json`)

### 4.1 Pool-level fields

| Field | Example | Meaning | Consumer |
|-------|---------|---------|----------|
| `shape_name` | `"classic"` | ROM/FOM namespace for this pool | `build_sample_input()`, shared export, ROM paths |
| `sampling` | `"lhs"` | Design method | Documentation / generators |
| `wood_assignment` | `"unrestricted_5x5"` | Top/back wood drawn independently from 5 species | `regenerate_lhs_pool.py` |
| `seed` | `123` | LHS RNG seed | `regenerate_lhs_pool.py` |
| `total_samples` | `500` | Declared pool size | Generators |
| `mpi_world_size` | `0` | Legacy MPI hint (0 = none) | Pool metadata |
| `dataset_version` | optional | Pool-level dataset marker | May appear after resets |
| `entries` | array | Per-sample rows | Entire pipeline |

**Shape-specific vs common:** `shape_name` is shape-specific. Field names and entry structure are shared across shapes; only values and ID prefix differ (e.g. `box_sample_NNN` for box).

### 4.2 Entry-level fields

| Field | Meaning | Consumer |
|-------|---------|----------|
| `id` | Sample ID, e.g. `sample_000` | Run paths, specs, ROM training |
| `parameters` | LHS draw (woods + `geometry.*`) | `build_sample_input()`, mesh, ROM features |
| `status` | `PENDING` / `RUNNING` / `COMPLETED` / `FAILED` | `select_lhs_samples()`, skip logic |
| `error` / `last_error` | Failure text | Bookkeeping |
| `last_run_id` | e.g. `sample_000_rom_official_v1` | Completion matching, ROM collection |
| `last_batch_id` | Batch that last touched entry | Audit |
| `last_run_dir` | Absolute path to run tree | Reconcile, human inspection |
| `last_finished_at` / `last_elapsed_s` | Timing | Bookkeeping |
| `last_aggregation_status` | e.g. `AGGREGATION_PASS` | ROM eligibility, skip logic |
| `last_deduped_mode_count` | Mode count after dedupe | Bookkeeping / QA |
| `last_participation_computed_count` | Region participation count | Bookkeeping |
| `last_audio_coupling_computed_count` | Audio coupling count | Bookkeeping |

### 4.3 `parameters` object (per entry)

Current classical entries contain **8 keys** (no `geometry.shape_type` in pool parameters):

| Key | Type | Meaning |
|-----|------|---------|
| `top_wood_id` | string | Top plate wood (`spruce`, `cedar`, `mahogany`, `rosewood`, `maple`) |
| `back_wood_id` | string | Back/side wood |
| `geometry.length` | float (m) | Body length |
| `geometry.width` | float (m) | Body width |
| `geometry.depth` | float (m) | Body depth |
| `geometry.top_thickness` | float (m) | Top plate thickness |
| `geometry.back_thickness` | float (m) | Back plate thickness (often derived in generator) |
| `geometry.hole_radius` | float (m) | Soundhole radius |

**Who consumes parameters:**

- `v2_b3_m4_lhs_pool_bridge.build_sample_input()` → `sample/sample_input.json`
- `v2_b3_m4_lprod_interfaces.extract_geometry_dict()` → mesh build numeric body
- `v2_b3_m4_modal_surrogate_lib.encode_lhs_parameters()` → ROM feature vector
- GUI / STK parameter export (downstream, not FOM solve)

### 4.4 Sample ID convention

- Classical IDs: `sample_NNN` (zero-padded 3 digits), e.g. `sample_000`, `sample_001`, …
- `regenerate_lhs_pool.py` generates `sample_{i+1:03d}` for `i in 0..N-1` (i.e. starts at `sample_001`); the live pool also contains `sample_000` (reference/seed entry).
- **Run ID** = `{sample_id}_{run_id_suffix}`, e.g. `sample_002_rom_official_v1`.
- **Frozen reference sample:** `sample_001` (`REFERENCE_SAMPLE_ID` in bridge) — excluded by default with `--exclude-reference`; mutation requires `--allow-reference-mutation`.

### 4.5 Pool statistics (this repo snapshot)

- **501** entries in `entries[]`
- **67** marked `COMPLETED` (remainder pending or not yet run with current suffix)

---

## 5. Shape handling

Classical shape is selected through **several coupled mechanisms** (not a single `--shape` flag today):

| Mechanism | Classical value | Where |
|-----------|-----------------|-------|
| Pool `shape_name` | `"classic"` | `ROM/classic/lhs_pool.json` top level |
| `rom_shapes.json` key | `"classic"` | `FEM/configs/rom_shapes.json` |
| `parameter_sweep` | `"geometry.shape_type": ["Classical"]` | `rom_shapes.json` → used when regenerating LHS |
| Base FEM config | `"shape_type": "Classical"` | `FEM/configs/guitar_3d.json` |
| GMSH STEP reference | `classic.step` | `FEM/geometry/build_3d_guitar.py` maps `classical`/`classic` → `classic.step` |
| `sample_input.json` | `shape_name: "classic"` | Written by `build_sample_input()` from pool |
| Shared export path | `/media/sf_gmar/classic/...` | `v2_b3_m4_shared_export.read_shape_name()` |
| ROM artifacts | `ROM/classic/` | `shape_rom_dir(repo_root, "classic")` |

**Not in LHS parameters:** `geometry.shape_type` is **not** stored per entry in the current classical pool. Body shape for meshing defaults to **Classical** via `v2_sensitivity_mesh.sample_geometry()`:

```python
NOMINAL_GEOMETRY = {
    "shape_type": "Classical",
    ...
}
geom = dict(NOMINAL_GEOMETRY)
geom.update(sample.get("geometry") or {})  # only numeric keys from LHS
```

So for classical production, shape is effectively **implicit** (hard-coded Classical in mesh helper + `guitar_3d.json`), while `shape_name: classic` is carried as metadata for export/ROM.

---

## 6. Geometry and bounds

### 6.1 Where bounds come from

`FEM/scripts/regenerate_lhs_pool.py`:

1. Loads `FEM/configs/rom_shapes.json` → `shapes["classic"]`
2. Loads base config `FEM/configs/guitar_3d.json`
3. Builds 7D sweep spec via `build_7d_lhs_sweep_spec()`:
   - Length/width/depth bounds from `_shape_length_width_depth_bounds(shape_type)` — for classical (default branch): length 0.35–0.60 m, width 0.20–0.45 m, depth 0.08–0.15 m
   - `geometry.top_thickness`: global wood library min/max
   - `geometry.hole_radius`: 0.035–0.055 m
   - `top_wood_id` / `back_wood_id`: all 5 woods
4. `finalize_lhs_thickness_params()` derives `geometry.back_thickness` from top thickness

### 6.2 Body vs generic parameters

| Category | Parameters | Shape-specific? |
|----------|------------|-----------------|
| Body envelope | `geometry.length`, `width`, `depth`, `hole_radius` | Bounds differ per shape in `regenerate_lhs_pool.py` |
| Plate thickness | `geometry.top_thickness`, `geometry.back_thickness` | Same thickness library; back derived |
| Materials | `top_wood_id`, `back_wood_id` | Common 5-wood set |
| Shape label | `shape_type` / `shape_name` | Shape-specific; not in LHS `parameters` today |
| Bout ratios | `upper_bout`, `waist`, `lower_bout` | In `guitar_3d.json` defaults, not LHS-swept for classic M4 |

### 6.3 Geometry fingerprint (FOM)

`v2_b3_m4_lprod_interfaces.GEOMETRY_FINGERPRINT_KEYS`:

```python
("length", "width", "depth", "hole_radius", "top_thickness", "back_thickness")
```

Used for mesh invalidation / provenance — extracted from `parameters` dot-keys.

---

## 7. GMSH integration

### 7.1 Call chain

```text
sample_input.json
  → extract_geometry_dict()  # numeric body only
  → build_lprod_mesh_for_case()  [v2_b3_m4_lprod_mesh_build.py]
  → build_level_mesh()  [v2_mesh_convergence_mesh.py]
       → writes per-sample config JSON from guitar_3d.json template
       → subprocess: python FEM/geometry/build_3d_guitar.py
            env: FEM_MESH_CONFIG, FEM_MESH_OUT, FEM_MESH_LC_SCALE,
                 FEM_MESH_EXPLICIT_CONTROLS_JSON, level build_env
  → output: v2_mesh_convergence/mesh/<level_id>/<sample_id>.msh
  → copied/linked under run tree lprod/mesh/
```

### 7.2 How parameters reach GMSH

1. Per-sample config JSON merges `guitar_3d.json` with `sample_geometry()` output (length, width, depth, thicknesses, hole_radius; **shape_type defaults to Classical**).
2. `build_3d_guitar.py` reads config from `FEM_MESH_CONFIG` environment variable.
3. CAD reference: `_reference_step_filename(shape_type)` → `classic.step` for classical.
4. Reference solid is scaled/morphed via `_scale_reference_to_target()` using length, depth, bout parameters.

### 7.3 Classical hard-coding in GMSH path

| Location | Assumption |
|----------|------------|
| `v2_sensitivity_mesh.NOMINAL_GEOMETRY` | `shape_type: "Classical"` |
| `guitar_3d.json` | `"shape_type": "Classical"` |
| `build_3d_guitar.py` default fallback | `classic.step` |
| `_reference_shape_family()` | Non-dreadnought → `"classical"` bout nominals |

Box/acoustic STEP files exist (`box.step`, `acoustic.step`) but are **not** selected by the classical LHS path unless `shape_type` is changed upstream.

---

## 8. FOM/FEM solve

### 8.1 Solver stack

| Layer | Technology |
|-------|------------|
| Scout / checkpoint | Stage-A checkpoint pipeline (`v2_b3_m4_lprod_checkpoint_run.py`, `v2_b3_checkpoint_pipeline_lib.py`) |
| Worker eigen solves | `v2_b3_checkpoint_solve_target_list.py` via **solver-mkl** subprocess |
| Linear solver | `--factor-solver mkl_pardiso` |
| Parallelism | `ProcessPoolExecutor` over frequency **chunks** (`run_chunks_fcfs_parallel`) |

Worker command (from `build_chunk_plan()`):

```text
{solver_python} FEM/.../v2_b3_checkpoint_solve_target_list.py \
  --checkpoint-dir <run>/lprod/checkpoint \
  --targets-json <run>/worker_results/<chunk_id>/chunk_targets.json \
  --factor-solver mkl_pardiso \
  --output-dir <run>/worker_results/<chunk_id>/
```

### 8.2 Config resolution per sample

1. **LHS parameters** → `sample_input.json` (`m4_sample_input_v1` schema)
2. **Mesh profile** applied: `mesh_profile`, `mesh_level_id`, `dataset_version`, `effective_controls_m` (`apply_mesh_profile_to_sample_input`)
3. **Production ROM mesh defaults:**
   - `mesh_profile = rom`
   - `dataset_version = m4_geometry_corrected_rommesh_v1`
   - Level ID `L_rom_prod` (via `v2_b3_m4_mesh_profile_lib`)
4. **Frequency policy** embedded in batch spec (`DEFAULT_FREQUENCY_POLICY`): band 60–550 Hz, scout spacing 7.5 Hz, zone spacings 6 / 9 / 12.5 Hz, `workers: 3`
5. **Strict production** (`--strict-production`): additional fail-fast gates in `v2_b3_m4_production_contracts.py` for corrected datasets

### 8.3 Modal / eigen output

Per chunk: `worker_results/<chunk_id>/solver_result.json`, `worker_result.json`  
Aggregated: `aggregation/modes_catalog.jsonl` (primary ROM input), `modes_summary.json`, plots, participation, audio coupling metadata.

---

## 9. Workers / processors

| Level | Behavior |
|-------|----------|
| **Batch** | Samples run **sequentially** in `run_production_batch()` |
| **Per sample** | `--workers N` (default 3) parallel **frequency chunks** |
| **Implementation** | `concurrent.futures.ProcessPoolExecutor` in `run_chunks_fcfs_parallel()` |
| **Subprocess isolation** | `--isolated-subprocess` spawns fresh `v2_b3_m4_run_one_sample.py` per sample |
| **Thread env** | `OMP_NUM_THREADS=1`, etc. — one BLAS thread per worker process |
| **Not used for M4 batch** | MPI across samples (`mpi_world_size: 0` in pool) |

**Clarification:** `--workers 3` is **not** 3 samples at once; it is 3 concurrent eigenvalue chunk solves **within** one guitar sample.

---

## 10. Output folders (classical)

**Physics root:**

```text
FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/
```

### 10.1 Per-sample run tree

```text
pipeline_runs/guitars/<sample_id>/runs/<run_id>/
├── sample/sample_input.json
├── scout/                          # coarse scout artifacts, density_zones.json
├── lprod/
│   ├── mesh/<level_id>/<sample_id>.msh
│   ├── checkpoint/                 # Stage-A export, region_dof_indices.npz
│   └── lprod_target_plan.json
├── worker_results/<chunk_id>/      # per-chunk solve + logs
├── aggregation/
│   ├── modes_catalog.jsonl         # ★ primary FOM → ROM contract
│   ├── modes_summary.json
│   ├── aggregation_result.json
│   ├── runtime_summary.json
│   ├── warnings_and_failures.json
│   ├── *.png                     # mode frequency plots
│   └── shared_export_manifest.json
├── freeze/                         # durable terminal outputs
├── compaction/                     # if --compact-after-sample
├── cleanup/                        # cleanup barrier manifest
├── logs/
└── pipeline_run_manifest.json
```

### 10.2 Batch / index / specs

```text
pipeline_runs/specs/generated/
├── <batch_id>.json
├── <batch_id>_plan.json
├── <batch_id>_summary.json
└── <run_id>.json                   # per-sample spec

pipeline_runs/index/
├── lhs_pool_status.json
├── lhs_production_runs_index.jsonl
└── rom_run_logs/                   # shell wrappers + logs

pipeline_runs/batches/<batch_id>/
└── batch_execution_summary.json
```

### 10.3 Convergence mesh cache (shared across runs)

```text
FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/mesh/<level_id>/<sample_id>.msh
```

### 10.4 ROM outputs (after `build_m4_rom_from_completed_fom.py`)

```text
ROM/classic/
├── m4_modal_surrogate.json
├── m4_modal_surrogate.npz
├── rom_model_manifest.json
├── official_rom_dataset.jsonl      # official ROM-mesh registry
└── m4_official_rom_*_report.json   # build/audit reports
```

---

## 11. Shared folder export

**Default root:** `/media/sf_gmar` (`FEM/scripts/paths.py`, overridable via `SHARED_HOST_DIR`).

Classical layout (from `v2_b3_m4_shared_export.py`):

| Asset | Path pattern |
|-------|--------------|
| Plots | `/media/sf_gmar/classic/plots/<sample_id>/<run_id>__<plot_name>.png` |
| Summaries | `/media/sf_gmar/classic/summaries/<sample_id>__<run_id>__summary.json` |
| Graph manifest | `/media/sf_gmar/classic/summaries/<sample_id>__<run_id>__graph_export_manifest.json` |

**Approved plot names exported:**

- `mode_frequency_vs_bridge_excitation.png`
- `mode_frequency_vs_mic_output_proxy.png`
- `mode_frequency_vs_radiation_proxy.png`
- `mode_frequency_vs_top_back_air_share.png`

**Excluded:** `mode_frequency_plot.png` (legacy).

**Local manifest:** `{run_root}/aggregation/shared_export_manifest.json`

**Export gate:** `try_export_sample_to_shared()` requires `is_run_usably_complete()` (aggregation pass, no failed/missing chunks).

**Not exported to shared by default:** full `modes_catalog.jsonl`, checkpoint NPZ, worker heavy artifacts — those stay in the run tree (and may be compacted locally). ROM training reads catalogs from the **local** run tree, not from shared.

---

## 12. ROM data contract

### 12.1 FOM → ROM training input

**Primary artifact per completed sample:**

```text
pipeline_runs/guitars/<sample_id>/runs/<run_id>/aggregation/modes_catalog.jsonl
```

**Collector:** `collect_completed_fom_training_rows()` in `v2_b3_m4_modal_surrogate_lib.py`:

1. Iterate `ROM/classic/lhs_pool.json` `entries[]`
2. Require `status == COMPLETED` (or matching `last_aggregation_status == AGGREGATION_PASS`)
3. Resolve `run_id` from `last_run_id` or `{sample_id}_{run_id_suffix}`
4. Load catalog; dedupe modes; enrich intensity derivatives
5. Pair with `parameters` from pool entry (or `sample/sample_input.json` override)

**Feature schema:** `m4_lhs_geometry_wood_v1` — 6 geometry floats + 2 wood indices (`GEOMETRY_KEYS` + woods).

### 12.2 ROM model outputs

| File | Purpose |
|------|---------|
| `ROM/classic/m4_modal_surrogate.json` | Model metadata, training sample IDs, schema version |
| `ROM/classic/m4_modal_surrogate.npz` | Feature matrix, frequencies, normalization stats |
| `ROM/classic/rom_model_manifest.json` | Active backend pointer (`m4_surrogate`) |

### 12.3 ROM online consumption (GUI)

`gui/app.py` → `run_rom_acoustics()`:

- Loads `ROM/classic/m4_modal_surrogate.{json,npz}` via `load_surrogate_model(repo_root, shape_name)`
- `shape_name` from `rom_namespace()` → `"classic"` for Classical UI shape
- Predicts modal catalog; writes body JSON for STK — **no FEM** in ROM path

**Contract:** ROM does **not** read `pipeline_runs/` at prediction time; it reads the trained surrogate under `ROM/<shape>/`. FOM pipeline must run first to populate training data, then `build_m4_rom_from_completed_fom.py` must be run to refresh the surrogate.

---

## 13. Completion / skip logic

### 13.1 Sample selection skips (`select_lhs_samples`)

Skip when:

- `is_lhs_entry_completed(entry, run_id)` — `status == COMPLETED` and `last_run_id` matches current suffix
- Sidecar `lhs_pool_status.json` shows `pass` + `AGGREGATION_PASS` for same `run_id`
- `include_only_pending` and entry already completed

Override: `--force`, `--no-skip-completed`, `--force-sample`

### 13.2 Run tree reuse (`_classify_run_status`)

- `already_complete_reuse` — aggregation pass + freeze present → skip solve, may still export/compact
- `resume_possible` — partial artifacts → resume from failing stage

### 13.3 Usable completion (`is_run_usably_complete`)

```python
aggregation_status == "AGGREGATION_PASS"
and failed_chunks == 0 and missing_chunks == 0
and final_aggregation_ready == True
```

### 13.4 Per-stage skip (`assess_stages` in `v2_b3_m4_run_one_sample.py`)

Stages with `pass=True` are skipped unless `--force` / `--force-checkpoint` / `--force-workers` / `--force-aggregation`.

### 13.5 Bookkeeping markers

| Marker | Location |
|--------|----------|
| `entries[].status` | `ROM/classic/lhs_pool.json` |
| `entries[].last_*` fields | Same |
| `lhs_pool_status.json` | Sidecar per-sample status |
| `pipeline_run_manifest.json` | `terminal_status`, stage statuses |
| `aggregation_result.json` | `aggregation_status` |
| `freeze/` outputs | Terminal freeze gate |

---

## 14. Failure handling

| Stage | Detection | Where written |
|-------|-----------|---------------|
| Sample selection | No eligible entries | stdout / exit code 2 |
| Scout / checkpoint / workers | Stage assess fail | `pipeline_run_manifest.json`, stage logs under `logs/` |
| Worker chunk | `solver_result.json` / non-zero exit | `worker_results/<chunk>/log.txt`, `worker_result.json` |
| Aggregation | Missing chunks / dedupe fail | `aggregation/warnings_and_failures.json`, `aggregation_result.json` |
| Batch accounting | `classify_batch_sample_outcome()` | `batch_execution_summary.json`, pool patch `status: FAILED` |
| Shared export | Missing plots / copy fail | `aggregation/shared_export_manifest.json` `export_status: FAILED` |
| Compaction / cleanup | `--compact-blocking` | stderr, batch row `compaction_error` |
| ROM build | Missing catalog / insufficient modes | stderr, skipped rows printed |

**Batch behavior:** Without `--continue-on-fail`, first sample failure stops the batch. Pool entry gets `FAILED` / `last_error` via `lhs_pool_entry_patch_from_run()`.

**Graceful stop:** `--request-stop` writes `STOP_AFTER_CURRENT_SAMPLE` control file; honored between samples.

---

## 15. Hard-coded classical assumptions

| Location | Hard-coded value |
|----------|------------------|
| `run_m4_production_pipeline.py` | `DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"` |
| `build_m4_rom_from_completed_fom.py` | same; `--shape-name classic` default |
| `v2_b3_m4_prepare_rom_official_batch.py` | `ROM/classic/lhs_pool.json`, official batch constants |
| `v2_b3_m4_lhs_pool_bridge.build_sample_input()` | `shape_name` default `"classic"` |
| `v2_b3_m4_shared_export.read_shape_name()` | fallback `"classic"` |
| `v2_b3_m4_official_rom_dataset_lib` | `OFFICIAL_DATASET_REGISTRY_REL = "ROM/classic/official_rom_dataset.jsonl"` |
| `v2_b3_m4_official_rom_dataset_lib` | `OFFICIAL_INITIAL_RUN_IDS` = `sample_000..004_rom_official_v1` |
| `FEM/configs/rom_shapes.json` | `"classic"` entry only for classical |
| `FEM/configs/guitar_3d.json` | `"shape_type": "Classical"` |
| `v2_sensitivity_mesh.NOMINAL_GEOMETRY` | `"shape_type": "Classical"` |
| `build_3d_guitar.py` | `classic.step` for classical/classic aliases |
| `FEM/scripts/paths.py` | `DEFAULT_SHAPE_NAME = "classic"`, `ROM_CLASSIC_SNAPSHOTS_DIR` |
| `FEM/scripts/regenerate_lhs_pool.py` | default `--shape classic` |
| `pipeline_runs/guitars/` | Sample IDs `sample_*` (not shape-prefixed) |
| `gui/app.py` | `ROM_ROOT / "classic" / m4_modal_surrogate.*` |
| `REFERENCE_SAMPLE_ID` | `"sample_001"` (frozen reference, not `sample_000`) |
| `v2_b3_m4_lprod_interfaces.BASELINE_GEOMETRY` | Classical numeric defaults |

**Note:** Shared export and ROM paths already use a `{shape}` segment pattern (`classic/`, `box/`, etc.) — the plumbing is partially shape-aware, but CLI defaults and LHS pool path are classical-specific.

---

## 16. Future generalization plan (no implementation)

Goal: one production command with shape selection, e.g.:

```bash
python FEM/.../run_m4_production_pipeline.py \
  --shape classic \
  --lhs-json ROM/classic/lhs_pool.json \
  ...
```

### 16.1 What already generalizes cleanly

| Component | Status |
|-----------|--------|
| `rom_shapes.json` | Has `classic`, `dreadnought`, `box` entries |
| `regenerate_lhs_pool.py --shape <name>` | Writes `ROM/<shape>/lhs_pool.json` with shape-specific bounds |
| `build_sample_input()` | Reads `pool["shape_name"]` |
| Shared export | `{shared_root}/{shape}/plots|summaries/` |
| `build_m4_rom_from_completed_fom.py --shape-name` | Writes `ROM/<shape>/` |
| `build_3d_guitar.py` | STEP mapping for box/acoustic/classical |
| `FEM/scripts/paths.get_shared_dir(shape, category)` | Shape-scoped shared paths |

### 16.2 What must change for `--shape classic|box|acoustic`

1. **CLI shape flag** on `run_m4_production_pipeline.py`  
   - Derive default `--lhs-json` → `ROM/<shape>/lhs_pool.json`  
   - Thread `shape_name` through batch spec without manual path args

2. **LHS sample IDs**  
   - Classical: `sample_NNN`  
   - Box (already): `box_sample_NNN`  
   - Acoustic: define convention (e.g. `acoustic_sample_NNN`)  
   - Update `select_lhs_samples`, index paths, and GUI ROM namespace consistently

3. **Geometry.shape_type in sample_input**  
   - Inject `geometry.shape_type` (or `m4_run_metadata.shape_type`) from `rom_shapes.json` per shape  
   - Stop relying on `NOMINAL_GEOMETRY["Classical"]` default in `sample_geometry()` for non-classic shapes

4. **GMSH / mesh**  
   - Pass correct `shape_type` into per-sample `guitar_3d.json` merge  
   - Verify bout scaling paths for box (may need box-specific nominal widths) and acoustic (`acoustic.step`)

5. **Parameter bounds**  
   - Extend `regenerate_lhs_pool._shape_length_width_depth_bounds()` for `acoustic`  
   - Ensure `tools/generate_box_lhs_pool.py` (or unified generator) matches `regenerate_lhs_pool` contract

6. **Output folder layout**  
   - Option A: keep flat `pipeline_runs/guitars/<sample_id>/` (IDs are globally unique)  
   - Option B: add `pipeline_runs/guitars/<shape>/<sample_id>/` — requires updating `GUITARS_ROOT`, bridge, surrogate collector, reconcile

7. **Hard-coded defaults**  
   - Replace `DEFAULT_LHS_REL`, `DEFAULT_SHAPE_NAME`, GUI `ROM/classic` paths with shape resolver  
   - Parameterize `OFFICIAL_*` constants per shape or move to shape-specific prep scripts

8. **ROM feature schema**  
   - Confirm `m4_lhs_geometry_wood_v1` features are sufficient for box/acoustic bodies  
   - May need shape-specific feature columns or separate surrogate models per `ROM/<shape>/`

9. **Overnight batch scripts**  
   - Classical: today = direct `run_m4_production_pipeline.py` (no dedicated shell; box has `run_box_fom_overnight_batch.sh`)  
   - Add symmetric launchers or one `run_fom_overnight_batch.sh --shape <name>`

10. **Completion / skip**  
    - Pool files are per-shape (`ROM/box/lhs_pool.json`); no cross-shape skip  
    - Reconcile must scan correct `guitars/` trees

### 16.3 Recommended generalization order

1. Document and freeze this classical contract (this report).  
2. Add `--shape` to production runner with derived LHS path only (classic behavior unchanged).  
3. Inject `shape_type` into `sample_input` + mesh config for box; validate one box sample E2E.  
4. Add acoustic to `rom_shapes.json` + LHS generator + STEP validation.  
5. Unify overnight launcher.  
6. ROM build per shape; GUI `rom_namespace()` already partially shape-aware.

### 16.4 Minimal diff principle

The user requirement is correct: **same pipeline code**, different:

- `--shape` / `--lhs-json`
- `rom_shapes.json` bounds + `geometry.shape_type`
- GMSH STEP selection (`classic.step` / `box.step` / `acoustic.step`)
- `ROM/<shape>/` and `/media/sf_gmar/<shape>/` outputs

The largest risk areas are **`sample_geometry()` Classical default**, **`GUITARS_ROOT` layout**, and **sample ID conventions** — these should be addressed before large batch runs for box/acoustic.

---

## Appendix A — Key file index

| Path | Role |
|------|------|
| `ROM/classic/lhs_pool.json` | LHS pool + bookkeeping |
| `FEM/configs/rom_shapes.json` | Shape registry |
| `FEM/configs/guitar_3d.json` | Base FEM config |
| `FEM/.../run_m4_production_pipeline.py` | Production entry |
| `FEM/.../v2_b3_m4_lhs_pool_bridge.py` | Pool ↔ specs bridge |
| `FEM/.../v2_b3_m4_lhs_production_batch.py` | Batch executor |
| `FEM/.../v2_b3_m4_run_one_sample.py` | Stage orchestrator |
| `FEM/.../v2_b3_m4_lprod_mesh_build.py` | L_prod mesh |
| `FEM/geometry/build_3d_guitar.py` | GMSH |
| `FEM/.../v2_b3_m4_worker_run_lib.py` | Chunk parallel solves |
| `FEM/.../v2_b3_checkpoint_solve_target_list.py` | Eigen solver CLI |
| `FEM/.../v2_b3_m4_aggregate_worker_results.py` | Catalog aggregation |
| `FEM/.../v2_b3_m4_shared_export.py` | Shared export |
| `FEM/.../build_m4_rom_from_completed_fom.py` | ROM training |
| `FEM/scripts/regenerate_lhs_pool.py` | LHS generation |
| `FEM/scripts/paths.py` | Shared folder helpers |
| `FEM/.../docs/B3_M4_FULL_LHS_PIPELINE_ORCHESTRATOR_CONTRACT.md` | Architecture spec |
| `FEM/.../docs/M4_PRODUCTION_RUNNER_OLD_NEW_AUDIT.md` | Runner audit + commands |

---

## Appendix B — Legacy path (do not use for M4 production)

`run_pipeline.sh` wraps `FEM/scripts/run_pipeline.py` with `FEM/configs/lhs_samples.json` — pre-M4 ROM snapshot workflow. The script itself points operators to `run_m4_production_pipeline.py` for B3 M4 production.
