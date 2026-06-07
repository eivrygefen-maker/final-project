# M4 operator mesh and soundhole root-cause report

**Date:** 2026-06-05  
**Status:** Decisive code-trace + runtime evidence  
**Constraint:** Read-only audit. No production pipeline changes. `sample_036` not touched.

---

## Executive verdict

| Hypothesis | Verdict |
|------------|---------|
| **A. Fixed FEM topology intentional and physically valid** | **Partially** — fixed topology is **intentional in current B3 Stage A code**, but **not valid** for LHS geometry-varying guitar bodies |
| **B. Sample Gmsh meshes generated but not used by operator assembly** | **Confirmed** |
| **C. Geometry enters only through coefficients/material scaling** | **Confirmed** (when `core_config_path` is set, materials from `resolved_core_config.json`; mesh coordinates fixed) |
| **D. Baseline/reference mesh reused incorrectly** | **Confirmed** for operator assembly; not a file-copy bug — **hard-coded canonical path** |
| **E. Audit compares generated mesh stats to different active computational mesh** | **Confirmed** |

**Production decision (pre-validation):** `RERUN_ALL_35_SAMPLES` after fixing Stage A mesh wiring + aperture mic proxy.

---

## Observed runtime evidence (full-retention samples)

| Sample | Generated mesh nodes | Checkpoint `n_w` | `active_dimension` | A CSR shape | A structure identical? |
|--------|---------------------|------------------|--------------------|-------------|------------------------|
| sample_000 | ~306,035 | 378,243 | 316,017 | 316017×316017 | Yes (all 4) |
| sample_001 | ~176,895 | 378,243 | 316,017 | 316017×316017 | Yes |
| sample_034 | ~409,871 | 378,243 | 316,017 | 316017×316017 | Yes |
| sample_035 | ~400,851 | 378,243 | 316,017 | 316017×316017 | Yes |

- Mesh SHA256 and geometry fingerprints **differ** across samples (generated artifacts are distinct).
- A/M **full file** SHA256 differs (matrix **values** differ — materials).
- A/M **CSR structure** (`indptr` + `indices` + `shape`) is **identical** across samples.
- `n_soundhole_dofs = 0`; `mic_output_method = cavity_pressure_max_proxy_v1` on all full samples.

This pattern is **exactly** what the code path predicts.

---

## Runtime provenance chain (code trace)

### Stage map

| Step | Function | Source file | What actually happens |
|------|----------|-------------|------------------------|
| 1. LHS | `load_lhs_pool` / `build_sample_input` | `v2_b3_m4_lhs_pool_bridge.py` | Per-sample `geometry.*`, wood IDs |
| 2. Scout resolve | `resolve_m4_sample` | `v2_b3_m4_pipeline_run_scout.py` | `resolved_core_config.json` with materials + `solver.mesh_file` pointing at run-tree sample mesh |
| 3. L_prod mesh build | `build_lprod_mesh_for_case` | `v2_b3_m4_lprod_mesh_build.py` | Gmsh → `v2_mesh_convergence/mesh/L_prod/{sample_id}.msh`, copy to `run_root/lprod/mesh/L_prod/{sample_id}.msh` |
| 4. Mesh summary | `write_json(..._mesh_build_summary.json)` | same | Records **generated** `n_nodes`, `n_tetrahedra` |
| 5. Stage A export | `run_checkpoint_export` | `v2_b3_checkpoint_export.py` | Calls `_b3_build_corrected_structural_active_operators` |
| 6. **Operator mesh load** | `_b3_build_corrected_structural_active_operators` | `run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py` | **`mesh_file = mesh_path(mesh_level, "baseline_coupled_v2")`** — ignores run-tree sample mesh |
| 7. Parent assembly | `_assemble_reduced_coupled_replay` | `v2_build_coupled_acoustic_seed.py` | Loads config from `resolved_core_config.json`, **overwrites** `solver.mesh_file` with argument `mesh_path` (= baseline) |
| 8. DOLFINx load | `fem3d._load_mesh_and_tags(mesh_file)` | `FEM/scripts/fem_main_3d.py` | Parent 3D mesh + tags |
| 9. Shell trace submesh | `create_submesh(msh, tdim-1, shell_facets)` | trace audit | Facets tags 1+3+4 only (top/back/ribs) |
| 10. Mixed W | `_solve_coupled_evp(solve_evp=False)` | `fem_main_3d.py` | Builds coupled operators; `n_w`, `p_idx`, maps |
| 11. B3 restriction | `_build_b3_scaled_restricted_operators_in_memory` | trace audit | Fixed sparsity pattern after restriction |
| 12. Active set | `_b3_struct_active_identify_inactive_and_aup_supported` | trace audit | `active_dimension` fixed for fixed mesh |
| 13. CSR export | `_export_operators` | `v2_b3_st_worker_scaling_benchmark.py` | `A_active_csr.npz`, `M_active_csr.npz` |
| 14. Region DOFs | `_b3_capture_region_dof_indices_for_checkpoint` | trace audit | Masks on **same baseline mesh**; `u_idx_soundhole` via shell trace |
| 15. Workers | `run_checkpoint_st_target` | `v2_b3_checkpoint_solve_target_list.py` | Solve on exported `A_active`/`M_active` |

### Critical code: operator mesh is always baseline

```3415:3438:FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    sample = sample_spec_from_case(case)
    mesh_file = mesh_path(str(mesh_level), CASE_ID)
    ...
    msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    ...
    A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
        mesh_file,
        sample,
        ...
        core_config_path=core_config_path,
    )
```

`CASE_ID = "baseline_coupled_v2"` (line 56). Per-sample `{sample_id}.msh` is **never** passed here.

### Critical code: replay overwrites config mesh path

```86:88:FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_build_coupled_acoustic_seed.py
        cfg = copy.deepcopy(json.loads(config_source.read_text(encoding="utf-8")))
        sc = cfg.setdefault("solver", {})
        sc["mesh_file"] = str(mesh_path.resolve())
```

Even when `resolved_core_config.json` lists `run_root/lprod/mesh/L_prod/sample_001.msh`, assembly uses the `mesh_path` argument (= baseline).

---

## Answers to required questions

### 1. Which mesh object is used to assemble A and M?

The DOLFINx mesh loaded from:

`FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/mesh/L_prod/baseline_coupled_v2.msh`

via `mesh_path("L_prod", "baseline_coupled_v2")`.

### 2. Is it the sample-specific `lprod/mesh/L_prod/sample_XXX.msh`?

**No.** That file is built/copied for provenance and metadata (`mesh_build_summary`, `resolved_core_config.solver.mesh_file`) but **not** loaded in Stage A operator build.

### 3. Why do 176k-node and 410k-node meshes produce identical W dimensions and CSR sparsity?

Because those node counts describe the **generated Gmsh artifact**, while `n_w=378243` and `active_dimension=316017` describe the **baseline** mesh used for assembly. Different generated meshes never enter the operator.

### 4. Remeshing, morphing, canonical topology, restriction?

| Mechanism | Present? |
|-----------|----------|
| Per-sample Gmsh remesh | Yes — but **decoupled** from operator |
| Canonical topology | **Yes** — `baseline_coupled_v2` |
| Mesh morphing | No |
| Shell trace submesh (tags 1,3,4) | Yes — fixed for baseline |
| Pressure restriction / active set | Yes — fixed sparsity for baseline |

### 5. Are `mesh_build_summary` node counts a different mesh than the operator mesh?

**Yes.** Summary reflects `{sample_id}.msh` in run tree; operator uses `baseline_coupled_v2.msh`.

### 6. Are sample-specific dimensions applied to Gmsh only, or also to operator coordinates?

**Gmsh only** (and config metadata). Operator coordinates are **baseline-fixed**.

### 7. Does geometry affect matrix structure or only matrix values?

With current wiring: **structure fixed**, **values** change via materials (`resolved_core_config.json` wood properties). LHS `geometry.*` in config does **not** alter FEM coordinates when `core_config_path` is supplied.

### 8. Is fixed sparsity intentional and mathematically valid?

**Intentional** for the B3 benchmark's fixed-topology + material-sensitivity model. **Not mathematically valid** for a geometry-varying LHS guitar ROM where cavity volume, hole area, and body modes should change with `length/width/depth/hole_radius`.

---

## Fixed CSR topology — exact explanation

The active operators `A_active_csr.npz` / `M_active_csr.npz` store:

| Component | Cross-sample behavior | Cause |
|-----------|----------------------|-------|
| `shape` | Identical | Same baseline mesh → same DOF numbering after restriction |
| `indptr`, `indices` | Identical | Same sparsity pattern (connectivity) |
| `data` | Different | Material properties (ρ, E, ν) from LHS woods |
| Full file SHA256 | Different | `data` array differs |

This is **not** checkpoint file reuse. It is **one shared discrete topology** with **per-sample material overlay**.

---

## Soundhole / mic proxy root cause

### How `u_idx_soundhole` is created

During operator build (`_b3_capture_region_dof_indices_for_checkpoint`):

1. Shell trace submesh includes facet tags **1, 3, 4** (top, back, ribs).
2. Soundhole is facet tag **2** — **excluded** from shell trace.
3. `_b3_trace_region_u_rows(..., tag=TAG_SOUNDHOLE)` → **zero facets** → **empty array**.

```3424:3424:FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py
    shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
```

### Physical representation of soundhole

| Representation | Tag | Role |
|----------------|-----|------|
| Soundhole aperture | Facet tag 2 | Acoustic pressure-release boundary (`p=0` Dirichlet) |
| Top plate | Tag 1 | Structural shell |
| Air cavity | Volume tag 10 | Pressure DOFs |

The hole is **not** a structural surface in the shell trace — it is an **acoustic boundary**. Empty `u_idx_soundhole` is **expected** for structural displacement masks.

### Why `soundhole_displacement_rms_proxy_v1` is conceptually wrong here

The soundhole has **no structural DOFs** on the trace submesh. Using structural displacement at the hole is the wrong physics channel. Output should use **acoustic pressure** near the aperture (or exterior probe), not shell displacement.

### Current fallback: `cavity_pressure_max_proxy_v1`

From `v2_b3_mode_audio_coupling.py`:

```187:201:FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_mode_audio_coupling.py
    p_vals = x_full[np.asarray(region.get("p_idx_air", []), dtype=np.int32)]
    ...
    air_max = float(np.max(np.abs(p_vals))) if p_vals.size else None
    ...
    air_proxy = _safe_norm_ratio(air_max if air_max is not None else air_rms, modal_norm)
```

`_mic_output_from_proxies` selects this when `u_idx_soundhole` is empty:

```115:118:FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_mode_audio_coupling.py
    if structural_available and soundhole_rms is not None:
        return soundhole_rms, "soundhole_displacement_rms_proxy_v1", "proxy"
    if pressure_available and cavity_pressure is not None:
        return cavity_pressure, "cavity_pressure_max_proxy_v1", "proxy"
```

**What it measures:** `max |p|` over **all** cavity air pressure DOFs (`p_idx_air` ≈ 60k DOFs), normalized by full-W L2 norm.

**Not:** soundhole-local pressure, aperture RMS, or microphone location.

**Why repeated amplitudes:** global max over a fixed baseline cavity is weakly sensitive to geometry (which isn't in the operator anyway) and can tie to dominant **air modes** at similar frequencies across material-only variation.

### Replacement recommendation (smallest physically defensible fix)

**Candidate A — aperture pressure RMS** (implemented experimentally in `v2_b3_aperture_pressure_mask.py`):

- Select air-volume pressure DOFs within radius `r` of geometry-derived soundhole centre.
- `mic_output_proxy = RMS(p_probe) / ||x||_W`
- Mask is **sample-specific** (centre scales with `length`, `width`, `depth`).
- Non-empty by construction (fallback to nearest air DOFs).

**Not chosen yet:** velocity flux (needs face integral), virtual mic in exterior air (needs exterior domain coupling).

**Production wiring:** behind experimental flag / separate validation script until VM confirms sensitivity.

---

## Repeated ~281 / ~390 Hz families

With fixed baseline topology:

- Air-mode frequencies cluster near **explicit L_prod target centres** (shift-invert returns nearest eigenvalue to target).
- Frequency variation across samples is **small** (material overlay only).
- Raw catalog duplicates from overlapping chunks amplify vertical stacks in plots.
- Dedupe at 0.05 Hz merges many; does not fix wrong-physics frequencies.

---

## Existing-data salvageability

| Field | Status | Notes |
|-------|--------|-------|
| `frequency_hz` | **INVALID** for geometry-varying LHS intent | Valid only for fixed-topology material-overlay model |
| `top_share` | **LIKELY_VALID** | Region masks on baseline; materials vary |
| `back_share` | **LIKELY_VALID** | Same |
| `air_share` | **LIKELY_VALID** | Same |
| `coupling_class` | **LIKELY_VALID** | Derived from shares |
| `dominant_region` | **LIKELY_VALID** | Same |
| `radiation_proxy` | **RECOMPUTABLE** | From shares + norms if vectors absent |
| `mic_output_proxy` | **INVALID** | Wrong proxy (`cavity_pressure_max`); empty soundhole mask |
| `bridge_excitation_abs` | **LIKELY_VALID** | Bridge/top proxy on baseline topology |

| Artifact | Salvage action |
|----------|----------------|
| `modes_catalog.jsonl` | **Regenerate** after rerun; optional patch script for audio fields only if vectors retained (not typical post-compaction) |
| ROM training tensors | **Discard** after full rerun — frequencies and mic fields both wrong for intended model |
| `aggregation/` metadata | Keep for audit; do not use for production ROM |
| Checkpoints sample_000/001/034/035 | Useful for validation only |

**Can old frequencies be retained if only mic proxy changes?** **No** — root cause is operator topology, not proxy alone.

---

## Minimal validation implementation (experimental only)

### Scripts (not wired to production)

| Script | Purpose |
|--------|---------|
| `audit_m4_operator_provenance.py` | Test A — provenance + CSR structure/value hashes |
| `v2_b3_aperture_pressure_mask.py` | Build `p_idx_aperture` probe mask |
| `v2_b3_mode_audio_coupling_experimental.py` | Aperture RMS mic proxy |
| `run_m4_geometry_audio_validation.py` | Orchestrates Test A + Test B |

### VM commands

**Test A — operator provenance (no solve):**

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_m4_operator_provenance.py \
  --samples sample_000,sample_001,sample_034,sample_035 \
  --dolfinx \
  --json-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_OPERATOR_PROVENANCE_AUDIT.json
```

**Test A + B — full validation orchestrator:**

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_geometry_audio_validation.py \
  --use-sample-mesh-for-mask \
  --json-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_GEOMETRY_AUDIO_VALIDATION.json
```

**Test B — narrow-band solve (per extreme sample, after mask built):**

```bash
# Commands emitted in M4_GEOMETRY_AUDIO_VALIDATION.json → test_b_narrow_band_commands
# Uses v2_b3_checkpoint_solve_target_list.py with targets 272–286 Hz and 382–396 Hz only
```

### Expected healthy validation outcomes

| Check | Pass criterion |
|-------|----------------|
| Provenance | `mesh_mismatch_flag=true`; generated nodes differ, `n_w` identical |
| Aperture mask | `n_p_aperture_dofs > 0`; `aperture_index_sha256` differs between extremes (with sample mesh) |
| Mic proxy | `experimental_mic_ratio_vs_legacy` not ≈1.0 across extremes |
| Frequencies | After Stage A mesh fix: ~281 Hz family shifts with Helmholtz-scale geometry |

---

## Task 5 — Production decision

### **`RERUN_ALL_35_SAMPLES`**

| Item | Detail |
|------|--------|
| **Invalidation reason** | Stage A assembles on `baseline_coupled_v2` regardless of LHS geometry; generated per-sample meshes are not used. Eigenfrequencies do not reflect intended geometry-varying guitar ROM. |
| **Invalid fields** | `frequency_hz` (primary); `mic_output_proxy` (wrong channel); ROM intensity targets derived from mic |
| **Likely valid / secondary** | Wood-sensitive structural shares on fixed topology (still wrong for geometry study) |
| **Old ROM** | **Discard** — training inputs encode wrong physics |
| **Estimated runtime** | ~45 min/sample × 35 ≈ **26 hours** (parallelizable) |
| **Safe migration** | 1) Fix Stage A to load sample mesh from `resolved_core_config` 2) Wire aperture pressure proxy 3) Run validation on 2 extremes 4) Rerun 000–035 5) Do **not** resume `sample_036` until fix verified 6) Rebuild ROM after new catalogs |

### Required code fixes (post-validation, not in this change)

1. `run_v2_B3_trace..._audit.py`: resolve `mesh_file` from M4 `core_config_path` / run-tree sample mesh when geometry ≠ baseline fingerprint.
2. `v2_b3_synthesis_export.py` / region capture: add `p_idx_aperture` from pressure field near tag-2 facets.
3. `v2_b3_mode_audio_coupling.py`: prefer aperture pressure RMS over cavity max (production flag after validation).

---

## What we are **not** claiming

- All 35 samples are "corrupt" in a file-reuse sense — they are **internally consistent** with the **current** (fixed-topology) implementation.
- `sample_036` should be modified, resumed, or compacted.
- Existing catalogs should be deleted automatically.

---

## Related artifacts

- `M4_REPEATED_AIR_MODE_FAMILY_AUDIT.{md,json,csv}` — catalog-level cluster audit
- `M4_OPERATOR_PROVENANCE_AUDIT.json` — output of provenance script (VM)
- `M4_GEOMETRY_AUDIO_VALIDATION.json` — validation orchestrator output (VM)
