# B3 M2.4 — material overlay execution contract / readiness check (planning-only)

## 1. Purpose

M2.3 defined real near-baseline `material_delta` values in the pilot JSONL. M2.4 answers whether Stage A can apply those deltas today, and defines the safest contract before any execution.

**Constraints (unchanged):**

- no Stage A/B/C execution in this step,
- no runtime manifests,
- no full orchestrator implementation,
- no cleanup/move/archive.

---

## 2. Executive summary

| Question | Answer |
|----------|--------|
| Can `v2_b3_checkpoint_export.py` apply JSONL `material_delta` today? | **No** |
| Can deltas be applied without editing canonical `coupled_physical_core_v2.json`? | **Yes** — via per-run resolved config files + a new `--core-config` (or equivalent) path into assembly |
| Safest first execution path | Shared `L_prod` mesh + material-only resolved configs + distinct checkpoint dirs per `sample_id` |
| Exact next step | Implement **dry-run-only** config resolver + readiness checker (M2.4.1), then update preview/manifest contract; only then run one sample Stage A |

---

## 3. Current Stage A config-loading path (as implemented)

Stage A (`v2_b3_checkpoint_export.py`) accepts only:

- `--mesh-level` (`L_prod` for pilot),
- `--output-dir`,
- compose backend / rich / synthesis-region-dofs flags.

It does **not** accept JSONL, `material_delta`, or a config path.

Operator build chain:

1. `v2_b3_checkpoint_export.py` → `audit._b3_build_corrected_structural_active_operators(mesh_level=...)`
2. That function loads `v2_mesh_convergence_manifest.json`, selects fixed case `baseline_coupled_v2`, builds `sample = sample_spec_from_case(case)` (geometry + `top_wood_id` / `back_wood_id` only).
3. Mesh file: `v2_mesh_convergence/mesh/<mesh_level>/baseline_coupled_v2.msh` (via `mesh_path(mesh_level, CASE_ID)`).
4. Assembly: `v2_build_coupled_acoustic_seed._assemble_reduced_coupled_replay(mesh_file, sample, ...)` which:
   - `copy.deepcopy(json.loads(V2_CONFIG))` where `V2_CONFIG = configs/coupled_physical_core_v2.json`,
   - sets `solver.mesh_file` to the mesh path,
   - applies `sample_geometry(sample)` from manifest case geometry,
   - applies `apply_wood_ids_to_config` **only if** `sample.materials` contains `top_wood_id` / `back_wood_id` (replaces full plate blocks from `wood_library`).

**Implication:** Pilot JSONL `parameter_payload.material_delta` is **not read anywhere** in the Stage A path. All three pilot samples would currently build **identical** operators (same mesh case, same baseline core config, same spruce/rosewood IDs).

M2.2 preview commands also omit any config/overlay argument — they only pass `--mesh-level` and `--output-dir`.

---

## 4. Answers to readiness questions

### Q1. Does Stage A support per-sample material overlay from JSONL?

**No.**

There is no JSONL ingestion, no `material_delta` merge, and no `--core-config` flag on `v2_b3_checkpoint_export.py`.

### Q2. Safest minimal way to apply deltas without modifying canonical core config?

**Recommended pattern (matches existing LHS `mesh_sync` / `pipeline_merged_configs` precedent):**

1. Keep `configs/coupled_physical_core_v2.json` **read-only** (canonical baseline).
2. For each `sample_id` / `run_id`, write a **resolved config** under an ignored runtime path (see Q4).
3. Resolve by:
   - `copy.deepcopy(baseline core config)`,
   - shallow/deep merge of `material_delta` into `materials.top` / `materials.back` (and later `geometry_delta` into `geometry` if approved),
   - set `solver.mesh_file` to the shared `L_prod` `baseline_coupled_v2.msh` path (repo-relative in file).
4. Plumb resolved path into assembly via a **single new optional argument** on Stage A (e.g. `--core-config <path>`), threaded to `_assemble_reduced_coupled_replay` so it loads that file instead of hardcoded `V2_CONFIG`.

**Do not** hand-edit `coupled_physical_core_v2.json` per sample.

**Alternative (not recommended for first pilot):** extend manifest `cases[]` with per-sample clones — pollutes tracked convergence manifest and mixes planning with production mesh suite.

### Q3. Create per-sample resolved configs under runtime/ignored path before Stage A?

**Yes.** This is the preferred contract.

### Q4. Exact location

Use:

`FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/<run_id>/`

Per sample:

| File | Role |
|------|------|
| `resolved_core_config.json` | Full effective config passed to Stage A assembly |
| `overlay_applied.json` | Small provenance record: base path, `material_delta`, `geometry_delta`, merge timestamp, content hash |
| `readiness_check.json` | Dry-run validator output (densities, mesh path, flags) |

`<run_id>` should match pilot `sample_id` for the 3-sample pilot (`lhs_pilot_001_timing`, etc.).

Add to `.gitignore` (when implementing):

`FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/`

### Q5. Tracked or ignored?

**Ignored (runtime-generated).**

Tracked sources remain:

- `configs/coupled_physical_core_v2.json` (baseline),
- `pipeline_runs/specs/m2_1_pilot_3_samples.jsonl` (declared deltas + policy flags).

Optional: commit a **spec example** under `pipeline_runs/specs/examples/resolved_core_config_lhs_pilot_001_timing.json` later — not required for first execution.

### Q6. Run manifest recording (M1 extension)

When runtime manifests are created (post–M2.4, not now), extend `source` (or add `config_overlay`) with:

```json
{
  "core_config_base": "FEM/experiments/.../configs/coupled_physical_core_v2.json",
  "mesh_convergence_manifest": "FEM/experiments/.../configs/v2_mesh_convergence_manifest.json",
  "mesh_level": "L_prod",
  "mesh_case_id": "baseline_coupled_v2",
  "parameter_payload": { "material_delta": { "top": { "density": 445.5 } }, "geometry_delta": {} },
  "resolved_core_config": "FEM/experiments/.../pipeline_runs/config_overlays/lhs_pilot_001_timing/resolved_core_config.json",
  "overlay_applied_json": "FEM/experiments/.../pipeline_runs/config_overlays/lhs_pilot_001_timing/overlay_applied.json",
  "material_applied": {
    "top.density": 445.5,
    "back.density": 830.0
  },
  "resolved_config_sha256": "<hex>",
  "requires_mesh_regeneration": false
}
```

Record **effective** densities after merge (not delta alone) so downstream audit does not require re-merging.

### Q7. What should the Stage A command reference?

**Resolved config path only** (plus existing flags):

```bash
python .../v2_b3_checkpoint_export.py \
  --mesh-level L_prod \
  --core-config "FEM/experiments/.../pipeline_runs/config_overlays/lhs_pilot_001_timing/resolved_core_config.json" \
  --B3-block-compose-backend csr_bulk \
  --B3-synthesis-region-dofs off \
  --output-dir "FEM/experiments/.../v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_lhs_pilot_001_timing"
```

Do **not** rely on “base + overlay” at execution time unless the orchestrator always writes the resolved file first and Stage A reads **only** the resolved file (simpler, auditable).

M2.2 preview should be updated (later) to include `--core-config` in `predicted_commands.stage_a` once the flag exists.

### Q8. Verify resolved config before execution (no DOLFINx)

Dry-run checks (no Stage A build):

1. **Resolver script** (proposed `scripts/v2_b3_resolve_pilot_core_config.py`, M2.4.1): read JSONL row → write `resolved_core_config.json` + `overlay_applied.json`.
2. **Assert effective values:**
   - `lhs_pilot_001_timing`: `materials.top.density == 445.5`, `materials.back.density == 830.0` (unchanged)
   - `lhs_pilot_002_timing`: `materials.back.density == 842.45`, `materials.top.density == 450.0`
   - `lhs_pilot_003_synthesis`: `materials.top.density == 456.75`, `materials.back.density == 830.0`
3. **Assert mesh unchanged:** `solver.mesh_file` points to `.../v2_mesh_convergence/mesh/L_prod/baseline_coupled_v2.msh` (or equivalent resolved path), same for all three samples.
4. **Assert `requires_mesh_regeneration == false`** in overlay record matches empty `geometry_delta`.
5. **Hash:** store `sha256` of resolved JSON in `readiness_check.json` for manifest linkage.
6. **Re-run M2.2 preview** after preview script knows `--core-config` path template.

No `fem3d`, no PETSc, no subprocess mesh build.

### Q9. Does density-only change require operator rebuild in Stage A?

**Yes.**

Mass matrix `M` (and potentially `A` through coupling) depend on density in the weak form. Stage A performs full coupled replay assembly and structural active reduction — this is a **full operator rebuild**, not a metadata-only tweak.

Stage B/C consume exported `A_active` / `M_active` from that checkpoint; they do not re-apply material deltas.

### Q10. Does density-only avoid mesh regeneration?

**Yes, for this pilot**, provided:

- `geometry_delta` remains `{}`,
- all samples use the same existing `L_prod` mesh file for case `baseline_coupled_v2`,
- `requires_mesh_regeneration` stays `false`.

Density does not change CAD or mesh topology; only constitutive parameters at assembly time change.

### Q11. Cached metadata / checkpoint reuse risks

| Risk | Mitigation |
|------|------------|
| Reusing same `--output-dir` across samples with different materials | **Forbidden.** One checkpoint dir per `sample_id` (already in M2.2 path template). |
| Reusing a baseline PASS checkpoint for a perturbed sample | **Forbidden.** Stage B solves are tied to exported operators; material change requires new Stage A export. |
| Same mesh file on disk, different materials | **OK** — assembly reads resolved config each run; operators differ if config differs. |
| `built_metadata.json` lacks material fingerprint | **Gap today.** Recommend recording `materials.top.density`, `materials.back.density`, and `resolved_config_sha256` in `built_metadata.json` / `checkpoint_export_manifest.json` when overlay support is implemented. |
| Manifest case `baseline_coupled_v2` wood IDs overriding deltas | **Watch:** assembly still applies `apply_wood_ids_to_config` from manifest sample **after** loading core config. For density-only deltas on spruce/rosewood baseline, manifest IDs match baseline — no conflict. If future deltas change `E_L` etc. without changing wood ID, merge order must be: baseline → wood IDs from manifest (if any) → `material_delta` overrides (delta wins on overlapping keys). |
| Stage C / synthesis metadata | Stage C uses checkpoint + rich modal artifacts; does not re-read material config. Physical interpretation of synthesis proxies is only meaningful if Stage A used the correct resolved config. |

### Q12. Dry-run-only validation before first execution

**Gate checklist (all must pass before first Stage A):**

| # | Check | Tool / action |
|---|--------|----------------|
| 1 | JSONL parses; each row has non-empty `material_delta` | existing JSONL |
| 2 | Resolver produces 3 `resolved_core_config.json` files | **M2.4.1** (proposed, not implemented) |
| 3 | Effective densities match M2.3 table | readiness script / `jq` |
| 4 | `geometry_delta` empty; `requires_mesh_regeneration: false` | readiness script |
| 5 | `solver.mesh_file` identical across samples (L_prod baseline mesh) | readiness script |
| 6 | Canonical `coupled_physical_core_v2.json` unchanged (git clean) | `git status` |
| 7 | M2.2 preview updated to show `--core-config` per sample | after flag + preview update |
| 8 | Distinct `output-dir` per `sample_id` in preview | re-run preview |
| 9 | No runtime manifests under `pipeline_runs/manifests/` until deliberate registration | manual / gitignore |
| 10 | Documented merge order and wood-ID interaction | this contract |

**Explicitly out of scope for dry-run gate:** Stage A/B/C execution, mesh rebuild, runtime manifest creation.

---

## 5. Mapping: JSONL deltas → config artifacts → Stage A

| Layer | Current role | M2.4 contract |
|-------|----------------|---------------|
| `guitar_3d.json` | Canonical GUI / legacy FEM pipeline base | **Not** Stage A input for B3 checkpoint path; reference only |
| `v2_mesh_convergence_manifest.json` | Selects mesh level + case `baseline_coupled_v2` (geometry + wood IDs) | Unchanged for pilot; supplies mesh path + case geometry/wood IDs |
| `coupled_physical_core_v2.json` | Baseline physics/solver template in `_assemble_reduced_coupled_replay` | **Base only**; never mutated per sample |
| `m2_1_pilot_3_samples.jsonl` | Declares `material_delta` + policy flags | Source of truth for deltas |
| `pipeline_runs/config_overlays/<run_id>/resolved_core_config.json` | *(not yet)* | **Stage A effective input** after resolver |
| Stage A command | `--mesh-level` only today | Add `--core-config` → resolved file |

**Merge order (normative when implemented):**

1. Load `coupled_physical_core_v2.json` (deep copy).
2. Apply manifest case geometry via `sample_geometry(sample)` (pilot: unchanged).
3. Apply manifest wood IDs via `apply_wood_ids_to_config` if present (pilot: spruce + rosewood, baseline-compatible).
4. Apply JSONL `material_delta` (shallow merge per plate: `top`, `back`).
5. Set `solver.mesh_file` to `mesh_path(L_prod, baseline_coupled_v2)`.
6. Write result to `resolved_core_config.json` (dry-run) or pass to assembly (execution).

---

## 6. Pilot sample expected resolved densities

| sample_id | material_delta | Expected `materials.top.density` | Expected `materials.back.density` | Remesh |
|-----------|----------------|-------------------------------------|-------------------------------------|--------|
| `lhs_pilot_001_timing` | `top.density = 445.5` | 445.5 | 830.0 (baseline) | no |
| `lhs_pilot_002_timing` | `back.density = 842.45` | 450.0 (baseline) | 842.45 | no |
| `lhs_pilot_003_synthesis` | `top.density = 456.75` | 456.75 | 830.0 (baseline) | no |

Baseline reference: `coupled_physical_core_v2.json` (`top` 450.0, `back` 830.0).

---

## 7. Required minimal implementation (later — not in M2.4)

**Phase M2.4.1 (dry-run only, smallest):**

- `scripts/v2_b3_resolve_pilot_core_config.py`
  - inputs: `--samples-jsonl`, `--sample-id` or all rows, `--output-root pipeline_runs/config_overlays`
  - outputs: `resolved_core_config.json`, `overlay_applied.json`, `readiness_check.json`
  - no DOLFINx, no PETSc, no mesh build

**Phase M2.4.2 (execution enablement):**

- Add `--core-config` to `v2_b3_checkpoint_export.py`
- Thread optional config path into `_b3_build_corrected_structural_active_operators` → `_assemble_reduced_coupled_replay` (replace hardcoded `V2_CONFIG` read when set)
- Record material fingerprint + `resolved_config_sha256` in `checkpoint_export_manifest.json` / `built_metadata.json`
- Update `v2_b3_lhs_orchestrator_preview.py` Stage A command template
- Add `.gitignore` entry for `pipeline_runs/config_overlays/`

**Reuse opportunity:** merge logic can mirror `wood_library.apply_lhs_parameters_to_config` dotted-key style, or a small `deep_merge(materials, material_delta)` helper limited to `materials` / `geometry` subtrees.

---

## 8. Safest first execution path (after gates)

1. Run M2.4.1 resolver for all 3 samples; inspect `readiness_check.json`.
2. Register run manifests (optional) referencing resolved config paths — still no stage execution.
3. Execute **one** timing sample first (`lhs_pilot_001_timing`): Stage A only → verify manifest + `built_metadata` material fingerprint → Stage B → compare scalars to baseline expectation (small shift, not identical).
4. If PASS, run `lhs_pilot_002_timing` (Stage A+B).
5. If PASS, run `lhs_pilot_003_synthesis` (A+B rich + C).

Keep shared mesh; distinct checkpoint/solve dirs per `sample_id`.

---

## 9. Final recommendation

| Item | Decision |
|------|----------|
| Current code applies JSONL deltas? | **No** |
| Canonical core config | **Do not modify** |
| Per-sample resolved configs | **Yes**, under ignored `pipeline_runs/config_overlays/<run_id>/` |
| Stage A reference | **Resolved config path** via new `--core-config` |
| First physical pilot mesh strategy | **Same L_prod mesh**, material-only (per M2.3) |
| Exact next step | **M2.4.1:** implement dry-run resolver + readiness checker; run on VM for all 3 samples; only then implement `--core-config` and update M2.2 preview |

---

## 10. Related documents

- [`B3_M2_3_PILOT_PERTURBATION_POLICY.md`](B3_M2_3_PILOT_PERTURBATION_POLICY.md) — perturbation values and mesh policy
- [`B3_M2_2_DRY_RUN_ORCHESTRATOR_PREVIEW.md`](B3_M2_2_DRY_RUN_ORCHESTRATOR_PREVIEW.md) — command preview (needs `--core-config` after M2.4.2)
- [`B3_M1_PIPELINE_RUN_MANIFEST_SPEC.md`](B3_M1_PIPELINE_RUN_MANIFEST_SPEC.md) — manifest schema to extend with overlay provenance
- [`B3_OFFICIAL_RICH_PIPELINE_COMMANDS.md`](B3_OFFICIAL_RICH_PIPELINE_COMMANDS.md) — validated stage commands (baseline config implicit today)
