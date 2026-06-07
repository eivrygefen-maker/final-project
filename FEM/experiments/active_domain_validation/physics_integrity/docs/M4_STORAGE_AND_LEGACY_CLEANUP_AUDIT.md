# M4 Storage and Legacy Cleanup Audit

**Date:** 2026-06-02  
**Status:** Audit complete — **no deletions performed**. Compaction tooling ready for VM dry-run.  
**Principle:** Preserve ROM/FOM/LHS/STK paths; archive heavy solve artifacts to shared storage.

---

## Executive summary

| Finding | Detail |
|---------|--------|
| **Disk pressure** | VM root ~53G/59G used; `pipeline_runs/guitars` ≈ **15G** of ~25G `FEM/` |
| **Per completed sample** | ~380–675 MB total; **~85–95%** is mesh + checkpoint + scout mesh |
| **ROM minimum** | `aggregation/modes_catalog.jsonl` + `ROM/classic/lhs_pool.json` (+ trained surrogate for compare) |
| **Safe compaction** | Completed `AGGREGATION_PASS` runs: archive/delete heavy dirs after verified `tar.zst` |
| **Do not touch** | `RUNNING` / `FAILED` / partial / resume-needed runs |
| **Estimated reclaim** | ~**12–16 GiB** local for ~30 completed samples (conservative VM estimate) |
| **Shared budget** | Target 20–40 GiB long-term across shapes; zstd archives ~40–60% of heavy raw |

**Tools added:**

| File | Role |
|------|------|
| `scripts/compact_completed_m4_runs.py` | Completed-run archive/compaction (default dry-run) |
| `scripts/audit_m4_legacy_storage.py` | Legacy/large-file worktree audit (read-only) |

---

## VM storage measurements (operator-reported)

```text
/dev/sda3: 59G total, ~53G used, ~3.8G available
~/final-project/FEM ≈ 25G
pipeline_runs ≈ 15G
pipeline_runs/guitars ≈ 15G
/media/sf_gmar ≈ 168G free (shared; budget ~20–40G project archives)
```

Typical completed sample (`sample_000`):

| Path | Size |
|------|------|
| `lprod/mesh` | ~98 MB |
| `lprod/checkpoint` | ~201 MB |
| `scout` | ~61 MB |
| `worker_results` | ~3 MB |
| `aggregation` | ~2 MB |
| `rom` | ~3 MB |
| **Total** | ~380–675 MB |

---

## Classification legend

| Class | Meaning |
|-------|---------|
| **ACTIVE_REQUIRED** | Current M4/ROM/LHS pipeline reads; keep local |
| **ACTIVE_OPTIONAL** | Helpful for ops/audit; small; keep unless desperate |
| **ARCHIVE_RECOMMENDED** | Heavy; archive to shared then may delete locally |
| **SAFE_TO_DELETE_AFTER_COMPLETION** | Removable after verified archive + completed run |
| **LEGACY_SAFE_TO_REMOVE** | Not used by M4; remove from worktree after approval |
| **UNKNOWN_REVIEW_REQUIRED** | Manual decision |

---

## Per-run directory audit (`pipeline_runs/guitars/<sample>/runs/<run_id>/`)

Code-audited readers: `build_m4_rom_from_completed_fom.py`, `run_m4_rom_compare.py`, `v2_b3_m4_modal_surrogate_lib.py`, `v2_b3_m4_rom_fom_compare_lib.py`, `v2_b3_m4_aggregate_worker_results.py`, `v2_b3_m4_lhs_pool_bridge.py`, `v2_b3_m4_shared_export.py`, STK audit scripts.

| Path / artifact | Typical size | M4 pipeline | ROM train | ROM compare | Resume | Git | Class | Action |
|-----------------|-------------|-------------|-----------|-------------|--------|-----|-------|--------|
| `aggregation/modes_catalog.jsonl` | 0.5–3 MB | Output | **Required** | **Required** | No | Ignored | **ACTIVE_REQUIRED** | **Keep local always** |
| `aggregation/aggregation_result.json` | <100 KB | Read | Gate | **Required** | Summary | Ignored | **ACTIVE_REQUIRED** | Keep |
| `aggregation/modes_summary.json` | <500 KB | Read | No | Indirect | No | Ignored | **ACTIVE_REQUIRED** | Keep |
| `aggregation/runtime_summary.json` | <100 KB | Read | No | Optional | No | Ignored | **ACTIVE_OPTIONAL** | Keep |
| `aggregation/warnings_and_failures.json` | <100 KB | Export | No | No | Debug | Ignored | **ACTIVE_OPTIONAL** | Keep |
| `aggregation/mode_frequency*.png` | 0.1–2 MB | Export | No | No | No | Ignored | **ACTIVE_OPTIONAL** | Keep |
| `aggregation/partial_*` | varies | Partial | No | No | **Resume** | Ignored | **ACTIVE_REQUIRED** | Never compact partial |
| `rom/rom_fom_comparison.json` | <1 MB | Written | No | Output | No | Ignored | **ACTIVE_REQUIRED** | Keep |
| `rom/rom_prediction_pre_fom.json` | <1 MB | Cache | No | Optional | No | Ignored | **ACTIVE_OPTIONAL** | Keep |
| `freeze/*` | <1 MB | Written | No | No | Milestone | Ignored | **ACTIVE_REQUIRED** | Keep |
| `logs/**` | 0.1–5 MB | Ops | No | No | Debug | Ignored | **ACTIVE_OPTIONAL** | Keep |
| `sample/sample_input.json` | <10 KB | Stage 0 | Param override | Optional | **Resume** | Ignored | **ACTIVE_REQUIRED** | Keep |
| `sample/resolved_core_config.json` | <100 KB | Stage 0 | No | No | **Resume** | Ignored | **ACTIVE_OPTIONAL** | Keep (small) |
| `pipeline_run_manifest.json` | <50 KB | All stages | No | Indirect | **Resume** | Ignored | **ACTIVE_REQUIRED** | Keep |
| `m4_sample_runtime_provenance.json` | <50 KB | Batch | No | No | Resume | Ignored | **ACTIVE_OPTIONAL** | Keep |
| `scout/scout_plan.json` etc. | <1 MB total | Scout | No | No | **Resume** | Ignored | **ACTIVE_OPTIONAL** | Keep metadata; archive meshes |
| `scout/mesh/**` | 30–80 MB | Scout | No | No | **Resume** | Ignored | **ARCHIVE_RECOMMENDED** | Archive+delete when complete |
| `scout/checkpoint/**` | 10–40 MB | Scout | No | No | **Resume** | Ignored | **ARCHIVE_RECOMMENDED** | Archive+delete when complete |
| `scout/discovery/**` | 1–20 MB | Scout | No | No | Resume | Ignored | **ARCHIVE_RECOMMENDED** | Archive+delete when complete |
| `lprod/lprod_target_plan.json` | <100 KB | Workers/agg | No | No | **Resume** | Ignored | **ACTIVE_OPTIONAL** | Keep (small) |
| `lprod/worker_chunk_plan.preview.json` | <100 KB | Workers/agg | No | No | **Resume** | Ignored | **ACTIVE_OPTIONAL** | Keep (small) |
| `lprod/mesh/**` | 80–120 MB | Checkpoint | No | No | **Resume** | Ignored | **SAFE_TO_DELETE_AFTER_COMPLETION** | Archive+delete |
| `lprod/checkpoint/**` | 150–250 MB | Workers | No | No | **Resume** | Ignored | **SAFE_TO_DELETE_AFTER_COMPLETION** | Archive+delete |
| `lprod/lprod_checkpoint_verify.json` | ~17 MB | Verify | No | No | Debug | Ignored | **SAFE_TO_DELETE_AFTER_COMPLETION** | Archive+delete |
| `worker_results/**` | 2–10 MB | Aggregation input | No* | No | Re-aggregate | Ignored | **SAFE_TO_DELETE_AFTER_COMPLETION** | Archive+delete |

\*After `AGGREGATION_PASS`, ROM reads **catalog only**, not worker JSON. Re-aggregation requires restore from archive.

### Heavy checkpoint file breakdown (example `sample_000`)

| File | ~Size | Needed after completion? |
|------|-------|--------------------------|
| `A_active.petsc.bin` | 68 MB | No (for ROM) |
| `M_active.petsc.bin` | 66 MB | No |
| `A_active_csr.npz` | 38 MB | No |
| `M_active_csr.npz` | 16 MB | No |
| `built_metadata.json` | 13 MB | No |
| Duplicate PETSc + CSR | — | **Redundant formats**; all archived together |

---

## `ROM/classic/` audit

| Path | M4 | ROM | Class | Action |
|------|----|-----|-------|--------|
| `lhs_pool.json` | **Required** | **Required** | **ACTIVE_REQUIRED** | Never delete |
| `m4_modal_surrogate.json` / `.npz` | Output | **Required** compare | **ACTIVE_REQUIRED** | Keep |
| `rom_model_manifest.json` | Output | **Required** | **ACTIVE_REQUIRED** | Keep |
| `comparisons/**` | Output | Audit/history | **ACTIVE_OPTIONAL** | Keep (small) |
| `experimental_v22/` / `v22b/` | Diagnostics | Experimental | **ACTIVE_OPTIONAL** | Keep |
| `reduced_basis.npz` | Legacy POD | Fallback only | **LEGACY_SAFE_TO_REMOVE** | Move to legacy archive if present |
| `snapshots/snapshot_*.npz` | Legacy FOM | Legacy basis | **LEGACY_SAFE_TO_REMOVE** | Move to shared legacy |

**ROM rebuild after compaction:** `build_m4_rom_from_completed_fom.py` needs only `lhs_pool.json` + per-sample `modes_catalog.jsonl`. **Confirmed — checkpoint/mesh not required.**

---

## `pipeline_runs/` (non-guitars)

| Path | Size (VM) | M4 | Class | Action |
|------|-----------|----|-------|--------|
| `batches/**` | varies | Batch resume | **ARCHIVE_RECOMMENDED** | Keep recent; archive old summaries |
| `specs/generated/**` | small | Spec replay | **ACTIVE_OPTIONAL** | Keep |
| `specs/m4_*`, `schemas/m4/` | small | Contracts | **ACTIVE_REQUIRED** | Keep in Git |
| `index/lhs_pool_status.json` | small | Resume sidecar | **ACTIVE_OPTIONAL** | Keep |
| `index/lhs_production_runs_index.jsonl` | small | Audit | **ACTIVE_OPTIONAL** | Keep |
| `config_overlays/**` | varies | Pilot overlays | **SAFE_TO_DELETE_AFTER_COMPLETION** | Delete after promotion |
| `logs/**` | varies | Ops | **ARCHIVE_RECOMMENDED** | Rotate/archive |
| `scout_density_reports/**` | varies | Reports | **LEGACY_SAFE_TO_REMOVE** | Archive or delete |

---

## Legacy / non-M4 worktree (Deliverable 3)

Run: `python FEM/.../scripts/audit_m4_legacy_storage.py` → `pipeline_runs/index/legacy_storage_audit.json`

| Area | Class | Action |
|------|-------|--------|
| `FEM/SORTING/**` | **LEGACY_SAFE_TO_REMOVE** | Safe to remove from worktree (legacy packaging workspace) |
| `v2_b3_m3_orchestrator*.py` | **MOVE_TO_LEGACY_ARCHIVE** | Keep in Git history; not production |
| `v2_b3_run_coarse_scout_lhs_batch.py` | **MOVE_TO_LEGACY_ARCHIVE** | M3.4 overnight scout only |
| `v2_b3_lhs_orchestrator_*.py` | **MOVE_TO_LEGACY_ARCHIVE** | Superseded by M4 dry-run/batch |
| `run_v2_B3_trace_*.py` | **KEEP_FOR_HISTORY** | Large audits; not production |
| `solver_benchmarks/**` | **LEGACY_SAFE_TO_REMOVE** | Dev benchmarks; not LHS |
| `FEM/scripts/run_pipeline.py` + legacy ROM | **KEEP_FOR_HISTORY** | Legacy snapshot/POD path |
| `ROM/classic/snapshots/` | **MOVE_TO_LEGACY_ARCHIVE** | Legacy basis inputs |
| Duplicate `config_overlays/lhs_pilot_*` | **SAFE_TO_DELETE_AFTER_COMPLETION** | Pilot timing artifacts |

**No code or legacy outputs deleted by this task.**

---

## Compaction tool behavior

### Eligibility (all must pass)

```text
LHS status == COMPLETED
last_aggregation_status == AGGREGATION_PASS (via aggregation_result.json)
modes_catalog.jsonl exists and non-empty
final_aggregation_ready, no missing/failed chunks
not RUNNING / FAILED / resume-needed
```

ROM/FOM comparison: reported if missing; **does not block** compaction.

### Local retention (always)

```text
aggregation/   (full tree incl. modes_catalog.jsonl, plots, summaries)
rom/
freeze/
logs/
sample/
pipeline_run_manifest.json
m4_sample_runtime_provenance.json
small scout/lprod JSON plans
compaction/compaction_manifest.json
```

### Heavy archive targets

```text
lprod/mesh/
lprod/checkpoint/
lprod/lprod_checkpoint_verify.json
scout/mesh/
scout/checkpoint/
scout/discovery/
worker_results/
```

### Archive layout

```text
/media/sf_gmar/classic/archives/
  sample_XXX__<run_id>__heavy.tar.zst
  sample_XXX__<run_id>__heavy.tar.zst.sha256
  sample_XXX__<run_id>__heavy.tar.zst.contents.txt
```

Per-run manifest: `<run_dir>/compaction/compaction_manifest.json`

### Safety

- Default `--dry-run` (no writes except report)
- `--delete-heavy-after-verify` requires successful archive + checksum + `tar -tf` + ROM-required files present
- Refuses delete if shared root missing/unwritable
- Idempotent if manifest + archive already verified
- No symlink traversal; deletes only under resolved run root

---

## Representative full-retention recommendation

**Operator must approve** before passing `--keep-full-samples`.

| Sample | Reason |
|--------|--------|
| `sample_001` | M4 reference E2E / freeze gold run |
| `sample_000` | First LHS anchor; ROM diagnostics baseline |
| `sample_005` | ROM v2.2b STK holdout |
| `sample_013` | ROM v2.2b STK holdout |
| `sample_024` | ROM v2.2b STK holdout |
| `sample_027` | ROM v2.2b STK holdout |

**Suggested policy for initial cleanup:**

```text
--keep-full-latest 2          # newest completed stay fully local
--keep-full-samples sample_001,sample_000,sample_005,sample_013,sample_024,sample_027
```

This keeps **6 named + 2 latest** (overlap likely) ≈ **5–8 full runs** — within the 5–10 representative target.

---

## Estimated space reclaim (VM)

Assumptions: ~30 completed samples in `0–29`, ~550 MB archivable each, 6–8 kept full (~600 MB each).

| Scenario | Local freed | Shared archive (zstd) |
|----------|-------------|------------------------|
| 30 samples compacted, 8 kept full | **~12–14 GiB** | **~6–9 GiB** |
| 35 samples (`0–34`), same policy | **~14–16 GiB** | **~7–10 GiB** |
| Per sample (heavy only) | **~350–600 MiB** | **~150–350 MiB** |

Retained per compacted sample: **~10–25 MiB** (`aggregation` + `rom` + `freeze` + metadata).

---

## Commands

### 1. Dry-run (required first)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/compact_completed_m4_runs.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --shape-name classic \
  --sample-range 0-34 \
  --shared-root /media/sf_gmar \
  --keep-full-latest 2 \
  --dry-run
```

### 2. Legacy worktree audit (optional)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_m4_legacy_storage.py
```

### 3. Archive + delete (after review)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/compact_completed_m4_runs.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --shape-name classic \
  --sample-range 0-34 \
  --shared-root /media/sf_gmar \
  --keep-full-latest 2 \
  --keep-full-samples sample_001,sample_000,sample_005,sample_013,sample_024,sample_027 \
  --archive-heavy \
  --delete-heavy-after-verify
```

(`--archive-heavy` disables dry-run automatically unless `--dry-run` is also passed.)

### 4. Post-compaction ROM verification

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_m4_rom_from_completed_fom.py \
  --lhs-json ROM/classic/lhs_pool.json --shape-name classic --completed-only

python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_rom_compare.py \
  --lhs-json ROM/classic/lhs_pool.json --max-samples 3
```

---

## Cannot safely delete (warnings)

| Item | Why |
|------|-----|
| `aggregation/modes_catalog.jsonl` | Sole ROM/FOM training/compare source |
| `ROM/classic/lhs_pool.json` | Sample parameters, completion status, run IDs |
| `ROM/classic/m4_modal_surrogate.*` | Production ROM model |
| Incomplete / failed / running runs | Resume and debugging |
| `worker_results/` **without archive** | Cannot re-aggregate if catalog corrupted |
| `lprod/checkpoint/` **without archive** | Cannot re-run workers or resume |
| `lhs_pool.json` backups in Git | Manual recovery |
| M4 production scripts/schemas | Pipeline contracts |

**Re-aggregation caveat:** If `modes_catalog.jsonl` is lost, restore `worker_results/` + `lprod/checkpoint/` from archive before re-running aggregation.

---

## ROM rebuild confirmation

After compaction of completed runs:

| Operation | Works? | Depends on |
|-----------|--------|------------|
| `build_m4_rom_from_completed_fom.py` | **Yes** | `modes_catalog.jsonl` per completed sample |
| `run_m4_rom_compare.py` | **Yes** | Catalog + `aggregation_result.json` + surrogate |
| STK v2.2b diagnostics | **Yes** | Catalog scalars (`mic_output_proxy`, etc.) |
| `v2_b3_m4_shared_export.py` | **Yes** | `aggregation/` plots + summaries |
| M4 resume on compacted sample | **No** | Requires restore from `classic/archives/*.tar.zst` |
| Legacy `ROMManager.solve_online` | **Unchanged** | Still needs `reduced_basis.npz` (separate legacy path) |

---

## Files added/changed

| File | Change |
|------|--------|
| `docs/M4_STORAGE_AND_LEGACY_CLEANUP_AUDIT.md` | **New** — this report |
| `scripts/compact_completed_m4_runs.py` | **New** — compaction workflow |
| `scripts/audit_m4_legacy_storage.py` | **New** — legacy audit JSON |
| `pipeline_runs/index/legacy_storage_audit.json` | Generated locally (dev; VM may differ) |
| `pipeline_runs/index/compaction_reports/` | Dry-run report output |

**Not changed:** FOM physics, solver, aggregation, ROM model, LHS statuses, production pipeline logic.
