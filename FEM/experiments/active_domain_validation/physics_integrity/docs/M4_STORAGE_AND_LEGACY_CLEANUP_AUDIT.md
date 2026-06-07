# M4 Storage and Legacy Cleanup Audit

**Date:** 2026-06-02 (updated policy)  
**Status:** Audit complete — **no deletions performed**. Compaction tooling ready for VM.  
**Principle:** Most completed FOM runs are retained as **ROM-ready compact runs only**. Only selected representative runs remain full. **Heavy artifacts are not archived by default.**

---

## Executive summary

| Finding | Detail |
|---------|--------|
| **Disk pressure** | VM root ~53G/59G used; `pipeline_runs/guitars` ≈ **15G** of ~25G `FEM/` |
| **Per completed sample** | ~380–675 MB total; **~85–95%** is mesh + checkpoint + scout mesh |
| **ROM minimum** | `aggregation/modes_catalog.jsonl` + `ROM/classic/lhs_pool.json` (+ trained surrogate for compare) |
| **Default policy** | **Direct delete** heavy artifacts after ROM-retention verification (`--delete-heavy-without-archive`) |
| **Full retention** | Only **2–3 representative** samples kept complete locally (`--keep-full-samples` + `--keep-full-latest`) |
| **Archiving** | **Optional legacy mode** (`--archive-heavy`) for explicit reference samples only — not default |
| **Do not touch** | `RUNNING` / `FAILED` / partial / resume-needed runs |
| **Estimated reclaim** | ~**12–16 GiB** local for ~30–35 completed samples (direct delete, 2–3 kept full) |
| **Shared budget** | Minimal archive use under new policy; reserve shared space for plots/exports and optional reference archives |

**Tools added:**

| File | Role |
|------|------|
| `scripts/compact_completed_m4_runs.py` | Completed-run compaction (`--delete-heavy-without-archive` default policy; dry-run default) |
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
| **SAFE_TO_DELETE_AFTER_COMPLETION** | Removable after ROM-retention verify on completed run (no archive required) |
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

### Default policy: direct delete (no archive)

```text
--delete-heavy-without-archive   # mutually exclusive with --archive-heavy
```

Most completed samples become **ROM-ready compact runs**: aggregation, rom, freeze, logs, sample metadata, and small manifests stay local; heavy solve artifacts are deleted after verification. A per-run `compaction/compaction_manifest.json` records deleted paths/bytes (`mode=delete_without_archive`).

### Eligibility (all must pass)

```text
LHS status == COMPLETED
last_aggregation_status == AGGREGATION_PASS
modes_catalog.jsonl exists and is readable
modes_summary.json or aggregation_result.json present
not RUNNING / FAILED / partial / resume-needed
not in --keep-full-samples or --keep-full-latest N
```

ROM/FOM comparison and shared-export plots: **reported** if missing; do not block delete.

### Local retention (always — never deleted)

```text
aggregation/          # modes_catalog.jsonl, modes_summary.json, plots, summaries
rom/
freeze/
logs/
sample/
pipeline_run_manifest.json
m4_sample_runtime_provenance.json
small scout/lprod JSON plans
compaction/compaction_manifest.json
ROM/classic/          # outside run tree; never touched
```

### Heavy delete targets (only these paths)

```text
lprod/checkpoint/
lprod/lprod_checkpoint_verify.json
lprod/mesh/
scout/checkpoint/
scout/discovery/
scout/mesh/
worker_results/
```

### Legacy optional archive mode

For explicit reference samples only:

```text
--archive-heavy --delete-heavy-after-verify --shared-root /media/sf_gmar
```

Archive layout (when used):

```text
/media/sf_gmar/classic/archives/sample_XXX__<run_id>__heavy.tar.zst
```

### Safety

- Default `--dry-run` (no deletes; reports planned actions)
- `--delete-heavy-without-archive` and `--archive-heavy` are **mutually exclusive**
- Destructive modes disable dry-run unless `--dry-run` is explicitly passed
- Refuses delete if ROM-required files missing (`rom_rebuild_safe=false`)
- Idempotent if `compaction_manifest.json` exists and heavy paths already gone
- No symlink traversal; deletes only under resolved run root

---

## Representative full-retention recommendation

**Operator must approve** before passing `--keep-full-samples`.

| Sample | Reason |
|--------|--------|
| `sample_000` | First LHS anchor |
| `sample_001` | M4 reference E2E / freeze gold run |
| `sample_034` (or latest completed) | Symbolic high-index / latest reference via `--keep-full-latest 1` |

**Suggested policy for initial cleanup:**

```text
--keep-full-latest 1
--keep-full-samples sample_000,sample_001
```

Keeps **2 named + 1 latest** (likely `sample_034` if newest in `0–34`) ≈ **2–3 full runs** locally. All other completed samples become ROM-ready compact runs only.

---

## Estimated space reclaim (VM)

Assumptions: ~30–35 completed samples, ~550 MB heavy each, **2–3 kept full** (~600 MB each). Direct delete — **no shared archive** by default.

| Scenario | Local freed | Shared archive |
|----------|-------------|----------------|
| 30 compacted, 3 kept full | **~13–15 GiB** | **0** (default policy) |
| 35 compacted (`0–34`), 3 kept full | **~15–17 GiB** | **0** |
| Per compacted sample (heavy only) | **~350–600 MiB** | — |

Retained per compacted sample: **~10–25 MiB** (`aggregation` + `rom` + `freeze` + metadata).

---

## Commands

### 1. Dry-run (required first)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/compact_completed_m4_runs.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --shape-name classic \
  --sample-range 0-34 \
  --keep-full-latest 1 \
  --keep-full-samples sample_000,sample_001 \
  --delete-heavy-without-archive \
  --dry-run
```

### 2. Direct delete (after review)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/compact_completed_m4_runs.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --shape-name classic \
  --sample-range 0-34 \
  --keep-full-latest 1 \
  --keep-full-samples sample_000,sample_001 \
  --delete-heavy-without-archive
```

### 3. Legacy worktree audit (optional)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_m4_legacy_storage.py
```

### 4. Optional archive mode (reference samples only)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/compact_completed_m4_runs.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --shape-name classic \
  --sample-range 0-34 \
  --shared-root /media/sf_gmar \
  --keep-full-samples sample_000,sample_001 \
  --archive-heavy \
  --delete-heavy-after-verify
```

### 5. Post-compaction ROM verification

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
| `worker_results/` after direct delete | Cannot re-aggregate if catalog corrupted (catalog is the ROM source of truth) |
| `lprod/checkpoint/` after direct delete | Cannot re-run workers or resume (acceptable for completed ROM-ready runs) |
| `lhs_pool.json` backups in Git | Manual recovery |
| M4 production scripts/schemas | Pipeline contracts |

**Re-aggregation caveat:** Under the default direct-delete policy, re-aggregation is only possible for **keep-full** samples. If `modes_catalog.jsonl` is lost on a compacted run, recovery requires a full FOM re-run.

---

## ROM rebuild confirmation

After compaction of completed runs:

| Operation | Works? | Depends on |
|-----------|--------|------------|
| `build_m4_rom_from_completed_fom.py` | **Yes** | `modes_catalog.jsonl` per completed sample |
| `run_m4_rom_compare.py` | **Yes** | Catalog + `aggregation_result.json` + surrogate |
| STK v2.2b diagnostics | **Yes** | Catalog scalars (`mic_output_proxy`, etc.) |
| `v2_b3_m4_shared_export.py` | **Yes** | `aggregation/` plots + summaries |
| M4 resume on compacted sample | **No** | Heavy artifacts deleted; only keep-full samples remain resumable |
| Legacy `ROMManager.solve_online` | **Unchanged** | Still needs `reduced_basis.npz` (separate legacy path) |

---

## Files added/changed

| File | Change |
|------|--------|
| `docs/M4_STORAGE_AND_LEGACY_CLEANUP_AUDIT.md` | **New** — this report |
| `scripts/compact_completed_m4_runs.py` | **Updated** — `--delete-heavy-without-archive` default policy |
| `scripts/audit_m4_legacy_storage.py` | **New** — legacy audit JSON |
| `pipeline_runs/index/legacy_storage_audit.json` | Generated locally (dev; VM may differ) |
| `pipeline_runs/index/compaction_reports/` | Dry-run report output |

**Not changed:** FOM physics, solver, aggregation, ROM model, LHS statuses, production pipeline logic.
