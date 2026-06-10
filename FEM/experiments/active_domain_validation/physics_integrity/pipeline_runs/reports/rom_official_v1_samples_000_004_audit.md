# ROM official v1 samples 000–004 — read-only audit

**Generated:** 2026-06-10  
**Mode:** read-only (no code, FEM, training, or run-artifact changes)  
**Machine JSON:** `pipeline_runs/reports/rom_official_v1_samples_000_004_audit.json`

---

## Executive summary

| Decision | Value |
|----------|-------|
| **Part A — official ROM data** | `OFFICIAL_ROM_DATA_BLOCKED` |
| **Part E — restore ROM shadow pipeline** | `BLOCKED_BEFORE_ROM_RESTORE` |

All five runs are **numerically complete** on the ROM production mesh (`COMPLETED`, `AGGREGATION_PASS`, 3 workers) per VM shared summaries. None pass the **full production finalization gate** required for ROM training eligibility on this audit machine.

---

## Part A — Official ROM-mesh samples 0–4

### Expected profile (all samples)

| Field | Expected | Observed (shared summary) |
|-------|----------|---------------------------|
| `mesh_profile` | `rom` | `rom` ✓ |
| `mesh_level_id` | `L_rom_prod` | `L_rom_prod` ✓ |
| `dataset_version` | `m4_geometry_corrected_rommesh_v1` | `m4_geometry_corrected_rommesh_v1` ✓ |
| `workers` | `3` | `workers_actual_parallel=3` ✓ |

### Status table

| Sample | Run ID | Terminal | Agg | Modes (deduped) | Chunks | Worker s | Cleanup | Compaction | Prod accept | Part A pass |
|--------|--------|----------|-----|-----------------|--------|----------|---------|------------|-------------|-------------|
| 000 | `sample_000_rom_official_v1` | COMPLETED | AGGREGATION_PASS | 676 | 12/12 | 800.23 | **failed** | **planned** | unverified | **BLOCKED** |
| 001 | `sample_001_rom_official_v1` | COMPLETED | AGGREGATION_PASS | 671 | 11/11 | 408.56 | null | null | unverified | **BLOCKED** |
| 002 | `sample_002_rom_official_v1` | COMPLETED | AGGREGATION_PASS | 612 | 13/13 | 1076.56 | null | null | unverified | **BLOCKED** |
| 003 | `sample_003_rom_official_v1` | COMPLETED | AGGREGATION_PASS | 692 | 14/14 | 345.56 | null | null | unverified | **BLOCKED** |
| 004 | `sample_004_rom_official_v1` | COMPLETED | AGGREGATION_PASS | 668 | 12/12 | 818.63 | null | null | unverified | **BLOCKED** |

**Evidence:** `C:\Users\eivry\OneDrive\Desktop\gmar\classic\summaries\sample_*__*_rom_official_v1__summary.json`  
**Gap:** Run trees are on the VM only; `pipeline_runs/guitars/` is absent on the dev workspace.

### Checklist by audit category

#### 1. Numerical completion — partial pass

- `terminal_status=COMPLETED` and `aggregation_status=AGGREGATION_PASS` for all five (shared summaries).
- `final_aggregation_ready`, `failed_chunks=0` — **not verified** without run-tree `aggregation/aggregation_result.json`.

#### 2. Production acceptance — blocked

- `production_acceptance_pass` / `production_acceptance_failures` are **not exported** in shared summaries and were **not read** from run trees on this machine.

#### 3. Cleanup and compaction — blocked

| Sample | Issue |
|--------|-------|
| 000 | `cleanup_status=failed`, `compaction_status=planned` (known VM finalization failure before repo fix) |
| 001–004 | `cleanup_status` / `compaction_status` null → finalization barrier not confirmed |

#### 4. Durable outputs — unverified locally

Required paths under each `run_root` were not present locally. VM graph manifests confirm aggregation PNG sources existed at solve time.

#### 5. Shared output — partial pass

| Asset | Status |
|-------|--------|
| Compact summary JSON | Present for all 5 under `classic/summaries/` |
| Graph export manifest | Present; `export_status=EXPORTED` for all 5 |
| Four approved PNGs | Copied on VM per manifest; **0 files** in Windows `classic/plots/` at audit time (OneDrive sync gap) |

#### 6. Mesh identity — unverified

- Profile fields confirm `L_rom_prod` intent.
- Operator/generated mesh hashes — require `freeze/physics_identity_manifest.json` on VM.

#### 7. Sample uniqueness — pass (available evidence)

- All LHS parameter vectors differ across samples 000–004.
- Deduped mode counts: 676, 671, 612, 692, 668 (all distinct).
- **No duplicate plot SHA256** across samples (20 PNG hashes, all unique).
- `modes_catalog_deduped.jsonl` file hashes — **not computed** (run trees unavailable).

#### 8. Data consistency — partial

Unique woods per sample: spruce/rosewood, mahogany/mahogany, rosewood/spruce, rosewood/cedar, mahogany/spruce. Scalar field distributions not fully audited without catalog files.

#### 9. Runtime — partial

| Field | All samples | Notes |
|-------|-------------|-------|
| `worker_runtime_s` | populated | from provenance `stage5_workers` |
| `total_runtime_s` | **null** | key mismatch (see below) |
| `scout_runtime_s` | **null** | stage key mismatch |
| `checkpoint_runtime_s` | **null** | stage key mismatch |
| `freeze_runtime_s` | **null** | stage key mismatch |
| `peak_rss_bytes_per_worker` | populated | per-chunk worker peaks |

**Root cause for null totals:** `v2_b3_m4_runtime_provenance.py` writes `total_pipeline_wall_seconds` and stages `stage4_lprod_checkpoint` / `stage6_aggregate`, while `v2_b3_m4_shared_export.build_compact_summary_payload` reads `total_runtime_s` and `stage1_scout_mesh` / `stage6_freeze`.

**Smallest future-only fix:** Add alias mapping in `build_compact_summary_payload` or provenance merge — no retroactive run edits.

### Part A decision

```
OFFICIAL_ROM_DATA_BLOCKED
```

**Blocking samples:** all five (`sample_000`–`sample_004`)

**Blocking fields (common):** `production_acceptance_pass`, `cleanup_status=completed`, `cleanup_verification.pass`, `compaction_status=completed`, `forbidden_heavy_artifact_count=0`, durable artifact verification

**Additional sample_000 fields:** `cleanup_status=failed`, `compaction_status=planned`

---

## Part B — Previous ROM-in-pipeline implementation

### Architecture map

| File | Function / class | Input artifacts | Output artifacts | Status |
|------|------------------|-----------------|------------------|--------|
| `v2_b3_m4_rom_fom_compare_lib.py` | `maybe_run_rom_prepredict`, `maybe_run_rom_compare`, `run_rom_online_prediction` | LHS params; `ROM/<shape>/m4_modal_surrogate.*`; FOM `modes_catalog.jsonl` | `rom/rom_prediction_pre_fom.json`; `rom/rom_fom_comparison.json` | **dormant** (CLI opt-in) |
| `v2_b3_m4_modal_surrogate_lib.py` | `collect_completed_fom_training_rows`, `build_surrogate_from_training_rows` | `lhs_pool.json`; per-run `aggregation/modes_catalog.jsonl` | `m4_modal_surrogate.{json,npz}` | **active** (standalone) |
| `build_m4_rom_from_completed_fom.py` | `main` | same as above | surrogate + manifest | **active** |
| `run_m4_rom_compare.py` | `main` | completed runs + pool | comparison + LHS patch | **active** |
| `v2_b3_m4_lhs_production_batch.py` | `run_lhs_production_batch` | batch spec + pool | prepredict before FOM; compare after cleanup | **dormant** hooks |
| `run_m4_production_pipeline.py` | CLI | passes `--run-rom-*` flags | — | **active**; ROM off by default |
| `v2_b3_m4_rom_scalar_fields.py` | field defs, metrics | catalog rows | comparison metrics | **active** |
| `FEM/rom/rom_manager.py` | `ROMManager` | legacy snapshots / `reduced_basis.npz` | online freqs | **legacy** fallback |
| `FEM/scripts/rom_pipeline.py` | CLI | snapshots | basis | **legacy** |

### Answers to design questions

1. **Model type:** M4 modal surrogate v2.1 (k-NN IDW); legacy POD `ROMManager` optional fallback.
2. **Targets:** mode `frequency_hz` list + Phase-2 scalars (shares, bridge/radiation/mic proxies, coupling class, intensity derivatives).
3. **Features:** 6 geometry floats + ordinal top/back wood IDs (8-D vector).
4. **Training files:** `aggregation/modes_catalog.jsonl` per LHS `last_run_id` — **no mesh/dataset filter today**.
5. **Data provenance:** mixed — pool-driven `last_run_id` could point at `m4prod2`, `rom_prod_004`, etc.; not safe for official ROM mesh restore without new filters.
6. **Model versions:** `ROM/classic/rom_model_manifest.json`; schema `m4_modal_surrogate_v2_1`.
7. **Train/validation:** production uses train-included model; holdout via `--exclude-target-from-training` / `--leave-one-out` in `run_m4_rom_compare.py`.
8. **Prediction before FOM:** yes — `maybe_run_rom_prepredict` runs before `run_pipeline` in batch loop.
9. **Comparison:** greedy frequency match in Hz band; scalar MAE + `meets_accuracy_target` tracking in `rom_fom_comparison_v4`.
10. **Model update between samples:** **not automatic in batch** — separate `build_m4_rom_from_completed_fom.py` run; LHS patched with comparison metrics only.
11. **Reusable unchanged:** surrogate math, comparison schema, shadow nonblocking, compaction keep-list for `rom/*.json`.
12. **Must rewrite:** training row collector filters; dataset_version gate (`M4_REQUIRED_ROM_DATASET_VERSION` still defaults `m4_geometry_corrected_v1`); official-run registry; batch retrain policy.

### Current batch ROM order (existing code)

```
prepredict → FOM pipeline → shared export → cleanup/compaction → rom compare (if pass)
```

Compare correctly uses frozen `rom_prediction_pre_fom.json` when not in holdout mode.

---

## Part C — Current-data-only ROM policy

### Allowed training runs (only)

```
sample_000_rom_official_v1 … sample_004_rom_official_v1
```

### Required filters (new policy — not yet enforced in code)

```
mesh_profile=rom, mesh_level_id=L_rom_prod,
dataset_version=m4_geometry_corrected_rommesh_v1,
terminal_status=COMPLETED, aggregation_status=AGGREGATION_PASS,
production_acceptance_pass=true, cleanup_status=completed
```

### Sample-count adequacy

| Use case | Verdict |
|----------|---------|
| Pipeline integration testing | **Enough** (once finalized) |
| Basic model fitting | **Marginal** (k=5 with 5 points = interpolation) |
| Meaningful validation | **Not enough** (need holdout beyond training set) |
| Production prediction quality | **Not enough** for reliable accuracy |

---

## Part D — Target restored shadow pipeline

### Prediction timing: **Option A**

Predict **before Scout** using LHS/sample parameters only. The M4 surrogate does **not** require Scout-derived features.

### Target sequence

```
sample input
→ load latest eligible ROM model (official-filtered)
→ predict + freeze rom/rom_prediction_pre_fom.json (timestamp < FOM start)
→ Scout → target plan → checkpoint → workers → aggregation → freeze
→ compare frozen prediction vs modes_catalog_deduped.jsonl
→ write rom/rom_vs_fom_comparison.json (+ compact rom_prediction_summary.json)
→ if accepted: register FOM sample for ROM dataset
→ optional retrain (policy-driven, not every sample with N=5)
→ shared graph export → compaction → cleanup → reconcile → next sample
```

### Shadow rules

- Prediction never alters FOM execution, target plan, or mesh.
- Comparison must read the **frozen** pre-FOM artifact only.
- Record `model_version` and `training_sample_ids` in prediction doc.

---

## Part E — Deliverables summary

| # | Item | Result |
|---|------|--------|
| 1 | Part A audit | `OFFICIAL_ROM_DATA_BLOCKED` |
| 2 | Status table | See Part A table |
| 3 | Duplication | No cross-sample PNG hash duplicates; mode counts distinct |
| 4 | Missing runtime fields | `total_runtime_s`, scout/checkpoint/freeze null — provenance key mismatch |
| 5 | Old ROM architecture map | Part B table |
| 6 | Model type / features / targets | k-NN IDW v2.1; 8-D LHS; freqs + Phase-2 scalars |
| 7 | Old training provenance | Pool `last_run_id` + `modes_catalog.jsonl` (unfiltered) |
| 8 | Reusable | Surrogate core, compare lib, shadow semantics, compaction keep-list |
| 9 | Do not reuse | Unfiltered collector, old dataset_version gate, legacy POD, stale LHS |
| 10 | Current-data-only policy | Part C |
| 11 | Restored pipeline sequence | Part D |
| 12 | Minimum code changes | See below |
| 13 | Risks | See below |
| 14 | Recommendation | `BLOCKED_BEFORE_ROM_RESTORE` |
| 15 | No modifications | Confirmed |

### Minimum code changes (future implementation)

1. Add `OfficialRomDatasetFilter` to `collect_completed_fom_training_rows` (mesh, dataset, acceptance, cleanup).
2. Set `M4_REQUIRED_ROM_DATASET_VERSION=m4_geometry_corrected_rommesh_v1` or hardcode official version in prepredict gate.
3. Wire `--run-rom-prepredict` / `--run-rom-compare` into official batch launch script (shadow, nonblocking).
4. Move ROM compare to **after freeze, before graph export** (compare needs catalog; export should not block compare).
5. Add optional post-batch retrain hook calling `build_m4_rom_from_completed_fom.py` with official filters.
6. Fix shared-summary runtime key aliases (future runs only).
7. Reconcile `lhs_pool.json` after finalization on VM.

### Risks before implementation

- Training on unfinalized runs would leak heavy artifacts and unverified physics identity.
- Five samples enable plumbing tests only; metrics will overfit.
- Stale `lhs_pool.json` on dev machine can mis-route training if filters are incomplete.
- `maybe_run_rom_prepredict` skips when pool `dataset_version` mismatches env default (`m4_geometry_corrected_v1`).
- Compare-after-cleanup ordering risks missing `rom/` if barrier misconfigured without `run_rom_compare=True`.

### VM next steps (read-only diagnose, then finalize)

```bash
# Per sample 000–004:
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_finalize_completed_run.py \
  --sample-id sample_000 --run-id sample_000_rom_official_v1 \
  --batch-id lhs_rom_official_v1_20260610 --workers 3 --diagnose

# After diagnose OK:
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_finalize_completed_run.py \
  --sample-id sample_000 --run-id sample_000_rom_official_v1 \
  --batch-id lhs_rom_official_v1_20260610 --workers 3
```

---

## Constraints confirmation

- No repository code modified for this audit.
- No FEM solve launched.
- No model trained.
- No completed run artifacts modified.
- Audit reports written only to `pipeline_runs/reports/`.
