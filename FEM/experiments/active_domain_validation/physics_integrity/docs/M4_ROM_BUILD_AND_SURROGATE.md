# M4 ROM build, audit, and Phase-1 surrogate

**Date:** 2026-06-05  
**Scope:** ROM training from completed M4 FOM outputs. **No FEM physics changes.** **No mode shapes.**

---

## Executive summary

| Item | Status |
|------|--------|
| `ROM/classic/reduced_basis.npz` on VM | **Missing** (expected — not trained yet) |
| Legacy ROM training path | Requires **full eigenvector snapshots** from legacy FOM |
| M4 `modes_catalog.jsonl` | Scalar modal metadata only — **not sufficient for POD basis** |
| **Phase-1 solution** | New **`m4_modal_surrogate.{json,npz}`** trained from M4 frequencies |
| Compare script | Uses surrogate automatically when available (`resolve_active_rom_backend`) |

---

## ROM audit (10 questions)

### 1. What does the current ROM model require as training input?

**Legacy POD ROM (`ROMManager`):**

| Stage | Input |
|-------|--------|
| Offline / `collect_snapshots` | Legacy coupled FOM per LHS sample → `ROM/<shape>/snapshots/snapshot_*.npz` |
| `build_basis` | All snapshot NPZ files under `ROM/<shape>/snapshots/` |

Each legacy snapshot contains:

- `freqs_hz`
- **`eigvecs_real`** (dense eigenvectors) **or** CSR `ev_*` arrays
- Optional telemetry: `participation_ratios`, `uniqueness_scores`

**M4 Phase-1 surrogate (new):**

- `aggregation/modes_catalog.jsonl` per completed sample (**scalar fields only**)
- Matching LHS `parameters` from `ROM/classic/lhs_pool.json`

---

### 2. Full snapshots vs modal scalars?

| Approach | Needs mode shapes? | Compatible with M4 catalog? |
|----------|-------------------|----------------------------|
| Legacy POD `build_basis` | **Yes** — stacks eigenvector columns | **No** — M4 catalog has no eigenvectors |
| M4 modal surrogate | **No** — uses `frequency_hz` lists only | **Yes** |

M4 aggregation intentionally avoids Stage C / full mode-shape storage. The old POD ROM **cannot** be rebuilt from `modes_catalog.jsonl` alone.

---

### 3. What should `reduced_basis.npz` contain?

Legacy file written by `ROMManager.build_basis()`:

```text
basis              (num_dof × rank) POD modes
singular_values    SVD singular values
energy_curve       cumulative energy fraction
selected_rank      chosen rank
snapshots_count    number of snapshot files used
source_mode_columns total mode columns stacked
```

This is a **FEM DOF-space projection basis**, not a frequency lookup table.

---

### 4. Which script creates `reduced_basis.npz`?

```bash
python FEM/scripts/rom_pipeline.py build-basis --shape classic
```

Or programmatically: `ROMManager.build_basis(shape_name)`.

**Prerequisite:** `ROM/classic/snapshots/snapshot_*.npz` from legacy offline FOM collection:

```bash
python FEM/scripts/rom_pipeline.py offline --shape classic
```

---

### 5. Is the old ROM compatible with new M4 production outputs?

**No — not without an adapter.**

| Gap | Detail |
|-----|--------|
| Training data | Legacy uses snapshot eigenvectors; M4 provides scalar catalog |
| FOM engine | Legacy `run_fom_for_rom` vs M4 B3 shift-invert + adaptive chunks |
| Mesh/operators | May differ between legacy ROM path and M4 L_prod |
| Pool status | Same `lhs_pool.json`, but legacy ROM ignores `last_run_dir` |

**Minimal adapter implemented:** `m4_modal_surrogate` — k-NN over LHS parameters with FOM frequency labels.

Full POD compatibility would require exporting M4 eigenvectors (out of scope — no Stage C).

---

### 6. Minimal adapter needed

**Implemented:** `v2_b3_m4_modal_surrogate_lib.py` + `build_m4_rom_from_completed_fom.py`

- **Train:** completed `modes_catalog.jsonl` → store parameter vectors + frequency matrices
- **Predict:** inverse-distance-weighted k-NN in normalized 8D feature space (6 geometry + 2 wood indices)
- **Output:** sorted `frequency_hz` list (STK fields marked unavailable in compare path)

---

### 7. What does `ROMManager.solve_online()` return?

```python
{
    "freqs_hz": [...],           # frequencies only
    "elapsed_s": float,
    "nev": int,
    "nev_requested": int,
    "num_basis_modes": int,
    "num_dof": int,
    "basis_path": str,
}
```

**Frequencies only.** No `top_share`, `coupling_class`, `radiation_proxy`, etc.

---

### 8. Frequencies only or modal/STK fields?

**Only frequencies.** STK/audio metadata is not part of legacy ROM online solve.

M4 surrogate Phase-1 is also **frequency-only**. Scalar STK fields remain FOM-only until a future surrogate stage.

---

### 9. How to train/update ROM using `sample_000`–`sample_015`?

**Recommended (M4 Phase-1):**

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_m4_rom_from_completed_fom.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --shape-name classic \
  --completed-only \
  --max-samples 16
```

This:

1. Selects LHS rows with `status=COMPLETED` and `aggregation/modes_catalog.jsonl`
2. Reads deduped `frequency_hz` per sample
3. Writes:
   - `ROM/classic/m4_modal_surrogate.json` (metadata + training manifest)
   - `ROM/classic/m4_modal_surrogate.npz` (compact numeric arrays, ~KB–low MB)
   - `ROM/classic/rom_model_manifest.json` (active backend = `m4_surrogate`)

Re-run the build script as more FOM samples complete to refresh the surrogate.

**Not recommended yet:** legacy `build-basis` until M4 eigenvector export exists.

---

### 10. Correct command to build the ROM basis/model

| Goal | Command |
|------|---------|
| **M4 frequency surrogate (Phase-1)** | `build_m4_rom_from_completed_fom.py` (above) |
| Legacy POD basis | `python FEM/scripts/rom_pipeline.py build-basis --shape classic` (needs snapshots) |
| Compare after build | `run_m4_rom_compare.py --force-sample sample_005` |

---

## Phase-1 surrogate specification

### Method

`knn_idw_sorted_modes`:

1. Encode LHS parameters → 8D feature vector
2. Normalize using training mean/std
3. Find `k` nearest training samples (default `k=5`)
4. Weight = `1 / distance²`
5. For each mode index `m` (sorted by frequency), weighted average across neighbors

### Inputs

```text
geometry.length, geometry.width, geometry.depth
geometry.top_thickness, geometry.hole_radius, geometry.back_thickness
top_wood_id, back_wood_id
```

### Outputs (prediction)

```json
{
  "frequencies_hz": [60.1, 66.2, "..."],
  "method": "knn_idw_sorted_modes",
  "runtime_s": 0.001,
  "k_neighbors_used": 5,
  "neighbor_sample_ids": ["sample_003", "sample_007"]
}
```

### Backend resolution (`resolve_active_rom_backend`)

| Priority (`active_backend=auto`) | Condition |
|----------------------------------|-----------|
| 1 | `m4_modal_surrogate.json` + `.npz` exist |
| 2 | `reduced_basis.npz` exists |
| 3 | Fail with build instructions |

---

## ROM/FOM accuracy definition (Phase 1)

All `rom_fom_comparison.json` artifacts use schema **`rom_fom_comparison_v2`**.

| Item | Value |
|------|--------|
| Frequency band | **60–550 Hz** (matching + metrics) |
| Primary metric | **`median_relative_error`** |
| Initial target | **`median_relative_error <= 0.05`** (5%) |
| Matching | Greedy nearest-neighbor in Hz, `max_match_distance_hz` default 15 |

**Required metrics in comparison JSON:**

```text
median_relative_error      # primary success criterion
mean_relative_error
p90_relative_error
median_abs_error_hz
mean_abs_error_hz
max_abs_error_hz           # diagnostic only (not primary)
matched_mode_count
unmatched_rom_count
unmatched_fom_count
training_sample_count_at_prediction
```

**Optional audio-weighted metrics** (when `fom_radiation_proxy` present):

```text
audio_weighted_mean_relative_error
audio_weighted_median_relative_error
top_radiation_modes_mean_relative_error
top_radiation_modes_median_relative_error
```

**Rolling accuracy history** (updated after each compare):

```text
ROM/classic/comparisons/rom_accuracy_history.csv
ROM/classic/comparisons/rom_accuracy_summary.json
```

### No-leakage validation (required before production ROM)

Train-included comparisons can show **zero error** when the target sample was in the surrogate training set. Use holdout mode:

```bash
python FEM/.../run_m4_rom_compare.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --force-sample sample_005 \
  --exclude-target-from-training
```

This builds a **temporary in-memory surrogate** (train: all completed except `sample_005`) and does **not** overwrite `ROM/classic/m4_modal_surrogate.*`.

Comparison JSON includes:

```text
validation_mode: holdout | leave_one_out | train_included
training_includes_target: true | false
accuracy_meaningful: true | false
excluded_sample_ids: [...]
```

Only trust `accuracy_spec.meets_target_meaningful` when `accuracy_meaningful=true`.

Use `rom_accuracy_summary.json` → `aggregate_meaningful_only` and `by_training_sample_count` to answer:

- After 16 FOM training samples, what is ROM accuracy?
- After 40 samples, did median relative error improve?
- Are we below 5% median relative error?

---

## VM workflow

```bash
# 1. Build surrogate from completed FOM samples (000–015)
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_m4_rom_from_completed_fom.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --shape-name classic \
  --completed-only \
  --max-samples 16

# 2. Compare ROM vs FOM for one sample
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_rom_compare.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --force-sample sample_005

# 3. Optional: production with ROM hooks (after surrogate exists)
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --max-samples 10 --workers 3 --execute --continue-on-fail \
  --run-rom-prepredict --run-rom-compare --rom-nonblocking
```

Expected compare output:

```text
pipeline_runs/guitars/sample_005/runs/<run_id>/rom/rom_fom_comparison.json
ROM/classic/comparisons/sample_005__<run_id>_rom_fom_comparison.json
```

---

## What remains unchanged

- M4 FOM solver, aggregation, participation/audio coupling
- No mandatory ROM in FOM runs
- No Stage C / full mode shapes
- Legacy `ROMManager` / `rom_pipeline.py` untouched

---

## Future phases

| Phase | Work |
|-------|------|
| 2 | Surrogate validation dashboard across all completed samples |
| 3 | Ridge/GPR surrogate or operator-aligned POD if eigenvectors exported |
| 4 | Predict `radiation_proxy` / shares from catalog scalars (optional) |
