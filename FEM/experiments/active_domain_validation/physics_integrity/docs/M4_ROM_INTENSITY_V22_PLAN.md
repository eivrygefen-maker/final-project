# M4 ROM Intensity v2.2 — Technical Plan

**Date:** 2026-06-02  
**Shape:** `classic` only  
**Production ROM (frozen):** `m4_modal_surrogate_v2_1_intensity` / `knn_idw_modal_surrogate_v2_1`  
**Scope:** ROM-only intensity improvement planning. **No FOM physics, solver, aggregation, or production pipeline changes.**

---

## Executive summary

| Area | Current v2.1 (30-sample LOO) | Target | Assessment |
|------|------------------------------|--------|------------|
| Frequency median rel. error | **~1.42%** | ≤ 5% | **Already strong** |
| `mic_output_proxy_p95_norm_mae` | **~0.233** | 0.10–0.15 | **Not met** (~55–130% above target) |
| `radiation_proxy_p95_norm_mae` | not in user summary (likely ~0.20–0.28) | 0.10–0.15 | **Not met** (estimated) |
| `radiation_proxy_rank_correlation` | **~0.164** | ≥ 0.6 | **Far below** |
| `top_k_radiation_overlap` (20%) | **~0.235** | ≥ 0.6 | **Far below** |
| Classification | **~88–91%** | stable | **Strong** |

**Primary conclusion:** Intensity weakness is **not** a frequency-training problem. It is a **combination of (B) weak intensity model**, **(C) modal correspondence/alignment**, and **(D/E) proxy noise + coarse FOM mic/radiation proxies**. Sample count **(A)** helps frequency and shares more than rank/top-k; **30 samples is insufficient alone** for the requested rank/top-k targets.

**Recommended minimal v2.2 experiment:** **class-aware + frequency-band-aware nearest-mode intensity matching** (ROM-only, frequency path unchanged, experimental model artifact). See [Task E](#task-e--minimal-implementation-proposal).

**Run diagnostics on VM** (no production ROM overwrite):

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_rom_intensity_v22_diagnostics.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --samples 0-29 \
  --json-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V22_DIAGNOSTICS.json \
  --csv-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V22_HOLDOUT_PER_SAMPLE.csv
```

---

## Validated baseline (user-reported, 30 LOO holdouts)

```text
meaningful_count = 30
leakage_count    = 0
training_pool    = 29 (leave-one-out)
```

| Metric | Median | Mean | Worst |
|--------|--------|------|-------|
| Frequency median relative error | 1.42% | 1.49% | 2.23% |
| `top_share_mae` | 0.0415 | — | — |
| `coupling_class_accuracy` | — | 88.4% | — |
| `dominant_region_accuracy` | — | 91.5% | — |
| `mic_output_proxy_p95_norm_mae` | 0.233 | 0.238 | 0.338 |
| `radiation_proxy_log_mae` | 0.363 | — | — |
| `radiation_proxy_rank_correlation` | 0.164 | — | — |
| `top_k_radiation_overlap` (20%) | 0.235 | — | — |
| ROM runtime | 13.7 s | — | — |

These numbers are the reference row **v2.1 aggregate** in all comparison tables below.

---

## Task A — Audit of the current v2.1 predictor

### A.1 Architecture overview

v2.1 uses a **two-stage hybrid** inside `predict_modal_catalog()` (`v2_b3_m4_modal_surrogate_lib.py`):

| Stage | Output | Method | Alignment key |
|-------|--------|--------|---------------|
| **1 — Frequencies** | `frequencies_hz[m]` | Sorted-index IDW over k LHS neighbors | **Mode index `m`** (not Hz) |
| **2 — Scalars** | Phase-2 fields + intensity derivatives | Nearest-frequency IDW per neighbor | **Predicted Hz** per mode |

Intensity fields predicted per mode:

```text
radiation_proxy
mic_output_proxy
bridge_excitation_abs
```

Plus derived outputs (v2.1):

```text
*_log10
*_p95_norm   (per-guitar normalization inside 60–550 Hz band, p95 scale)
```

Raw proxy values are still emitted for downstream compatibility.

### A.2 LHS / geometric / material features used today

Encoding: `encode_lhs_parameters()` — **9 features**, no derived acoustics.

| # | Feature | Type |
|---|---------|------|
| 1–6 | `geometry.length`, `geometry.width`, `geometry.depth`, `geometry.top_thickness`, `geometry.hole_radius`, `geometry.back_thickness` | continuous |
| 7–8 | `top_wood_id`, `back_wood_id` | categorical → index in `{spruce, cedar, mahogany, rosewood, maple}` |

**Not used for neighbor selection or intensity:** cavity volume, hole area, aspect ratios, wood density/Q, bracing, coupling_class, frequency band, mode index beyond sorted order.

### A.3 Parameter distance normalization

1. Stack training rows into feature matrix `X` (n_samples × 9).
2. Per-dimension standardization stored in NPZ:
   - `feature_mean`, `feature_std` (std floored at `1e-12`)
   - `feature_matrix_norm = (X - mean) / std`
3. Query guitar: `x_norm = (x - mean) / std`
4. Distance: **Euclidean** `||x_norm - x_train_i||₂`

Wood indices are treated as numeric dimensions (ordinal), not one-hot. This is acceptable for 5×5 wood grid but mixes heterogeneous units.

### A.4 k-nearest sample selection

- Default `k_neighbors = 5` (capped by training count).
- `nn_idx = argsort(dists)[:k]`
- IDW weights: `w_i = 1 / (d_i + 1e-8)²`, normalized to sum 1.
- Same neighbors used for **both** frequency and scalar stages.

### A.5 Frequency alignment (why frequency works)

For each output mode index `m ∈ [0, min_count-1]`:

```text
pred_freq[m] = Σ_j w_j · neighbor_freq_j[m]
```

Neighbors' catalogs are sorted by Hz. Alignment is by **sorted modal index**, not by matching FOM mode identity. This works because:

- Classical guitar LHS samples share similar mode **density** per band after dedupe.
- Frequencies vary smoothly with geometry/material in 9D LHS space.
- k-NN in standardized geometry space captures this smooth map well.

`min_count = min(neighbor mode counts)`; output mode count truncated to shortest neighbor catalog.

### A.6 Scalar selection from each neighbor

For each predicted frequency `f_hz` (stage 2), per neighbor catalog `j`:

1. `idx_j = argmin_i |catalog_j[i].frequency_hz - f_hz|` (**nearest-frequency**, not index `m`).
2. Read scalar fields from `catalog_j[idx_j]`.

**Important asymmetry:** frequency at index `m` is blended by index; scalars at the same `m` are taken from **different modal indices** in each neighbor that happen to be Hz-close. This is the core correspondence risk.

### A.7 IDW application for scalars

For each numeric Phase-2 field (including the three intensity proxies):

```text
pred_field = Σ_j w_j · neighbor_field_j / Σ_j w_j
```

Categorical fields (`coupling_class`, `dominant_region`, `secondary_region`): **weighted plurality vote**.

Intensity derivatives (`*_log10`, `*_p95_norm`): separate IDW blend of precomputed training values via `append_intensity_derivatives_to_prediction()`. Fallback for `log10`: compute from blended raw proxy.

### A.8 Does class / domain / band affect scalar selection?

| Signal | Affects neighbor guitars? | Affects per-neighbor mode pick? |
|--------|---------------------------|-------------------------------|
| `coupling_class` | **No** | **No** |
| `dominant_region` | **No** | **No** |
| Frequency band | **No** | **No** (only continuous Hz distance) |
| Sorted mode index | **Yes** (frequency only) | **No** (scalars use Hz nearest) |

Classification is accurate (~88%) because voting aggregates coarse bins. **Intensity fails when the Hz-nearest neighbor mode is the wrong physical family** (e.g. air mode vs top/back mixed at similar frequency).

### A.9 Raw vs log vs normalized — internal vs output

| Representation | Training | Internal blend | Output in `rom_prediction_pre_fom.json` |
|----------------|----------|----------------|----------------------------------------|
| Raw proxies | Stored in NPZ | **Yes** — primary IDW target | **Yes** |
| `log10(proxy+ε)` | Precomputed at train | **Yes** — IDW on derived | **Yes** |
| `p95_norm` | Precomputed per training guitar | **Yes** — IDW on derived | **Yes** |

Compare metrics prefer `p95_norm` MAE and rank/top-k on raw proxy ordering within guitar. **v2.1 does not re-normalize predictions per holdout guitar** at inference; it blends neighbors' pre-normalized values. That mismatches FOM evaluation, which recomputes p95_norm on the **holdout** guitar's own p95 scale.

This is a **systematic train/infer normalization mismatch** and partially explains elevated p95_norm MAE even when raw proxy ranking is mediocre.

### A.10 Different mode counts between samples

- Training stores `mode_counts[i]` and pads frequency/scalar arrays to `max_modes`.
- Prediction outputs `min(neighbor counts)` modes (or `nev` cap).
- Sorted-index frequency alignment assumes mode `m` in different guitars is "the same ordinal mode."
- When one guitar has extra cavity or missing branch modes, **index `m` diverges in physical meaning** → scalar Hz-match may attach to a different branch.

Deduped catalogs (~per sample) reduce duplicate Hz rows but do not fix branch alignment.

### A.11 Why frequency is strong but rank/top-k is weak

| Factor | Frequency | Intensity rank / top-k |
|--------|-----------|------------------------|
| Target smoothness in LHS space | Very smooth | Less smooth; proxy noise |
| Alignment | Sorted index (consistent) | Hz-nearest (family-ambiguous) |
| Class information | Unused (not needed) | Unused (needed) |
| Per-guitar normalization | N/A | Blended across guitars (scale leak) |
| FOM proxy resolution | N/A | Coarse mic/radiation (see air/mic audit) |
| Metric sensitivity | Relative Hz error | Rank of ~50–80 modes per guitar |

**Net:** v2.1 is a **geometry interpolator for Hz** and a **geometry + Hz proxy blender for amplitudes**, not a **physics-consistent mode correspondence model**.

---

## Task B — Limitation diagnosis (algorithm vs data vs proxy)

Use read-only script: `audit_rom_intensity_v22_diagnostics.py`.  
Fill tables below from `M4_ROM_INTENSITY_V22_DIAGNOSTICS.json` after VM run.

### B.1 Learning curve (pool sizes 8 / 12 / 16 / 20 / 24 / 29)

**Method:** For each holdout sample, train on the `K` **nearest LHS neighbors** among the other 29 completed samples (no leakage). Rebuild surrogate; evaluate p95_norm MAE, rank, top-k.

| Pool size | Freq med err | Mic p95 MAE | Rad p95 MAE | Bridge p95 MAE | Rad rank | Mic rank | Top-k rad |
|-----------|--------------|-------------|-------------|----------------|----------|----------|-----------|
| 8 | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* |
| 12 | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* |
| 16 | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* |
| 20 | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* |
| 24 | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* | *VM* |
| 29 | **~1.42%** | **~0.233** | *VM* | *VM* | **~0.164** | *VM* | **~0.235** |
| v2.1 prod (29) | 1.42% | 0.233 | — | — | 0.164 | — | 0.235 |

**Interpretation guide:**

| Pattern | Diagnosis |
|---------|-----------|
| Intensity metrics improve steadily 8→29, gap to target small at 29 | **(A) data-limited** — more FOM may suffice |
| Intensity flat from ~16–20 upward; rank/top-k stagnant | **(B)+(C)+(D/E)** — algorithm/proxy/alignment limited |
| Frequency improves with K but rank does not | Confirms **decoupled** problems |

**Pre-VM estimate (from v2.1 level):** rank/top-k at 0.16–0.24 are so far below 0.6 that **sample count alone is unlikely** to close the gap without alignment changes. Expect partial MAE gains with more data, not rank doubling.

### B.2 Nearest-neighbor distance vs error

Per holdout row in `M4_ROM_INTENSITY_V22_HOLDOUT_PER_SAMPLE.csv`:

| Field | Purpose |
|-------|---------|
| `nearest_training_distance` | Min standardized LHS distance |
| `mean_k_neighbor_distance` | Mean over k=5 neighbors |
| `mic_output_proxy_p95_norm_mae` | Holdout mic error |
| `radiation_proxy_rank_correlation` | Holdout rank |

**Analysis:** Correlate distance with errors (script output; add scatter in notebook). If high errors cluster at high distance → **(A)+(B)** interpolation gap. If errors high even at low distance → **(C)+(D)+(E)**.

**Expectation:** With 29 training points in 9D LHS, most holdouts have reasonably close neighbors (frequency ~1.4% proves this). **High errors at low distance** would implicate alignment/proxy, not sparse sampling.

### B.3 Repeatability / target variance

**LHS geometry repeatability** (script section `repeatability_lhs_geometry`): closest pairs in 9D parameter space.

**Mode-level repeatability** (manual follow-up on VM): for the 10 closest sample pairs, match deduped modes within 2 Hz and compare:

```text
|Δ mic_output_proxy_p95_norm|
|Δ radiation_proxy_p95_norm|
```

If typical Δ > 0.15–0.20 for "similar" guitars, **noise floor** exceeds target MAE 0.10–0.15 → **(D)**.

**Prior audit signal** ([M4_AIR_MIC_PROXY_SANITY_AUDIT.md](M4_AIR_MIC_PROXY_SANITY_AUDIT.md)): ~281 Hz air family shows **near-identical mic proxy across large geometry swings** → proxy **insensitivity** **(E)** caps learnable signal.

### B.4 Alignment quality

Script reports per holdout:

- `overall_class_match_rate` — Hz-nearest neighbor mode has same `coupling_class` as FOM mode
- `by_coupling_class`, `by_band` breakdowns

**Also compare** scalar errors for:

| Subset | Expected if (C) dominant |
|--------|--------------------------|
| Class match = true | Lower p95_norm MAE |
| Class match = false | Much higher MAE |
| `air_dominant` | Higher error (known proxy cluster) |
| `top_back_mixed` | Moderate |
| `back_dominant` | Moderate |

### B.5 Baselines (same 29-train LOO)

Script compares:

| Baseline | Description |
|----------|-------------|
| `v21_knn_idw` | Current production v2.1 |
| `nearest_single` | k=1 neighbor only |
| `global_band_mean` | Band mean of neighbor `p95_norm` |
| `class_aware` | Hz-nearest with class/region penalty |
| `class_band_aware` | Class-aware + restrict to frequency band |

| Method | Mic p95 MAE | Rad p95 MAE | Rad rank | Top-k rad | Freq med err |
|--------|-------------|-------------|----------|-----------|--------------|
| global_band_mean | *VM* | *VM* | *VM* | *VM* | *VM* |
| nearest_single | *VM* | *VM* | *VM* | *VM* | *VM* |
| **v21_knn_idw** | **0.233** | *VM* | **0.164** | **0.235** | **1.42%** |
| class_aware | *VM* | *VM* | *VM* | *VM* | *VM* |
| class_band_aware | *VM* | *VM* | *VM* | *VM* | *VM* |

**Decision:** Adopt v2.2 only if `class_band_aware` (or better) beats v2.1 on rank/top-k **without** frequency regression.

### B.6 Root-cause verdict matrix

| Code | Hypothesis | Evidence today | Confidence |
|------|------------|----------------|------------|
| **A** Insufficient samples | More FOM → better intensity | Freq good at 30; intensity weak | Medium — needs learning curve |
| **B** Weak k-NN/IDW intensity | Global blender wrong for amplitudes | Rank 0.16, top-k 0.24 | **High** |
| **C** Modal correspondence | Hz-nearest ≠ same physical mode | Class ~88% but rank poor; index/Hz split | **High** |
| **D** Noisy targets | Similar guitars differ | Needs pair audit | Medium |
| **E** Coarse FOM proxy | Mic cluster insensitivity | Documented air/mic audit | **High** |

**Combined verdict:** **B + C + E primary; A secondary for MAE but not sufficient for rank ≥ 0.6 at N=30.**

---

## Task C — Candidate v2.2 approaches (ranked)

### Option 1 — Class-aware modal matching ⭐ **Recommended core**

Prefer neighbor modes with matching `coupling_class` / `dominant_region` when selecting Hz-nearest; penalize or exclude mismatches.

| Pros | Cons |
|------|------|
| Directly attacks **(C)** | Needs predicted class first (already ~88% accurate) |
| ROM-only, small diff | Wrong class vote poisons match |
| No new dependencies | Tuning penalty weights |

**Rank:** #1 for minimal experiment.

### Option 2 — Frequency-band-specific predictors

Bands (data-driven starting point):

```text
60–150 Hz   (fundamental / first air)
150–300 Hz  (low-mid; 281 Hz cluster)
300–425 Hz  (mid; bridge/top)
425–550 Hz  (upper air / local)
```

Separate neighbor mode pools or band-conditioned means.

| Pros | Cons |
|------|------|
| Reduces cross-band contamination | More hyperparameters |
| Complements Option 1 | Band boundaries arbitrary |

**Rank:** #2 — combine with Option 1.

### Option 3 — Derived geometric/acoustic features

| Feature | Already implicit? | Intensity value |
|---------|-------------------|-----------------|
| length, width, depth, thicknesses, hole_radius | **Yes** (explicit) | High for freq; moderate for intensity |
| `cavity_volume ≈ L×W×D` | **No** (product) | Medium — air modes |
| `hole_area = π r²` | **No** | Medium — mic proxy |
| `hole_area / cavity_volume` | **No** | High for air/mic |
| `L/W`, `D/L` | **No** | Medium |
| Wood one-hot vs index | Partial | Low–medium |

**Rank:** #3 — add in regression experiment, not first k-NN patch.

### Option 4 — Dedicated amplitude regression

Keep frequency k-NN; regress `p95_norm` from `[LHS features, pred_freq, predicted_class, mode_index]`.

| Model | Dependency | Fit |
|-------|------------|-----|
| Ridge | `numpy` only | Good first test |
| Random forest / GBM | `sklearn` | Needs approval |
| MLP | heavy | Not justified yet |

Nested LOO: train regressor on 29 guitars' **matched** mode rows only.

**Rank:** #4 — second experiment if matching patch insufficient.

### Option 5 — Ranking / top-k model

Predict **ordinal rank** or **top-20% label** per mode instead of absolute `p95_norm`.

| Pros | Cons |
|------|------|
| Aligns with STK weighting | Still needs correspondence |
| May be easier than absolute | Harder to calibrate level |

**Rank:** #5 — metric/loss change, good v2.3 direction.

### Option 6 — Per-class calibration

`pred = base_intensity × calibration_factor(coupling_class)`.

**Rank:** #6 — cheap add-on after Option 1.

### Recommended combination for v2.2 experimental

```text
Option 1 + Option 2 (+ light Option 6 calibration)
```

Keep frequency path **byte-identical** to v2.1.

---

## Task D — STK relevance

### D.1 What STK uses today

From `cpp/guitar_stk.cpp` and `gui/app.py`:

| Input | Source today | Intensity sensitivity |
|-------|--------------|------------------------|
| `modes_hz[]` | ROM frequencies | **Critical** — resonator tuning |
| `mode_weights[]` | **Fallback rolloff** `1/(1+0.25·i)` in GUI ROM path | **High** — drives wet level per mode |
| Wood `Q` | User/material | Medium |
| `rad_k`, `wet_gain`, `mix` | User knobs | Post-hoc |

**Critical gap:** GUI `write_stk_body_json()` does **not** yet map `radiation_proxy` / `mic_output_proxy` into `mode_weights`. M4 FOM JSON can carry `mode_weights`; ROM Phase-2 predicts proxies but STK path ignores them.

### D.2 STK requirements (answered)

| Question | Answer |
|----------|--------|
| 1. Absolute proxy magnitude required? | **Not strictly** — STK normalizes weights to max=1 per guitar |
| 2. Per-guitar normalized amplitude sufficient? | **Yes** — `p95_norm` or rank-based weights are STK-appropriate |
| 3. Rank / top-k vs exact magnitude? | **Rank/top-k more important** for timbre coloration than 10% absolute error on weak proxies |
| 4. Best correlation with audible similarity? | **Top modes overlap + share/class correctness**; absolute mic proxy least trusted (see air/mic audit) |
| 5. Hybrid viable? | **Yes — recommended:** ROM frequencies + classes/shares + **rank-calibrated weight envelope** + per-note normalization |

### D.3 Measurable “good enough for STK” (proposed)

| Tier | Criteria | Purpose |
|------|----------|---------|
| **P0** | Frequency med err ≤ 2%; class ≥ 85% | Resonator locations |
| **P1** | `top_k_radiation_overlap` ≥ **0.45** (20%) | Loud modes coverage |
| **P2** | `radiation_proxy_rank_correlation` ≥ **0.45** | Relative shading |
| **P3** | `mic_output_proxy_p95_norm_mae` ≤ **0.18** | Air/mic envelope |
| **Stretch** | rank ≥ 0.6, top-k ≥ 0.6, MAE ≤ 0.12 | User target — ambitious |

**Do not gate on raw relative proxy error** — diagnostic only.

---

## Task E — Minimal implementation proposal

### E.1 Recommended v2.2 experimental model

| Field | Value |
|-------|-------|
| `model_version` | `m4_modal_surrogate_v2_2_intensity_experimental` |
| `prediction_method` | `knn_idw_frequency_v2_1_plus_class_band_intensity_v2_2` |
| `surrogate_schema` | `m4_modal_surrogate_v2_2_experimental` |
| Artifacts | `ROM/classic/m4_modal_surrogate_v2_2_experimental.json` (+ `.npz`) — **do not overwrite** v2.1 |

### E.2 Algorithm delta (intensity only)

1. **Frequency:** unchanged v2.1 sorted-index IDW.
2. **Classification:** unchanged vote (feeds matching).
3. **Intensity scalars:**
   - Predict `coupling_class`, `dominant_region` first (existing).
   - For each `pred_freq`, select neighbor mode by **scored nearest frequency**:

     ```text
     score = |ΔHz| + λ_class·𝟙[class≠pred] + λ_region·𝟙[region≠pred] + λ_band·𝟙[band mismatch]
     ```

     Defaults: `λ_class=8`, `λ_region=3`, band filter on 60–150 / 150–300 / 300–425 / 425–550.
   - IDW blend selected `p95_norm` and raw proxies.
   - **Inference fix:** recompute holdout `p95_norm` from blended **raw** using holdout's own p95 scale (or predict raw + normalize at end) to match FOM evaluation.

4. Optional **per-class calibration** on training residuals (air/top_back/back multipliers).

### E.3 Implementation footprint

| File | Change |
|------|--------|
| `v2_b3_m4_rom_scalar_fields.py` | `predict_mode_scalars_class_band_aware()`, post-hoc p95 renormalization |
| `v2_b3_m4_modal_surrogate_lib.py` | experimental schema flag; separate save path |
| `build_m4_rom_from_completed_fom.py` | `--experimental-v2-2` flag |
| `run_m4_rom_compare.py` | `--model-json` override for LOO |
| New: `compare_rom_intensity_v21_v22.py` | 30-sample LOO table |

**Estimated effort:** ~1–2 days ROM-only; no FOM rerun.

### E.4 Comparison table (fill after experiment)

| Metric | v2.1 aggregate | v2.2 aggregate | Δ abs | Δ relative |
|--------|----------------|----------------|-------|------------|
| Frequency median error | 1.42% | *TBD* | *TBD* | *TBD* |
| Mic p95 norm MAE | 0.233 | *TBD* | *TBD* | *TBD* |
| Radiation p95 norm MAE | *TBD* | *TBD* | *TBD* | *TBD* |
| Bridge p95 norm MAE | *TBD* | *TBD* | *TBD* | *TBD* |
| Radiation rank corr. | 0.164 | *TBD* | *TBD* | *TBD* |
| Mic rank corr. | *TBD* | *TBD* | *TBD* | *TBD* |
| Top-k radiation overlap | 0.235 | *TBD* | *TBD* | *TBD* |
| Top-k mic overlap | *TBD* | *TBD* | *TBD* | *TBD* |
| Runtime median (s) | 13.7 | *TBD* | *TBD* | *TBD* |

**Adopt v2.2 only if:**

- Frequency median error ≤ **~1.8%** (no material regression)
- Mic or radiation p95 MAE improves ≥ **15–20% relative**
- Rank or top-k improves ≥ **30% relative** (e.g. rank 0.16 → 0.21+ minimum; stretch 0.35+)
- Runtime ≤ **~20 s** median
- All validation remains LOO / no leakage

### E.5 Sample-count estimates (honest uncertainty)

Assuming learning curve shows **diminishing returns after ~20–24 samples** for rank:

| Target mic p95 MAE (absolute) | Samples (rough) | Confidence |
|-------------------------------|-----------------|------------|
| ≤ 0.20 | 35–45 | Low |
| ≤ 0.15 | 45–70 | Low |
| ≤ 0.10 | 70–120+ | Very low without algorithm change |

**Without v2.2 alignment fix, rank ≥ 0.6 at N≤100 is unlikely.** Proxy ceiling **(E)** may cap rank regardless of N.

---

## VM command reference

### 1. Diagnostics (Task B) — run first

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_rom_intensity_v22_diagnostics.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --samples 0-29 \
  --json-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V22_DIAGNOSTICS.json \
  --csv-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V22_HOLDOUT_PER_SAMPLE.csv
```

Paste aggregate rows into §B.1 and §B.5 of this document.

### 2. Reconfirm v2.1 baseline (optional)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_m4_rom_from_completed_fom.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --completed-only \
  --max-samples 30

for i in $(seq 0 29); do
  SID=$(printf "sample_%03d" "$i")
  python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_rom_compare.py \
    --lhs-json ROM/classic/lhs_pool.json \
    --force-sample "$SID" \
    --exclude-target-from-training
done
```

### 3. v2.2 experiment (after implementation)

```bash
# Build experimental artifact only (not production manifest)
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_m4_rom_from_completed_fom.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --completed-only \
  --max-samples 30 \
  --experimental-v2-2

# Compare v2.1 vs v2.2 LOO (script TBD)
python FEM/experiments/active_domain_validation/physics_integrity/scripts/compare_rom_intensity_v21_v22.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --samples 0-29 \
  --json-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V21_V22_COMPARE.json
```

(`--experimental-v2-2` and `compare_rom_intensity_v21_v22.py` are **planned** — not yet implemented.)

---

## Key findings (concise)

1. **Frequency and classification are not the bottleneck** — 9D k-NN works for smooth spectra.
2. **Intensity uses a different, weaker mechanism** — Hz-nearest scalar pull ignores modal family.
3. **v2.1 p95_norm blends neighbors' normalized values** — mismatches holdout evaluation normalization.
4. **Rank/top-k at ~0.16 / ~0.24** vs target **0.6** indicates **correspondence + proxy limits**, not just "need more samples."
5. **STK cares about weight ranking** more than absolute proxy error; ROM→STK weight mapping not wired yet.
6. **Minimal v2.2:** class+band-aware matching + inference-time p95 renormalization; keep frequency frozen.
7. **User targets (MAE 0.10–0.15, rank 0.6)** may require **algorithm change + more samples + STK weight integration**; rank 0.6 at N=30 is **optimistic**.

---

## Clear conclusion

| Limitation | Share of problem |
|------------|------------------|
| **Algorithm (B)** — IDW on wrong modes | **~30%** |
| **Alignment (C)** — index/Hz split, no class filter | **~35%** |
| **Proxy (D+E)** — coarse/insensitive mic & radiation | **~25%** |
| **Data (A)** — 30 LHS points | **~10%** for MAE; **<10%** for rank at current ceiling |

**Overall:** **Combination, dominated by B+C with proxy ceiling E.** More FOM alone is **insufficient** for rank/top-k ≥ 0.6; v2.2 matching + normalization fix is the justified next step before scaling to new shapes or large sample counts.

---

## References

- [M4_ROM_INTENSITY_PREDICTION_AUDIT.md](M4_ROM_INTENSITY_PREDICTION_AUDIT.md) — v2.0 audit
- [M4_AIR_MIC_PROXY_SANITY_AUDIT.md](M4_AIR_MIC_PROXY_SANITY_AUDIT.md) — proxy insensitivity
- [M4_ROM_OUTPUT_GAP_ANALYSIS.md](M4_ROM_OUTPUT_GAP_ANALYSIS.md) — STK field gaps
- Code: `v2_b3_m4_modal_surrogate_lib.py`, `v2_b3_m4_rom_scalar_fields.py`, `v2_b3_m4_rom_fom_compare_lib.py`
