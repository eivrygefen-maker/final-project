# BOX vs CLASSIC pipeline forensic report

**Date:** 2026-06-16  
**Scope:** Code inspection + documented VM/run metadata only. **No production code changes.** **No FEM/FOM/STK/WAV/C++ executed in Cursor.**  
**VM is runtime authority** — several conclusions require VM artifact confirmation (paths and commands below).

---

## Executive summary

CLASSIC and BOX share a **single unified M4 FOM pipeline** (`run_m4_production_pipeline.py` → scout → L_prod mesh/checkpoint → parallel ST workers → aggregation → freeze). CLASSIC’s 67-completed baseline reliably yields **~500–600 deduped modes** per run in the 60–550 Hz discovery band. BOX `box_sample_000` completes (`AGGREGATION_PASS`, 12/12 chunks) but yields only **~9–10 deduped modes** from a similar target plan (~58 targets).

**CLASSIC preservation:** Code review shows **PASS with medium residual risk** — CLASSIC locked validation profile and shape guards exist, but **solver acceptance, assembly, and worker env are globally shared**. Recent shape-aware work added BOX paths without changing locked CLASSIC thresholds, yet any change to shared modules (`v2_b3_st_sinvert_solver_lib.py`, aggregation, compaction) affects all shapes.

**Most likely root-cause categories (pending VM raw-catalog counts):**

| Rank | Category | Likelihood | One-line reason |
| --- | --- | --- | --- |
| 1 | **E** — Raw diagnostic not active in workers | **HIGH** | `BOX_RAW_MODAL_DISCOVERY` is **not** in worker subprocess env allowlist |
| 2 | **D** — Classic-guitar assumptions on BOX | **HIGH** | Same `support_participation_fail` + plate/cavity/soundhole assembly for all shapes |
| 3 | **B** — Solver/target ineffectiveness | **MEDIUM–HIGH** | Same `nev=12` cap; if raw converged counts are low, ST setup is suspect |
| 4 | **F** — Plots/catalogs show filtered modes only | **MEDIUM** | Default plots use deduped accepted catalog by design |
| 5 | **C** — Assembly/BC over-constraint | **MEDIUM** | Large `active_dim` (~167k) but few modes → unlikely “empty matrix”; more likely filter/solver path |
| 6 | **A** — BOX genuinely has few modes | **LOW** | Simple box cavity should still have many modes in 60–550 Hz if assembly/solver are healthy |

**Immediate VM action:** Inspect `validation/raw_solver_candidate_catalog.json` and per-chunk `raw_modal_diagnostic.jsonl`. If missing or empty → **E confirmed**. If raw counts ≫ 9 but accepted ≈ 9 → **D/F confirmed**.

---

## Evidence table (high-signal)

| Observation | CLASSIC (documented) | BOX `box_sample_000` (reported) | Implication |
| --- | --- | --- | --- |
| Pipeline entry | `run_m4_production_pipeline.py` + `ROM/classic/lhs_pool.json` | Same script + `ROM/box/lhs_pool.json` | Shared orchestration |
| Deduped modes / run | ~500–600 (`M4_ROM_OUTPUT_GAP_ANALYSIS.md`) | ~9 | ~50× mode loss |
| Completed LHS sims | 67 frozen baseline | Early BOX samples | CLASSIC is reference |
| Target band | 60–550 Hz discovery (`DEFAULT_DISCOVERY_BAND_HZ`) | Same | Not a band mismatch |
| Target count (typical) | ~58 gapless L_prod plan | ~58 (modal audit tests / reports) | Not target-plan sparsity vs CLASSIC |
| `nev` / `ncv` default | 12 / 24 (`v2_b3_checkpoint_solve_target_list.py`) | Same | Not different solver request |
| Acceptance policy | `ACCEPTANCE_POLICY_DISCOVERY` + per-target windows | Same (`acceptance_config_from_chunk_targets`) | Shared filters |
| `support_participation_fail` | Applied in `collect_accepted_st_modes` | Same | Guitar-oriented structural/air support test |
| Scout density policy | `intrinsic_discovered_modes_v1` | `box_discovered_modes_v2` (softer) | BOX scout gates relaxed; **worker acceptance unchanged** |
| Validation profile | `classical_guitar_reference_v1_locked` | `box_body_plausibility_v1` | Advisory only; does not change worker acceptance |
| L_prod mesh nodes | Classic ROM runs vary (e.g. 177k–409k generated; older operator audit showed fixed `active_dim` on legacy path) | ~119,409 nodes, ~632,437 tets | BOX uses shape-specific Gmsh mesh |
| `active_dim` | Classic ROM reference ~316k (legacy audit) or profile-dependent | ~167,221 | Large DOF space — few modes ≠ tiny model |
| `BOX_RAW_MODAL_DISCOVERY=1` in batch log | N/A | Parent shell only | **Does not prove worker diagnostic ran** |

---

## CLASSIC preservation check

**Status: PASS (thresholds frozen) with MEDIUM shared-code risk**

### Locked CLASSIC artifacts (unchanged in repo)

| Item | Location | Status |
| --- | --- | --- |
| Locked profile ID | `FEM/scripts/m4_shape_validation_profile.py` → `CLASSIC_LOCKED_PROFILE_ID = "classical_guitar_reference_v1_locked"` | Present |
| Config mirror | `FEM/configs/shape_validation_profiles.json` → `classical_guitar_reference_v1_locked` with `"locked": true` | Present |
| Reference baseline label | `classic_lhs_67_simulations` | Present |
| Locked field guard | `_LOCKED_PROFILE_FIELD_KEYS` — overrides cannot change thresholds when `locked: true` | Present |
| CLASSIC LHS pool | `ROM/classic/lhs_pool.json` — overnight batch uses `--ensure-existing` for classic | Protected by ops convention |

### Recently changed shared files (git `HEAD~15..HEAD`, shape-related)

These touch **all shapes** if imported on the shared path:

- `v2_b3_st_sinvert_solver_lib.py` — acceptance + `collect_all_st_mode_candidates` (diagnostic only when flag set)
- `v2_b3_checkpoint_solve_target_list.py` — worker solve + audit hooks
- `v2_b3_m4_aggregate_worker_results.py` — aggregation + catalog merge
- `v2_b3_m4_minimal_rom_compaction.py` — retention lists
- `v2_b3_m4_lprod_interfaces.py`, `v2_b3_m4_lprod_checkpoint_run.py`, `v2_b3_m4_pipeline_run_scout.py`
- `m4_shape_context.py`, `m4_shape_registry.py`, `m4_shape_validation_profile.py`
- `tools/run_shape_fom_overnight_batch.sh`

**BOX/acoustic-only logic** is mostly gated by `shape_name` / `geometry_shape_type` / `sample_id` prefix — not by changing CLASSIC profile thresholds.

### Shape guards that protect CLASSIC

- `resolve_shape_context_from_sample_input` — non-classic must not resolve to `geometry_shape_type=Classical` when registry says Box/Acoustic
- `m4_shape_registry.infer_shape_from_sample_id` — `sample_*` → classic, `box_sample_*` → box
- `evaluate_shape_physical_acceptance.py` — advisory; classic uses locked profile
- Overnight batch: `ALLOW_CLASSIC_REGEN` gate prevents accidental LHS regen

### Code paths that **can** still affect CLASSIC

| Path | Risk | Why |
| --- | --- | --- |
| `collect_accepted_st_modes` | **HIGH** | Global acceptance; any change affects all shapes |
| `acceptance_config_from_chunk_targets` | **HIGH** | Same discovery band/windows for BOX and CLASSIC |
| `v2_b3_m4_aggregate_worker_results._dedupe_catalog` | **MEDIUM** | Shared dedup tolerance 0.05 Hz |
| Compaction retain lists | **LOW** | Additive optional files |
| Modal discovery audit | **NONE** | Advisory outputs only |

### Risk level

**MEDIUM** — CLASSIC **thresholds and LHS** are frozen, but **physics path is not forked**. BOX and CLASSIC share the same V2/B3 assembly export, ST solver, and acceptance gates.

### Recommendation

Until BOX mode yield is explained, treat **solver + acceptance + assembly** as **shared risk surface**. Prefer **shape-specific worker acceptance modules** (copy-on-write from CLASSIC) rather than further edits to shared `collect_accepted_st_modes` without CLASSIC regression tests on a frozen sample.

---

## CLASSIC vs BOX pipeline stage comparison

| Stage | CLASSIC behavior | BOX behavior | Shared code? | Risk | Evidence |
| --- | --- | --- | --- | --- | --- |
| LHS / sample input | `ROM/classic/lhs_pool.json`, `sample_NNN`, bounds from classic branch | `ROM/box/lhs_pool.json`, `box_sample_NNN`, box bounds (length/width/depth/hole) | Same bridge (`v2_b3_m4_lhs_pool_bridge.py`), different pool | LOW | `m4_shape_registry.py` |
| Resolved core config | `geometry.shape_type=Classical` | `geometry.shape_type=Box` | Same resolver; shape injected via `ShapeContext` | LOW | `m4_shape_context.py`, `v2_b3_m4_lprod_interfaces.py` |
| Geometry builder | Gmsh morph from `classic.step` | Gmsh morph from box reference STEP family | Same `FEM/geometry/build_3d_guitar.py` | **MEDIUM** | `_reference_shape_family`, `_is_box_shape` branches |
| Gmsh mesh | `build_lprod_mesh_for_case` | Same | **YES** | LOW | `v2_b3_m4_lprod_mesh_build.py` |
| Physical tags | Volumes 1/2/3/10 → top/back/ribs/air; facets 1–5 | **Same tag contract** | **YES** | **HIGH** | `REQUIRED_VOLUME_TAGS`, `REQUIRED_FACET_TAGS` in lprod mesh build |
| Facet tags incl. soundhole | soundhole facet + fixed BC | **Same policy** (`has_soundhole=True` for box in registry) | **YES** | **HIGH** | `m4_shape_registry` box entry |
| DOF / region export | `region_dof_indices.npz`, aperture mask | Same production contracts | **YES** | **HIGH** | `v2_b3_m4_production_contracts.py` |
| BC / support | Dirichlet on fixed facets; inactive structural DOFs | Same B3 checkpoint export | **YES** | **MEDIUM** | `v2_b3_checkpoint_export.py` (shared Stage A) |
| Matrix assembly | V2/B3 coupled A/M on active DOFs | Same | **YES** | **MEDIUM** | `v2_b3_operator_checkpoint_portable` |
| Active DOF reduction | `built_metadata.json` → `active_local`, `inactive_local`, `bc_rows` | Same | **YES** | MEDIUM | `built_from_checkpoint_metadata` |
| Checkpoint export | Portable NPZ operators | Same | **YES** | LOW | `v2_b3_m4_lprod_checkpoint_run.py` |
| Target plan | Gapless 60–550 Hz, ~58 targets, zone spacing 6/9/12.5 Hz | **Same planner** | **YES** | LOW | `v2_b3_m4_scout_planner_lib.py` |
| ST shift-invert solve | `run_checkpoint_st_target`, `nev=12`, `ncv=24` | Same | **YES** | **HIGH** | `v2_b3_checkpoint_solve_target_list.py` |
| Candidate collection | `collect_accepted_st_modes` gates | Same | **YES** | **HIGH** | `v2_b3_st_sinvert_solver_lib.py` |
| Raw diagnostic (optional) | Off unless env + box | Intended on with `BOX_RAW_MODAL_DISCOVERY=1` | Same hook | **HIGH** | Env not passed to workers (see below) |
| Acceptance | Discovery band + window + residual + support participation | Same | **YES** | **HIGH** | `collect_accepted_st_modes` |
| Worker result | `accepted_mode_records` from accepted only | Same | **YES** | HIGH | `v2_b3_checkpoint_solve_target_list.py` |
| Aggregation | Merge accepted; dedup 0.05 Hz | Same | **YES** | MEDIUM | `v2_b3_m4_aggregate_worker_results.py` |
| Freeze / export | Shared freeze gate | Same | **YES** | LOW | `v2_b3_m4_production_freeze.py` |
| Scout density verify | `intrinsic_discovered_modes_v1` | `box_discovered_modes_v2` (warnings softened) | Shape-specific policy id only | LOW | `v2_b3_m4_scout_intrinsic_coverage.py` |

**Conclusion:** BOX is **not a separate physics pipeline**. It is CLASSIC’s V2/B3 guitar-cavity-plate **assembly and acceptance stack** with **Box Gmsh geometry** and **softer scout density scoring**.

---

## Assembly compatibility risk

### Is V2/B3 assembly physically valid for BOX?

**Partially / unproven.** The assembly engine assumes a **guitar-like volumetric decomposition**:

- Plates: top (vol 1), back (vol 2), ribs/sides (vol 3)
- Air cavity (vol 10)
- Facets: top, back, ribs, **soundhole**, fixed support
- Global **aperture mask** and `p_idx_aperture` mic proxy (`aperture_pressure_rms_proxy_v1`) — **required for production** even on BOX (`m4_shape_registry`: `has_soundhole=True`, `requires_aperture_mask=True`)

BOX LHS still sweeps `geometry.hole_radius`; the box body gets a **soundhole and guitar-style opening policy** not native to a simple box.

### Classic-guitar assumptions reused for BOX

| Assumption | Evidence |
| --- | --- |
| Top/back/ribs/air region split | Same tag map in mesh build |
| Soundhole + aperture DOFs | BOX registry `has_soundhole=True` |
| `support_participation_fail` in solver | `u_norm > 1e-8` AND `(p_support > 1e-6 OR ...)` — favors coupled guitar-like modes |
| Bridge / radiation / mic proxies | Attached post-acceptance; metadata is guitar-oriented (`v2_b3_mode_audio_coupling.py`) |
| Participation on top/back/ribs | `region_dof_indices.npz` — classical plate layout |

### BOX tag mapping

Mesh build validates **identical** `REQUIRED_VOLUME_TAGS = (1, 2, 3, 10)` and `REQUIRED_FACET_TAGS = (1, 2, 3, 4, 5)`. BOX Gmsh path uses box-specific morph (`_is_box_shape`) but **emits the same semantic tags** so downstream assembly unchanged.

### Over-constraint / stiffness hypotheses

- **Dirichlet/fixed facets** — same facet tag 5 “fixed” contract; could over-constrain a box differently than a guitar if fixed patch area differs.
- **Inactive DOF elimination** — `inactive_dof_violation` rejects modes with energy on inactive structural DOFs; shared.
- **Large `active_dim` (~167k) with ~10 modes** — suggests the **operator is non-trivial** but **mode discovery/acceptance** is failing, not an empty DOF space.

### Assembly compatibility verdict

**MEDIUM–HIGH risk** that BOX modes are filtered or poorly excited because the **classical guitar FSI/participation model** is applied to a **box morphed into guitar-like tags**, not because BOX has no elastic modes.

---

## Numerical scale comparison

### Expected CLASSIC ranges (from repo docs / audits)

| Metric | CLASSIC (indicative) | Source |
| --- | --- | --- |
| Deduped modes | ~500–600 | `M4_ROM_OUTPUT_GAP_ANALYSIS.md` |
| Generated mesh nodes | ~177k–410k (varies by sample) | `M4_OPERATOR_MESH_AND_SOUNDHOLE_ROOT_CAUSE.md` |
| Legacy operator `active_dim` | 316,017 (fixed topology era) | Same doc — **may not apply to current ROM mesh profile** |
| Target count | ~58 in 60–550 Hz plan | Scout planner / modal audits |
| Modes per target (implied) | ~500/58 ≈ **8–9 accepted** if uniform | Derived |

### BOX `box_sample_000` (user-reported VM)

| Metric | Value |
| --- | --- |
| `n_nodes` | ~119,409 |
| `n_tetra` | ~632,437 |
| `active_dim` | ~167,221 |
| Volume tags | top / back / ribs / air |
| Facet tags | top / back / ribs / soundhole / fixed |
| Raw modes | ~10 |
| Deduped modes | ~9 |
| Modes per target (implied) | ~9/58 ≈ **0.15 accepted** |

### Does `active_dim ≈ 167k` make sense for only ~10 modes?

**Yes, that combination is a red flag for acceptance/solver path, not for “model too small.”** A 167k-DOF coupled problem should support **many** eigenmodes in 60–550 Hz. Getting **O(10) accepted** modes implies:

1. ST solve returns few converged eigenpairs per target (`nev=12` upper bound), **and/or**
2. `collect_accepted_st_modes` rejects most converged pairs (`outside_acceptance_window`, `support_participation_fail`, etc.), **and/or**
3. Aggregation only sees accepted modes (by design).

**VM compare commands** (read-only):

```bash
# CLASSIC reference run (pick one completed)
C_RUN="FEM/experiments/.../guitars/sample_000/runs/sample_000_rom_official_v1"
B_RUN="FEM/experiments/.../guitars/box_sample_000/runs/box_sample_000_box_fom_v1"

for R in "$C_RUN" "$B_RUN"; do
  echo "=== $R ==="
  jq '.matrix_contract,.built_metadata_diag,.aggregate' "$R/lprod/checkpoint/built_metadata.json" 2>/dev/null | head
  jq '{raw,deduped,targets:.total_targets_attempted}' "$R/aggregation/aggregation_result.json"
done
```

---

## Solver and target plan comparison

### Shared configuration (CLASSIC = BOX)

| Parameter | Value | Code |
| --- | --- | --- |
| Discovery band | 60–550 Hz | `DEFAULT_DISCOVERY_BAND_HZ` in `v2_b3_m4_lprod_interfaces.py` |
| Acceptance policy | `discovery_band_and_target_window` | `acceptance_config_from_chunk_targets` |
| Per-target windows | From chunk `window_hz` | Per-target map in `AcceptanceConfig` |
| Default half-width fallback | 6.25 Hz | `target_window_half_width_hz=6.25` |
| `nev` / `ncv` | 12 / 24 | `v2_b3_checkpoint_solve_target_list.py` argparse defaults |
| Factor solver | `mkl_pardiso` | Worker command builder |
| Residual gate | `eps_err <= 1e-4` | `collect_accepted_st_modes` |
| Legacy band constants (unused in discovery) | 220–265 Hz | `ACCEPTANCE_FREQ_LO_HZ/HI` — discovery mode overrides per-target |

### Target plan density

Gapless planner (`v2_b3_m4_scout_planner_lib.py`):

- Zone spacing: 6.0 / 9.0 / 12.5 Hz
- Uniform baseline reference: 5.5 Hz
- Typical L_prod target count ≈ **58** over 60–550 Hz (BOX audits)

**CLASSIC and BOX use the same target planner** on the shared pipeline; BOX is **not** on a sparser target grid than CLASSIC.

### Per-target mode yield math

If each target accepted **1 mode** on average → ~58 raw modes before dedup. BOX observes **~10 raw** → average **~0.17 modes/target**, i.e. **~94% loss per target** vs the naive 1 mode/target expectation.

CLASSIC **~500 modes** with **~58 targets** → **~8.6 modes/target** (upper bound if all unique frequencies).

### Shift-invert validity

Same `configure_eps_krylovschur_sinvert` for all shapes. BOX matrices differ in values/sparsity pattern from geometry; ST **may** converge to different shift spectra — requires per-target `converged_mode_count` from `solver_result.json` or raw diagnostic.

### Solver comparison verdict

**Not a target-plan sparsity issue.** BOX and CLASSIC share band, spacing, `nev`, and acceptance. The gap is almost certainly **per-target converged/accepted yield**, not missing targets.

---

## Raw diagnostic validity check

### Implementation (code)

| Component | Behavior |
| --- | --- |
| Enable gate | `box_raw_modal_discovery_enabled()` requires `BOX_RAW_MODAL_DISCOVERY=1` **and** shape `box` |
| Worker hook | `run_checkpoint_st_target(..., raw_diagnostic=True)` → `collect_all_st_mode_candidates` |
| Per-chunk file | `worker_results/<chunk>/raw_modal_diagnostic.jsonl` |
| Run-level merge | `merge_box_raw_catalogs_for_run` → `validation/raw_solver_candidate_catalog.json`, etc. |
| Plots | `raw_frequency_vs_target.png`, etc. — **separate** from `mode_frequency_plot.png` |
| Modal audit | `## Raw vs filtered mode discovery` when catalog present |

### Critical gap: environment propagation

Worker subprocess env is built in `production_worker_subprocess_env` → `_solver_mkl_subprocess_env_strict` → `_minimal_subprocess_base()`.

**Inherited env keys** (`v2_b3_run_coarse_scout_lhs_batch.py`):

```python
_SUBPROCESS_INHERIT_KEYS = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "TERM")
```

**`BOX_RAW_MODAL_DISCOVERY` and `SHAPE` are NOT inherited.**

Therefore:

- Overnight batch log `SHAPE_FOM_BOX_RAW_MODAL_DISCOVERY enabled=1` only proves the **parent shell** exported the variable.
- Worker processes **likely run with `raw_diagnostic=False`** unless a chunk was solved outside the isolated env (e.g. manual CLI in a full environment).

### Are final graphs “filtered only”?

**Yes, by default.** `mode_frequency_plot.png` uses **deduped accepted** catalog (`_try_mode_plots` in aggregation). Even when raw diagnostic works, users must open:

- `aggregation/raw_frequency_vs_target.png`
- `validation/raw_solver_candidate_catalog.json`

### User conclusion check

| User belief | Supported? |
| --- | --- |
| “Filters removed but still few modes” | **Cannot confirm without raw catalog on VM** |
| “Plots show ~9 modes” | **Expected** — those are accepted/deduped plots |
| “Raw diagnostic was on” | **Not proven** — env likely **not** in worker subprocess |

### VM verification checklist

```bash
RUN=".../box_sample_000_box_fom_v1"

# 1. Worker env — did solve record diagnostic?
jq '.box_raw_modal_discovery, .raw_modal_diagnostic_count' \
  "$RUN/worker_results/"*"/solver_result.json" 2>/dev/null | head

# 2. Per-chunk raw JSONL
find "$RUN/worker_results" -name 'raw_modal_diagnostic.jsonl' -exec wc -l {} +

# 3. Merged catalogs
jq '.row_count, .raw_solver_candidate_count' \
  "$RUN/validation/raw_solver_candidate_catalog.json" 2>/dev/null

# 4. Rejection histogram
jq '.raw_vs_filtered_analysis' "$RUN/validation/modal_discovery_audit.json"
```

**Interpretation:**

- **No raw files / `box_raw_modal_discovery: false` in solver_result** → category **E** confirmed.
- **Raw row_count ≫ 9, accepted ≈ 9** → category **D** (filters) dominant.
- **Raw row_count ≈ 9** → category **B** (solver/nev/ST) dominant.

---

## Likely root cause ranking

### A. Shape physics genuinely produces few modes

- **Likelihood:** LOW  
- **Evidence:** Simple box cavity + plates should have many elastic/acoustic modes in 60–550 Hz; `active_dim` is large.  
- **Confirm:** Analytical or coarse FEM sanity on box-only structural model.  
- **Refute:** Raw catalog shows hundreds of finite candidates per run.

### B. Solver / shift-invert ineffective for BOX

- **Likelihood:** MEDIUM–HIGH  
- **Evidence:** Same `nev=12`; if `converged_mode_count` is low per target, ST shift may not capture local spectrum for box operators.  
- **Confirm:** VM `solver_result.json` → `converged_mode_count`, `diagnostic_candidate_count`.  
- **Refute:** Raw/solver converged counts high per target.

### C. Matrix assembly / BC / active DOF issue

- **Likelihood:** MEDIUM  
- **Evidence:** Guitar tag/BC model on box; fixed/soundhole facets may differ; historical audits showed topology/value split issues on classic path.  
- **Confirm:** Compare `built_metadata` inactive counts, BC rows, region DOF counts CLASSIC vs BOX.  
- **Refute:** Healthy converged modes exist in raw catalog but fail only at participation filter.

### D. Classic-guitar assumptions reused incorrectly

- **Likelihood:** HIGH  
- **Evidence:** `support_participation_fail`; soundhole/aperture required on BOX; same top/back/ribs semantics.  
- **Confirm:** Raw diagnostic rejection histogram dominated by `support_participation_fail` / `outside_acceptance_window`.  
- **Refute:** Most raw candidates pass normal filters but aggregation drops them (unlikely).

### E. Raw diagnostic not capturing candidates

- **Likelihood:** HIGH (for diagnostic layer)  
- **Evidence:** Subprocess env allowlist omits `BOX_RAW_MODAL_DISCOVERY`.  
- **Confirm:** Missing `raw_modal_diagnostic.jsonl` on VM despite batch log.  
- **Refute:** Files present with thousands of lines.

### F. Aggregation uses filtered modes only

- **Likelihood:** MEDIUM (by design, not bug)  
- **Evidence:** `_collect_mode_records_from_chunk` reads `accepted_modes` only; dedup on accepted set.  
- **Confirm:** N/A — expected. Raw catalogs are separate artifacts.  
- **Refute:** N/A — does not explain low **accepted** count at source.

---

## Recommended isolation plan

### Freeze (do not modify without CLASSIC regression)

- `ROM/classic/lhs_pool.json` and completed 67-run artifacts  
- `classical_guitar_reference_v1_locked` profile thresholds  
- CLASSIC production command defaults (`rom_official_v1` lineage)  
- Shared acceptance function **behavior** for `shape_name=classic` (branch, don’t edit in place)

### Copy (fork for BOX experimentation)

- `acceptance_config` + `collect_accepted_st_modes` policy module → `box_acceptance_policy.py`  
- Optional simplified assembly export profile (box: no soundhole requirement / relaxed support test)  
- BOX target planner knobs (if ever needed) — currently same as CLASSIC  
- Worker env builder: explicit `BOX_RAW_MODAL_DISCOVERY` propagation

### Keep shared (safe utilities)

- LHS bridge I/O, run manifest, compaction shell, logging  
- Aggregation **machinery** (JSONL merge, dedup shell) with shape-specific inputs  
- Plotting, modal audit, freeze gate orchestration  
- `m4_shape_context` / registry (metadata only)

### Shape-specific (must diverge)

| Concern | CLASSIC | BOX |
| --- | --- | --- |
| Geometry / Gmsh | `Classical` STEP morph | `Box` morph |
| Acoustic opening | Soundhole + aperture mask | Should be **revisited** — box may need slot/port model |
| Acceptance | Locked guitar-like | Relaxed / box-native participation |
| Scout density | `intrinsic_discovered_modes_v1` | `box_discovered_modes_v2` (already) |
| Validation profile | Locked reference | `box_body_plausibility_v1` (already) |

### Minimal implementation plan (post-review)

1. **Fix diagnostic plumbing** — add `BOX_RAW_MODAL_DISCOVERY`, `SHAPE` to worker env when set (no acceptance change).  
2. **Re-run one BOX sample** — read raw vs filtered catalogs.  
3. **Branch acceptance** — `if shape==box: support_ok = ...` (box-native rule) behind flag; CLASSIC uses frozen branch.  
4. **CLASSIC regression** — single frozen `sample_000` mode-count parity test on VM.  
5. Only then consider assembly/aperture policy fork for BOX.

### Avoid breaking CLASSIC

- Never change locked profile JSON thresholds.  
- Use `shape_name` branches in new modules; do not alter default path for `sample_*` IDs.  
- Run `bash tools/run_shape_fom_smoke.sh` + one CLASSIC dry-run after any shared-file edit.

---

## Urgency / conference mode

### Option 1 — Safe short-term demo path (recommended)

- Present **CLASSIC 67× / 500+ modes** as the **validated FOM baseline**.  
- Present **BOX** as **shape-aware pipeline extension under forensic review**, not physics-complete.  
- Show: BOX geometry mesh, tag contract, pipeline stages completing, **raw vs filtered diagnostic** (once env fix verified).  
- **Do not claim** BOX modal density matches guitar until raw catalog analysis proves filter vs solver cause.

### Option 2 — Technical fix path

1. Propagate raw diagnostic env to workers (1-line allowlist + explicit merge in `production_worker_subprocess_env`).  
2. Re-run `box_sample_000` with `BOX_RAW_MODAL_DISCOVERY=1`.  
3. If raw ≫ filtered → fork BOX acceptance (support participation / window).  
4. If raw ≈ filtered → fork BOX ST setup (`nev`, shift strategy) or assembly BC review.  
5. Optional: small-box analytical benchmark outside production pipeline.  
6. CLASSIC frozen branch unchanged throughout.

---

## Immediate next actions (VM)

1. Run forensic checklist in **Raw diagnostic validity** section on `box_sample_000`.  
2. Pick one completed **CLASSIC** run (`sample_000` or official ROM) and diff `aggregation_result.json` + checkpoint `built_metadata.json` against BOX.  
3. If raw diagnostic absent → re-run **one** BOX sample after env-propagation fix (code change **after** this report is reviewed).  
4. If raw catalog shows filter dominance → schedule BOX acceptance fork; keep CLASSIC on frozen branch.  
5. Update conference slides: CLASSIC = production baseline; BOX = experimental with documented mode-gap forensic status.

---

## Appendix: mode rejection gates (shared)

From `collect_accepted_st_modes` (`v2_b3_st_sinvert_solver_lib.py`), in order of tally:

1. `nonfinite_eigenvalue`  
2. `non_positive_frequency`  
3. `outside_acceptance_window` (discovery band + per-target window)  
4. `residual_too_large` (`eps > 1e-4`)  
5. `inactive_dof_violation`  
6. `boundary_dof_violation`  
7. `lambda_near_unity`  
8. `support_participation_fail`  

Bridge/radiation/mic/top-back-air shares are **not** rejection gates in this function.

---

*Report generated from repository inspection only. No solver behavior was modified in producing this document.*
