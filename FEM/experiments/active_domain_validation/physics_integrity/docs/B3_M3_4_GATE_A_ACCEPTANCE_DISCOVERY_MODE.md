# B3 M3.4 Gate A — Acceptance / discovery mode (planning-only)

**Status:** Planning-only — no Stage B execution, no code changes authorized by this document.  
**Blocks:** Wide-band coarse scan (60–550 Hz) per M3.4-pre `execution_status=requires_acceptance_band/general_target_set_support_before_execution`  
**Related:** [`B3_M3_4_PRE_COARSE_FREQUENCY_PLANNER.md`](B3_M3_4_PRE_COARSE_FREQUENCY_PLANNER.md), M3.3 `m3exec2` PASS

---

## 0. Goal

Define a **safe, opt-in discovery mode** so Stage B can report **deduped accepted modal frequencies** across a wide band (e.g. 60–550 Hz) for coarse density calibration, **without changing** default `full9` timing behavior validated on M3.

---

## 1. Where is the acceptance band enforced?

### 1.1 Constants (hardcoded defaults)

**File:** `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_st_sinvert_solver_lib.py`

| Symbol | Value (Hz) | Role |
|--------|------------|------|
| `ACCEPTANCE_FREQ_LO_HZ` | 220.0 | Lower bound for **accepted** mode frequency |
| `ACCEPTANCE_FREQ_HI_HZ` | 265.0 | Upper bound for **accepted** mode frequency |
| `L_PROD_ST_FULL9_TARGETS_HZ` | 221.5 … 264.0 | Target list only; **not** the acceptance interval |
| `FREQ_PARITY_TOL_HZ` | 0.05 | Dedup tolerance (aggregate / parity) |

These constants are **not** derived from `--target-set full9`. They are independent policy knobs that happen to overlap the validated pilot band.

### 1.2 Core functions

| Function | Acceptance role |
|----------|-----------------|
| `collect_converged_modes(..., freq_lo, freq_hi)` | Tags each converged mode with `inside_acceptance_interval`; diagnostic |
| `collect_accepted_st_modes(..., freq_lo, freq_hi)` | **Authoritative filter:** `mode_pass` requires `inside` (line ~395) plus physics checks (EPS error, BC support, λ≈1 rejection, etc.) |
| `run_checkpoint_st_target(...)` | Calls `collect_accepted_st_modes` **without** passing `freq_lo`/`freq_hi` → always module defaults 220–265 |
| `deduplicate_frequencies_hz(...)` | Post-aggregate dedup; not acceptance |

### 1.3 Call path (Stage B / checkpoint pipeline)

```text
v2_b3_checkpoint_solve.py
  └─ run_checkpoint_solver_multi_benchmark()   [v2_b3_checkpoint_solver_multi_benchmark.py]
       └─ run_checkpoint_st_target()           [v2_b3_st_sinvert_solver_lib.py]
            ├─ configure_eps_krylovschur_sinvert(target_hz=…)  # shift center
            ├─ eps.solve()
            ├─ collect_converged_modes()       # default 220–265 tagging
            └─ collect_accepted_st_modes()       # default 220–265 acceptance
```

Same `run_checkpoint_st_target` path is used by:

- `v2_b3_checkpoint_target_density_experiment.py`
- `v2_b3_checkpoint_target_alignment_experiment.py`
- `v2_b3_checkpoint_solver_benchmark.py` (single-target wrapper)

**Metadata only (not enforcement):** benchmark scripts write `acceptance_interval_hz: [220, 265]` into JSON results; they do not pass overrides into `collect_accepted_st_modes`.

### 1.4 What is *not* acceptance today

| Mechanism | Effect |
|-----------|--------|
| ST shift `target_hz` | Locates the search; does **not** widen acceptance |
| `--target-set full9` | Selects nine shift centers only |
| `--targets-hz` override | Custom shift list only |
| Per-target “window” | **Not implemented** in acceptance logic |

---

## 2. What defines the accepted frequency interval today?

| Source | Defines acceptance? |
|--------|---------------------|
| `ACCEPTANCE_FREQ_LO_HZ` / `ACCEPTANCE_FREQ_HI_HZ` | **Yes** — global fixed interval for all checkpoint ST solves |
| `L_PROD_ST_FULL9_TARGETS_HZ` | **No** — targets only |
| CLI on Stage B | **No** — no `--acceptance-min-hz` / discovery flags |
| Interval around each target | **No** — not used in `collect_accepted_st_modes` |

**Net:** A solve shifted to 60 Hz may converge modes near 60 Hz, but they fail `inside` because 60 ∉ [220, 265]. The target row may still `status=PASS` if the solve succeeds, with **empty** `accepted_frequencies_hz`.

---

## 3. Safest way to add wide-band discovery — option comparison

### Option A — Global acceptance range CLI

```bash
--acceptance-min-hz 60 --acceptance-max-hz 550
```

| Pros | Cons |
|------|------|
| Minimal code surface | Easy to misuse on timing/full9 runs if defaults change |
| Thread `freq_lo`/`freq_hi` through existing parameters | One global band may accept spurious modes far from shift at low/high targets |
| | Hard to express “local discovery around each shift” |

**Risk:** Medium — behavior change if flags leak into production timing paths.

### Option B — Per-target acceptance window

```bash
--target-window-half-width-hz 7.5
```

Accept modes where `|f_mode - target_hz| <= half_width` (plus existing physics checks).

| Pros | Cons |
|------|------|
| Natural for shifted ST solves | Does not alone cap global band (need clip to discovery band) |
| Adapts as targets move 60→550 | Slightly more logic in `collect_accepted_st_modes` |
| | Must define behavior at band edges |

**Risk:** Low–medium if combined with a global discovery band cap.

### Option C — Explicit discovery mode (recommended)

```bash
--B3-discovery-mode \
--discovery-band-hz 60 550 \
--target-window-half-width-hz 7.5
```

**Default (no flags):** unchanged global [220, 265] acceptance — **full9 parity**.

**Discovery mode:**

1. Require explicit opt-in flag (fail closed if band/window missing).
2. For each target, accept modes where:
   - `discovery_lo <= f_mode <= discovery_hi`, **and**
   - `|f_mode - target_hz| <= target_window_half_width_hz`, **and**
   - existing physics checks unchanged.
3. Record policy in `result.json` (see §5).

| Pros | Cons |
|------|------|
| Clearest separation from validated timing | Slightly more CLI + manifest fields |
| Cannot accidentally widen full9 | Two code paths to test |
| Matches coarse planner semantics | |

**Recommendation:** **Option C** as the Gate A design, implementing **Option B semantics inside** discovery mode, with **Option A’s band** as `--discovery-band-hz` bounds (not as a silent global replacement).

---

## 4. Preserving existing `full9` validation

### 4.1 Requirements

- Default Stage B behavior **unchanged** when discovery flags absent.
- `--target-set full9` → same targets, same acceptance [220, 265], same `result.json` shape (plus optional new null/false fields).
- M3 m3exec2 / pilot `result.json` files remain valid references.

### 4.2 Design rules

| Rule | Implementation sketch |
|------|------------------------|
| Opt-in only | `--B3-discovery-mode` flag (store_true); default false |
| Default acceptance | `freq_lo=ACCEPTANCE_FREQ_LO_HZ`, `freq_hi=ACCEPTANCE_FREQ_HI_HZ` when flag false |
| No change to target sets | `full9` list untouched |
| Regression gate | Re-run `full9` on m3exec2 checkpoint after implementation; compare to baseline `result.json` with existing parity helpers (`compare_checkpoint_results_to_baseline`, `deduplicate_frequencies_hz`) |
| Orchestrator | Timing orchestrator does **not** pass discovery flags until a separate approved workflow |

### 4.3 Acceptance policy enum (proposed)

```json
"accepted_frequency_policy": "legacy_global_interval"   // default
"accepted_frequency_policy": "discovery_band_and_target_window"  // discovery mode
```

---

## 5. `result.json` in discovery mode (proposed fields)

### 5.1 Top-level (additive)

```json
{
  "discovery_mode": true,
  "discovery_band_hz": [60.0, 550.0],
  "target_window_half_width_hz": 7.5,
  "accepted_frequency_policy": "discovery_band_and_target_window",
  "acceptance_interval_hz": [220.0, 265.0],
  "legacy_acceptance_interval_hz": [220.0, 265.0],
  "targets_hz": [60.0, 75.0, "..."],
  "aggregate": {
    "unique_accepted_frequencies_hz": ["..."],
    "unique_accepted_mode_count": 0,
    "dedupe_tolerance_hz": 0.05
  }
}
```

When `discovery_mode=false` (default):

- `discovery_mode`: false or omitted
- `discovery_band_hz`: null
- `target_window_half_width_hz`: null
- `accepted_frequency_policy`: `"legacy_global_interval"`
- Existing fields unchanged

### 5.2 Per-target rows (additive)

```json
{
  "target_frequency_hz": 75.0,
  "accepted_frequencies_hz": [74.2, 76.1],
  "per_target_acceptance_window_hz": [67.5, 82.5],
  "accepted_mode_count_in_interval": 2,
  "converged_mode_count": 5,
  "discovery_mode": true
}
```

Keep `converged_modes` for diagnostics; acceptance lists drive calibration.

### 5.3 Manifest / checkpoint solve manifest

Mirror discovery block in `checkpoint_solve_manifest.json` for audit (same keys as `result.json` policy section).

---

## 6. Duplicate handling

### 6.1 Recommendation

| Step | Policy |
|------|--------|
| Per-target lists | Raw accepted frequencies per shift (may overlap) |
| Aggregate | `deduplicate_frequencies_hz(all_accepted, tol_hz=FREQ_PARITY_TOL_HZ)` |
| Tolerance | **Keep `FREQ_PARITY_TOL_HZ = 0.05`** unless calibration shows systematic splits |
| Window assignment | After dedupe, assign each unique mode to **one** analysis window (by frequency) |

### 6.2 Overlapping target windows

With `half_width=7.5` and `step=15`, adjacent targets have touching windows — same mode may appear in multiple per-target lists. **Do not** sum per-target counts for density; always dedupe before window binning.

### 6.3 Optional future field

```json
"aggregate": {
  "unique_accepted_frequencies_hz": [...],
  "per_target_accepted_frequencies_hz": { "60.0": [...], "75.0": [...] }
}
```

---

## 7. How the coarse planner consumes discovery results (future)

After Gate A implementation + approved scan:

1. **Input:** `discovery_result.json` (or standard `result.json` with `discovery_mode=true`).
2. **Extract:** `aggregate.unique_accepted_frequencies_hz` (or re-dedupe from per-target rows).
3. **Bin:** partition 60–550 into windows (e.g. 10 Hz); compute `mode_count`, `modes_per_hz`.
4. **Classify zones:** rank windows; set `calibration_status: estimated_from_first_coarse_scan` (thresholds derived, not hard-coded).
5. **Emit:** updated `coarse_frequency_plan.json` with:
   - `regions[].mode_count_observed`
   - `zone_policy_status: estimated_from_first_coarse_scan`
   - `spacing_recommendation` per region
6. **Feed orchestrator:** approved `targets_hz` / new target-set name in JSONL specs.

**M3.4-pre planner change (later, small):** add `--discovery-result-json` to ingest scan output and emit calibration pass — planning stub only until Gate A lands.

---

## 8. Safest first actual scan after Gate A

| Parameter | Value |
|-----------|--------|
| Checkpoint | `st_worker_scaling_L_prod_lhs_pilot_001_timing_m3exec2` |
| Discovery band | 60–550 Hz |
| Target spacing | 15 Hz uniform (~34 targets) |
| Window half-width | 7.5 Hz (starting hypothesis; tune after review) |
| Solver | `mkl_pardiso`, solver-mkl venv, isolated env (M3.3 pattern) |
| Output | **New** dir under `v2_mesh_convergence/diagnostics/solver_benchmarks/` e.g. `discovery_coarse_60_550_15hz_<utc>/` |
| Concurrency | **Exclusive** VM solver slot (~1–4 h estimated) |
| Manifest | **No** orchestrator runtime manifest / index unless explicitly approved later |
| Entry script | `v2_b3_checkpoint_solve.py --targets-hz <list> --B3-discovery-mode ...` **or** extended density experiment |

### 8.1 Pre-scan checklist

- [ ] Gate A code merged + `py_compile` / unit smoke on VM
- [ ] `full9` regression on m3exec2 matches baseline (`compare_checkpoint_results_to_baseline`)
- [ ] Dry-run planner shows `executable_now_count=34`, `blocked_count=0` under discovery policy
- [ ] New output path confirmed absent (no overwrite)
- [ ] User approval recorded

### 8.2 Smaller interim alternative (if Gate A delayed)

Execute **220–265 Hz only** with **current** acceptance (no code change) — validates tooling and timing, **does not** answer 60–550 structure. Not a substitute for Gate A.

---

## 9. Proposed code changes (describe only — not implemented)

**Scope estimate:** small–medium, localized; **do not implement until approved.**

### 9.1 `v2_b3_st_sinvert_solver_lib.py`

- Add `AcceptancePolicy` helper or kwargs to `collect_accepted_st_modes`:
  - `policy="legacy_global_interval"` → current behavior
  - `policy="discovery_band_and_target_window"` → band + `|f - target_hz| <= half_width`
- Thread optional `freq_lo`, `freq_hi`, `target_window_half_width_hz`, `policy` through `run_checkpoint_st_target`.

### 9.2 `v2_b3_checkpoint_solver_multi_benchmark.py`

- CLI: `--B3-discovery-mode`, `--discovery-band-hz LO HI`, `--target-window-half-width-hz`
- Validate: discovery flag requires band + half-width; band must satisfy `lo < hi`.
- Write policy block to `result.json`.

### 9.3 `v2_b3_checkpoint_solve.py`

- Forward discovery flags to multi-benchmark argv builder (`build_checkpoint_multi_benchmark_argv` in pipeline lib).

### 9.4 `v2_b3_frequency_coarse_planner.py` (reporting only — partial)

- ✅ Add `executable_now_count` / `blocked_count` under `executable_feasibility` (done in M3.4-pre schema fix).
- Later: when `--discovery-policy assumed` dry-run flag, set counts as if discovery enabled.

### 9.5 Tests / regression

- Fixture or VM smoke: `full9` without discovery → bit-identical acceptance lists vs baseline JSON.
- Discovery smoke: single target at 75 Hz returns accepted mode near 75 Hz when in band.

**Files explicitly not changed in Gate A unless needed:** Stage A export, orchestrator, rich modal, synthesis.

---

## 10. Decision points for review

| # | Question | Suggested default |
|---|----------|-------------------|
| 1 | Implement Option C? | **Yes** |
| 2 | Default `target_window_half_width_hz` in discovery? | **7.5** (half of 15 Hz step) |
| 3 | Record converged-but-rejected modes in discovery? | Optional `converged_modes` already present; add summary counts only |
| 4 | Extend density experiment or only checkpoint_solve? | Both via shared `run_checkpoint_st_target` |
| 5 | First scan 60–550 vs narrow slice? | **60–550 after Gate A**; narrow slice only if Gate A slips |

---

## 11. References

| Artifact | Path / note |
|----------|-------------|
| Acceptance constants | `scripts/v2_b3_st_sinvert_solver_lib.py` |
| Stage B entry | `scripts/v2_b3_checkpoint_solve.py` |
| Multi-target runner | `scripts/v2_b3_checkpoint_solver_multi_benchmark.py` |
| Density experiment | `scripts/v2_b3_checkpoint_target_density_experiment.py` |
| M3.4-pre planner | `scripts/v2_b3_frequency_coarse_planner.py` |
| Validated baseline | m3exec2 `result.json` (full9) |

---

*Gate A planning draft — no execution authorized.*
