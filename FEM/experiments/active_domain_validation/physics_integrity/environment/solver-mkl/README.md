# Two-stage L_prod solver pipeline (production export → solver-mkl solve)

Validated MKL PARDISO full9 (~736–760 s) vs legacy production path (~112 min).

**Production defaults are unchanged.** This pipeline is dev/benchmark only until explicitly promoted.

## Rich modal export (before LHS / audio synthesis)

Checkpoint export/solve optimizes **operator reuse and ST/EPS timing**. It does **not** by default
export synthesis-ready mode shapes (eigenvectors, bridge/mic coupling, DOF maps, per-mode Q data).

Before expensive LHS or wide sweeps intended for audio, STK, or microphone post-processing, read
`docs/B3_RICH_MODAL_EXPORT_TODO.md` and verify the required checklist.

Optional flag (disabled by default; **opt-in for designated synthesis runs**):

```text
--B3-export-rich-modal-data
```

- **Stage A** always writes `synthesis_metadata.json` + `checkpoint_export_manifest.json` on successful export. Region DOF locate is **off by default** (`--B3-synthesis-region-dofs off`); use `best_effort` only when needed.
- **Stage B** writes `rich_modal/` active eigenvectors only when the flag is set.
- **Stage C:** `v2_b3_rich_modal_post.py` (production `.venv`) for region participation and audio output proxies.

Do **not** pass this flag for solver timing benchmarks.

See `docs/B3_RICH_MODAL_EXPORT_TODO.md`.

## Stage A — Production environment (export only)

**Requires:** project production `.venv`, DOLFINx, system PETSc. **Do not** activate `~/solver-mkl/venv`.

```bash
source ~/final-project/.venv/bin/activate
cd ~/final-project

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py \
  --mesh-level L_prod \
  --B3-block-compose-backend csr_bulk
```

Optional explicit output directory:

```bash
export CKPT="FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_$(date -u +%Y%m%dT%H%M%SZ)"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py \
  --mesh-level L_prod \
  --B3-block-compose-backend csr_bulk \
  --B3-synthesis-region-dofs off \
  --output-dir "$CKPT"
```

**Writes per checkpoint directory:**

| File | Purpose |
|------|---------|
| `A_active.petsc.bin` | PETSc binary |
| `M_active.petsc.bin` | PETSc binary |
| `A_active_csr.npz` | Portable CSR fallback |
| `M_active_csr.npz` | Portable CSR fallback |
| `csr_metadata.json` | Shape/nnz/norm metadata |
| `built_metadata.json` | Active index maps for acceptance |
| `synthesis_metadata.json` | Mesh, tags, GNHEP/solver physics metadata (v1) |
| `region_dof_indices.npz` | Optional (`--B3-synthesis-region-dofs best_effort`); default deferred to Stage C |
| `checkpoint_export_manifest.json` | Export status + verification |

No ST/EPS solve in this stage.

## Stage B — solver-mkl environment (load + solve)

**Requires:** `source ~/solver-mkl/activate_solver_mkl.sh`, `mkl_pardiso` available. **No** DOLFINx/FEM assembly.

```bash
source ~/solver-mkl/activate_solver_mkl.sh
cd ~/final-project

export CKPT="FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_20260531T083626Z"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve.py \
  --checkpoint-dir "$CKPT" \
  --factor-solver mkl_pardiso \
  --target-set full9 \
  --B3-export-rich-modal-data
```

**Writes:** auto-created directory under `solver_benchmarks/checkpoint_solve_<factor>_<set>_<utc>/` with `result.json`, `result.md`, `checkpoint_solve_manifest.json`, and (with flag) `rich_modal/modes_active.npz`.

### Stage C — rich modal post (production `.venv`)

```bash
source ~/final-project/.venv/bin/activate
cd ~/final-project

export SOLVE_OUT=".../checkpoint_solve_mkl_pardiso_full9_<utc>"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_rich_modal_post.py \
  --checkpoint-dir "$CKPT" \
  --rich-modal-dir "$SOLVE_OUT/rich_modal"
```

**Writes:** `$SOLVE_OUT/rich_modal_post/modes_synthesis.json` (region participation + audio output proxies; not microphone pressure).

Optional explicit output directory:

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve.py \
  --checkpoint-dir "$CKPT" \
  --factor-solver mkl_pardiso \
  --target-set full9 \
  --output-dir "$OUT"
```

### MUMPS fallback (solver-mkl env)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve.py \
  --checkpoint-dir "$CKPT" \
  --factor-solver mumps \
  --target-set full9
```

## Safety checks (both stages)

| Check | Stage A | Stage B |
|-------|---------|---------|
| Correct venv | production `.venv`, not solver-mkl | `~/solver-mkl/venv` |
| DOLFINx | required | must not be required |
| Factor solver probe | n/a | `mkl_pardiso` (or `mumps` if selected) |
| Checkpoint files | written + verified | must exist |
| Matrix load smoke | after export | before solve |

Failures exit with clear messages (wrong venv, incomplete checkpoint, probe failure).

## Compare runs

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solver_compare.py \
  --baseline "$RUN1/result.json" \
  --candidate "$RUN2/result.json" \
  --output-json "$RUN2/compare.json"
```

## Cleanup dry-run (no deletions)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_diagnostics_cleanup_dry_run.py
```

Produces `solver_benchmarks/cleanup_dry_run_<utc>.json` listing keep vs review candidates. **Does not delete anything.**

## Canonical validated artifacts (keep)

- Checkpoint: `st_worker_scaling_L_prod_20260531T083626Z`
- MKL full9 runs under `solver_benchmarks/checkpoint_multi_mkl_pardiso_full9_*`
