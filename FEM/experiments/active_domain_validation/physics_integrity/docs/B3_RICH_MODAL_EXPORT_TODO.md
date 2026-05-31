# TODO: Rich modal export before audio / STK / microphone synthesis

**Status:** planning / not implemented  
**Default:** disabled (`--B3-export-rich-modal-data` is opt-in only)

## When this matters

Before any **expensive LHS** or **wide frequency sweep** whose results will feed:

- audio synthesis,
- STK modal resonator chains,
- microphone / listener response, or
- body-observation post-processing,

verify that runs export **rich modal data**, not just accepted frequencies and timing summaries.

The solver-mkl checkpoint pipeline (`v2_b3_checkpoint_export.py`, `v2_b3_checkpoint_solve.py`) currently optimizes **operator reuse and ST/EPS timing**. It does **not** by default export synthesis-ready mode shapes.

**Solver benchmarks must keep rich export disabled** unless explicitly testing export itself.

## Required checklist (verify before large sweeps)

| Item | Question to answer |
|------|------------------|
| **Eigenvectors / mode shapes** | Are full displacement + pressure mode columns saved (not only Hz)? |
| **Mode normalization convention** | Documented L2 / mass / GNHEP undo / block scaling used by SLEPc export? |
| **Excitation coupling** | Bridge / string input DOFs or participation vectors for pluck or drive? |
| **Output coupling** | Microphone, listener, or body observation locations and pickoff matrices? |
| **DOF mapping metadata** | `active_local`, `u_idx`, `p_idx`, `free_rows`, `bc_rows`, mesh tags reproducible? |
| **Per-mode material / plate participation** | Enough data for future damping / Q assignment per mode and region? |

## Current production FOM reference (partial)

Production coupled runs via `fem_main_3d.py` may write:

- `FEM/outputs/modes_3d/coupled_modes_raw.npz` — eigenvector columns + layout
- XDMF mode shapes for visualization

These are **not** automatically produced by checkpoint solver benchmarks. Confirm layout, normalization, and coupling metadata before reusing checkpoint-only results for synthesis.

## Planned opt-in flag

```text
--B3-export-rich-modal-data
```

- **Default:** off everywhere (including solver benchmarks)
- **When enabled (future):** attach rich modal artifacts to export/solve output directories
- **Today:** requesting the flag fails with a clear message pointing to this document

## Implementation notes (future)

1. Hook after `EPSSolve` in synthesis-oriented paths (not default benchmark path).
2. Persist normalization convention in JSON sidecar next to mode arrays.
3. Reuse `built_metadata.json` index maps; extend with bridge/mic pickoff indices.
4. Keep checkpoint timing benchmarks on frequencies + summary only unless flag explicitly set.

## Related docs

- [environment/solver-mkl/README.md](../environment/solver-mkl/README.md) — two-stage solver pipeline
- [v2_mesh_convergence/diagnostics/solver_benchmarks/README.md](../v2_mesh_convergence/diagnostics/solver_benchmarks/README.md) — timing benchmarks
