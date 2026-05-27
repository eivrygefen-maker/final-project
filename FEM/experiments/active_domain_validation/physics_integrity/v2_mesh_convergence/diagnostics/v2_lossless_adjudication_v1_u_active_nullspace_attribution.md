# Lossless adjudication v1: u_active nullspace attribution

Generated: 2026-05-27T09:10:17Z

Classification: `UNRESOLVED_U_ACTIVE_NULLSPACE`
Subtype: ``

## Guard
- no_new_eigensolve_executed=True
- eps_run_count_for_this_lane=1
- re_invoking_authorized_runner_would_block_eps=True

## Evidence (aggregate support fractions)
- shell_tag_union_top_back_ribs: median_l2_fraction=3.671068e-01 (min=5.036431e-03, max=9.999999e-01)
- tag_1_top_shell_displacement: median_l2_fraction=1.376947e-04 (min=2.240165e-08, max=2.400683e-01)
- tag_3_back_shell_displacement: median_l2_fraction=4.951797e-02 (min=1.371183e-05, max=4.146649e-01)
- tag_4_ribs_side_displacement: median_l2_fraction=1.339847e-01 (min=6.195940e-05, max=9.999999e-01)
- tag_5_pinned_fix_displacement: median_l2_fraction=0.000000e+00 (min=0.000000e+00, max=0.000000e+00)
- u_non_shell_displacement_complement: median_l2_fraction=9.301478e-01 (min=5.247644e-04, max=9.999873e-01)

## Seed control (same layout/operators)
- seed ||Mx||=6.835493617660683e-13
- seed xH_Mx=5.18357532956158e-13
- seed Rayleigh frequency_hz=243.07541711755843

## Threshold audit
- in_or_near_null_M_abs_norm_thresh=1e-12
- seed ||Mx||=6.835493617660683e-13 -> in_or_near_null_M=True

## Remediation design options (no EPS)
- recommended_option_subtype=

### Project out identified M-null u_active DOFs before EPS
- Expected ability to recover p_active branch: High if null subspace is correctly identified and projection is aligned with replay operators.
- Implementation risk: Requires a reliable mapping from identified DOF subsets to reduced eigenproblem basis; risk of removing true physics mass-bearing components.

### Construct EPS on the physical mass-bearing reduced subspace
- Expected ability to recover p_active branch: High if the basis spans the acoustic branch and is consistent with replay maps/operators.
- Implementation risk: Basis construction is nontrivial; must preserve coupled u/p layout and BC reductions.

### Explicit null-space deflation / constrained generalized eigenproblem
- Expected ability to recover p_active branch: Medium-to-high, depending on deflation quality and stability for SINVERT.
- Implementation risk: Solver-level change; requires validation for singular generalized eigenproblems and consistent backtransforms.

### Singular generalized eigenproblem method (robust for rank-deficient M)
- Expected ability to recover p_active branch: Medium; provides correct spectral information but might still require subspace constraints.
- Implementation risk: More invasive solver configuration and backtransformation semantics.

### PGNHEP / purification only if it is justified by nullspace attribution
- Expected ability to recover p_active branch: Unknown until validated; might help if purification targets the detected null subspace.
- Implementation risk: May alter modeled operator content; must validate with no-EPS replay checks first.

### ST regularization only as a diagnostic (not a final physical verdict path)
- Expected ability to recover p_active branch: Low-to-medium; expected to shift the solver away from null mass but must be validated.
- Implementation risk: Could mask the underlying targeting issue; risk of false confidence.

