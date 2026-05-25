# Coupled physical core v2 — formulation report

Experiment-only branch `coupled_physical_core_v2` supersedes the v1 scaled coupled path
(`fsi_coupling_gain=1e6`, `physical_fsi_alpha`, Nitsche tuning). v1 diagnostics remain
historical evidence only.

## Strong forms (linearized, frequency domain)

**Structure (displacement `u`, test `v`):**

\[
-\nabla\cdot\sigma(u) = \omega^2 \rho_s u \quad \text{on plates}
\]

**Acoustic cavity (pressure `p`, test `q`):**

\[
-\frac{1}{\rho_{\mathrm{air}}}\nabla^2 p = \omega^2 \frac{1}{\rho_{\mathrm{air}} c^2} p
\quad \text{in air volume}
\]

**Interface \(\Gamma\) (outward normal `n` from air into structure on the meshtags seam):**

\[
\sigma(u)\,n = -p\,n \quad \text{(pressure traction on structure)}
\]
\[
\frac{\partial p}{\partial n} = -\rho_{\mathrm{air}}\,\omega^2\, u\cdot n
\quad \text{(normal acceleration forcing in acoustic equation)}
\]

## Weak forms (Galerkin blocks)

With interface measure \(d\Gamma\) on validated air↔wood facets (tag 20, `meshtags_ds`):

| Block | Bilinear form | Matrix slot | Units (SI) |
|-------|---------------|-------------|------------|
| `A_up` | \(-\int_\Gamma p\,(n\cdot v)\,d\Gamma\) | stiffness, u←p | N (traction work) |
| `A_pu` | \(\int_\Gamma (u\cdot n)\,q\,d\Gamma\) | stiffness, p←u | m³/s (flux test) |
| `M_pu` | \(\rho_{\mathrm{air}}\int_\Gamma (u\cdot n)\,q\,d\Gamma\) | mass, p←u | kg/(m²·s) at ω² |

Diagonal blocks (unchanged from passing integrity cases):

| Block | Form | Units |
|-------|------|-------|
| `A_uu` | orthotropic shell stiffness on wood facets | N/m |
| `M_uu` | \(\rho_s t \int u\cdot v\,dS\) | kg |
| `A_pp` | \((s^2/\rho_{\mathrm{air}})\int \nabla p\cdot\nabla q\,d\Omega\) | — |
| `M_pp` | \((s^2/(\rho_{\mathrm{air}} c^2))\int p\,q\,d\Omega\) | — |

Here \(s=\) `pressure_dof_scale` is **numerical similarity only** on pressure DOFs; it is **not**
applied to v2 coupling UFL. Physical reporting undoes \(s\) via GNHEP metadata (`s_pp`, etc.).

## Sign convention

- `FacetNormal` on the assembled seam follows DOLFINx/mesh orientation (outward from air subdomain on tag-20 facets).
- `A_up` enters the structural test equation with **minus** sign (pressure pushes structure along \(-n\) when \(p>0\)).
- `A_pu` and `M_pu` enter the pressure test equation with **plus** sign (normal displacement drives acoustic flux / acceleration).

## Reciprocity / sign sanity

For symmetric structure and consistent normals, the discrete coupling should satisfy
\(\langle A_{up} p, u\rangle \approx \langle A_{pu} u, p\rangle\) on interface-supported modes.
The validation runner reports `reciprocity_ratio = |p^T A_{pu} u| / |u^T A_{up} p|` for unit interface patterns.

## Explicit exclusions (v2)

- No `fsi_coupling_gain` amplification (forced to `1.0` in code path).
- No `physical_fsi_alpha` continuation scaling.
- No Nitsche interface terms in the initial v2 core.
- No v1 branch-continuation or harvest `p_frac` ranking.

## Numerical scaling (reversible)

Optional `gnhep_block_frobenius_normalize` scales assembled blocks for SLEPc conditioning.
Eigenvalues and energy participation in reports use **physical undo** of `s_uu`, `s_pp`, `s_couple`.
