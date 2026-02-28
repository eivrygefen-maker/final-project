# FEM/scripts/bar_modes.py
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# ------------------------------------------------------------
# Simple 1D bar eigenmodes (robust sanity test)
# K u = lambda M u, fixed at left end (u(0)=0)
# ------------------------------------------------------------

# Geometry / discretization
L = 1.0          # bar length [m]
n_elem = 10      # number of elements
n_nodes = n_elem + 1
x = np.linspace(0.0, L, n_nodes)

# Material / section (choose any reasonable values)
E = 2.0e11       # Young's modulus [Pa]
rho = 7800.0     # density [kg/m^3]
A = 1.0          # cross-sectional area [m^2] (arbitrary for this test)

# Element length (uniform mesh)
Le = L / n_elem

# Local element matrices for linear 2-node bar element
Ke_local = (E * A / Le) * np.array([[1.0, -1.0],
                                   [-1.0, 1.0]])

Me_local = (rho * A * Le / 6.0) * np.array([[2.0, 1.0],
                                           [1.0, 2.0]])

# Assemble global matrices
K = lil_matrix((n_nodes, n_nodes), dtype=float)
M = lil_matrix((n_nodes, n_nodes), dtype=float)

for e in range(n_elem):
    n1 = e
    n2 = e + 1
    dofs = [n1, n2]

    for a in range(2):
        for b in range(2):
            K[dofs[a], dofs[b]] += Ke_local[a, b]
            M[dofs[a], dofs[b]] += Me_local[a, b]

K = csr_matrix(K)
M = csr_matrix(M)

# Apply essential BC: u(0)=0  -> remove DOF 0 (fixed)
free = np.arange(1, n_nodes)   # dofs 1..end
K_ff = K[free][:, free]
M_ff = M[free][:, free]

# Solve for a few smallest eigenvalues
n_modes = 5
# 'SM' = smallest magnitude (good for lowest frequencies)
vals, vecs = eigsh(K_ff, k=n_modes, M=M_ff, which='SM')

# Sort results
idx = np.argsort(vals)
vals = vals[idx]
vecs = vecs[:, idx]

print(f"Bar eigenmodes sanity test (fixed-free), n_elem={n_elem}")
print(f"E={E:.3e} Pa, rho={rho:.1f} kg/m^3, A={A:.2f} m^2, L={L:.2f} m")
print(f"First {n_modes} eigenfrequencies (Hz):")

for i, lam in enumerate(vals, start=1):
    omega = np.sqrt(lam)               # rad/s
    freq = omega / (2.0 * np.pi)       # Hz
    print(f"mode {i}: {freq:.2f} Hz")
