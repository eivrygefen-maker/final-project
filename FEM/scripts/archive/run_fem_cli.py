# FEM/scripts/run_fem_cli.py

from fem_api import GuitarGeometry, MaterialProps, run_guitar_fem

# ask user for basic geometry (meters)
L = float(input("Body length [m] (e.g. 0.48): "))
W = float(input("Body width  [m] (e.g. 0.36): "))
D = float(input("Body depth  [m] (e.g. 0.11): "))
T = float(input("Top thickness [m] (e.g. 0.003): "))

# ask material (can use defaults for now)
E   = float(input("Young's modulus [Pa] (e.g. 1.0e10): "))
rho = float(input("Density [kg/m^3] (e.g. 450): "))
nu  = float(input("Poisson's ratio [-] (e.g. 0.3): "))

geom = GuitarGeometry(length=L, width=W, depth=D, top_thickness=T)
mat  = MaterialProps(young=E, density=rho, poisson=nu)

res = run_guitar_fem(geom, mat)

print("\n=== FEM/physics result (placeholder for now) ===")
print(f"Main body fundamental f0: {res.f0:.2f} Hz")
print("Modes:", ", ".join(f"{f:.2f}" for f in res.modes))
