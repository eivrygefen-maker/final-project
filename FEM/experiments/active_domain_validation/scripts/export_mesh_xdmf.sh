#!/usr/bin/env bash
# Optional ParaView path: convert experiment mesh to XDMF via meshio.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
EXP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
python3 - <<PY
from pathlib import Path
import meshio
msh = Path("$EXP_ROOT/mesh/validation_tiny_guitar_3d.msh")
out = Path("$EXP_ROOT/mesh/validation_tiny_guitar_3d.xdmf")
mesh = meshio.read(str(msh))
meshio.write(str(out), mesh)
print("Wrote", out)
PY
