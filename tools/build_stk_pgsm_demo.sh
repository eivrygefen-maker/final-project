#!/usr/bin/env bash
# Build PGSM STK guitar demo on VM. Requires STK_ROOT or discoverable STK install.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build"

echo "== PGSM STK demo build =="
echo "repo: ${REPO_ROOT}"

if [[ -z "${STK_ROOT:-}" ]]; then
  if [[ -d "${HOME}/stk/include" ]]; then
    export STK_ROOT="${HOME}/stk"
  elif [[ -d "/home/vboxuser/stk/include" ]]; then
    export STK_ROOT="/home/vboxuser/stk"
  fi
fi

if [[ -n "${STK_ROOT:-}" ]]; then
  echo "STK_ROOT=${STK_ROOT}"
else
  echo "STK_ROOT not set — CMake will search common paths."
fi

python3 "${REPO_ROOT}/tools/probe_stk_environment.py" || true

mkdir -p "${BUILD_DIR}"
cmake -S "${REPO_ROOT}/cpp/stk_pgsm_guitar_demo" -B "${BUILD_DIR}" -DSTK_ROOT="${STK_ROOT:-}"
cmake --build "${BUILD_DIR}" --config Release -j"$(nproc 2>/dev/null || echo 2)"

echo "Built: ${BUILD_DIR}/stk_pgsm_guitar_demo"
echo "Run: ${REPO_ROOT}/tools/run_stk_pgsm_demo.sh"
