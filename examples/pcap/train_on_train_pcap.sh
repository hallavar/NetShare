#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NETSHARE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${NETSHARE_DIR}/../.." && pwd)"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/benchmark_programs/.cache/matplotlib}"

NETSHARE_VENV="${NETSHARE_VENV:-${REPO_ROOT}/benchmark_programs/envs/netshare}"
if [[ -x "${NETSHARE_VENV}/bin/python" ]] && "${NETSHARE_VENV}/bin/python" -c "pass" >/dev/null 2>&1; then
  PYTHON_BIN="${NETSHARE_VENV}/bin/python"
else
  echo "[NetShare] Skipping unusable venv: ${NETSHARE_VENV}" >&2
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

cd "${NETSHARE_DIR}"
"${PYTHON_BIN}" examples/pcap/driver.py --skip-visualize "$@"
