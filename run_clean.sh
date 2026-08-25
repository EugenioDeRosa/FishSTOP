#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
ENTRYPOINT="${PROJECT_ROOT}/streamlit_app.py"

find_python311() {
    local candidate
    for candidate in \
        "${FISHSTOP_PYTHON:-}" \
        "${HOME}/.local/bin/python3.11" \
        "/opt/homebrew/bin/python3.11" \
        "/usr/local/bin/python3.11" \
        "python3.11"; do
        if [[ -n "${candidate}" ]] && command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ ! -f "${ENTRYPOINT}" ]]; then
    echo "FishSTOP entry point not found: ${ENTRYPOINT}" >&2
    exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
    if ! PYTHON311="$(find_python311)"; then
        cat >&2 <<'EOF'
Python 3.11 is required but was not found.

Install it with Homebrew:
  brew install python@3.11

If Homebrew does not provide a package for your macOS version, install uv:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv python install 3.11

Then run this script again. If Python 3.11 is installed in a custom location:
  FISHSTOP_PYTHON=/path/to/python3.11 ./run_clean.sh
EOF
        exit 1
    fi

    echo "Creating virtual environment with ${PYTHON311} ..."
    "${PYTHON311}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_PYTHON}" -c 'import streamlit' >/dev/null 2>&1; then
    echo "Installing FishSTOP dependencies (first run may take several minutes) ..."
    "${VENV_PYTHON}" -m pip install --upgrade pip
    "${VENV_PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

echo "Stopping old FishSTOP Streamlit processes ..."
while IFS= read -r process_line; do
    process_id="${process_line%% *}"
    process_command="${process_line#* }"
    if [[ "${process_id}" != "$$" ]] \
        && [[ "${process_command}" == *"${PROJECT_ROOT}"* ]] \
        && { [[ "${process_command}" == *"streamlit"* ]] \
            || [[ "${process_command}" == *"streamlit_app.py"* ]] \
            || [[ "${process_command}" == *"src/app.py"* ]]; }; then
        kill "${process_id}" 2>/dev/null || true
    fi
done < <(ps -Ao pid=,command= | sed -E 's/^[[:space:]]+//')

echo "Removing project Python caches ..."
find "${PROJECT_ROOT}" \
    -path "${VENV_DIR}" -prune -o \
    -type d -name __pycache__ -exec rm -rf {} +

cd "${PROJECT_ROOT}"
echo "Starting FishSTOP at http://localhost:8501 ..."
exec "${VENV_PYTHON}" -m streamlit run "${ENTRYPOINT}"
