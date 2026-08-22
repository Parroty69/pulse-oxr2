#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
    echo "Missing ${python_bin}. Run the setup steps in README.md first." >&2
    exit 1
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"

exec "${python_bin}" -m streamlit run "${repo_root}/src/ui/app.py" "$@"
