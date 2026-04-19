#!/usr/bin/env bash
# run_upload_tae.sh — Launch (or re-attach to) a tmux session for TAE batch upload.
#
# Usage:
#   bash run_upload_tae.sh          # start or resume the session
#   tmux attach -t tae_upload       # re-attach manually
#   tmux kill-session -t tae_upload # stop
#
# The Python script auto-resumes from the last completed page on restart.

set -euo pipefail

SESSION="tae_upload"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/cad_pipeline/.venv/bin/python"
LOGFILE="${SCRIPT_DIR}/cad_pipeline/.checkpoints/upload_tae.log"

mkdir -p "${SCRIPT_DIR}/cad_pipeline/.checkpoints"

# ── Validate venv ─────────────────────────────────────────────────────────────
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "ERROR: venv not found at ${VENV_PYTHON}"
    echo "Run: cd ${SCRIPT_DIR}/cad_pipeline && uv sync"
    exit 1
fi

# ── Build the command to run inside tmux ─────────────────────────────────────
RUN_CMD="cd '${SCRIPT_DIR}' && '${VENV_PYTHON}' batch_upload_tae.py 2>&1 | tee -a '${LOGFILE}'"

# ── Check if session already exists ──────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists."
    echo "  → Attaching (Ctrl-B D to detach without stopping)"
    tmux attach -t "${SESSION}"
    exit 0
fi

echo "Starting new tmux session '${SESSION}'..."
echo "Python: ${VENV_PYTHON}"
echo "Log:    ${LOGFILE}"
echo ""
echo "To re-attach later:  tmux attach -t ${SESSION}"
echo "To stop:             tmux kill-session -t ${SESSION}"
echo ""

# ── Create session and run ────────────────────────────────────────────────────
tmux new-session -d -s "${SESSION}" -x 220 -y 50
tmux send-keys -t "${SESSION}" "${RUN_CMD}" Enter

# Attach so the user sees the output immediately
tmux attach -t "${SESSION}"
