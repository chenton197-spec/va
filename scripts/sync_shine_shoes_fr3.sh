#!/usr/bin/env bash
# Sync shine_shoes_fr3 dataset from remote teleop machine (password SSH).
#
# Usage:
#   bash scripts/sync_shine_shoes_fr3.sh
#   REMOTE_PASS=xxx bash scripts/sync_shine_shoes_fr3.sh
#
# Env overrides: REMOTE_USER REMOTE_HOST REMOTE_PASS REMOTE_PATH LOCAL_PATH
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-casbot}"
REMOTE_HOST="${REMOTE_HOST:-10.42.0.2}"
REMOTE_PASS="${REMOTE_PASS:-123456}"
REMOTE_PATH="${REMOTE_PATH:-/home/casbot/teleop_project/datasets/shine_shoes_fr3/}"
LOCAL_PATH="${LOCAL_PATH:-/home/casbotskill/ct/va/shine_shoes_fr3/}"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no"

mkdir -p "${LOCAL_PATH}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync not found. Install with: sudo apt install rsync" >&2
  exit 1
fi

echo "Syncing:"
echo "  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "  -> ${LOCAL_PATH}"
echo

run_with_sshpass() {
  sshpass -p "${REMOTE_PASS}" rsync -avh --progress \
    -e "ssh ${SSH_OPTS}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
    "${LOCAL_PATH}"
}

run_with_pexpect() {
  # Prefer project/lerobot python that has pexpect; fall back to python3.
  local py=""
  for cand in \
    "${SYNC_PYTHON:-}" \
    /home/casbotskill/miniconda3/envs/lerobot/bin/python \
    python3 \
    python
  do
    [[ -z "${cand}" ]] && continue
    if command -v "${cand}" >/dev/null 2>&1 || [[ -x "${cand}" ]]; then
      if "${cand}" -c "import pexpect" >/dev/null 2>&1; then
        py="${cand}"
        break
      fi
    fi
  done
  if [[ -z "${py}" ]]; then
    echo "Error: need sshpass or Python pexpect." >&2
    echo "  sudo apt install sshpass" >&2
    echo "  # or: pip install pexpect" >&2
    exit 1
  fi

  REMOTE_USER="${REMOTE_USER}" \
  REMOTE_HOST="${REMOTE_HOST}" \
  REMOTE_PASS="${REMOTE_PASS}" \
  REMOTE_PATH="${REMOTE_PATH}" \
  LOCAL_PATH="${LOCAL_PATH}" \
  SSH_OPTS="${SSH_OPTS}" \
  "${py}" - <<'PY'
import os
import sys

import pexpect

user = os.environ["REMOTE_USER"]
host = os.environ["REMOTE_HOST"]
password = os.environ["REMOTE_PASS"]
remote_path = os.environ["REMOTE_PATH"]
local_path = os.environ["LOCAL_PATH"]
ssh_opts = os.environ["SSH_OPTS"]

cmd = (
    f'rsync -avh --progress -e "ssh {ssh_opts}" '
    f"{user}@{host}:{remote_path} {local_path}"
)
child = pexpect.spawn("/bin/bash", ["-lc", cmd], encoding="utf-8", timeout=None)
child.logfile_read = sys.stdout

while True:
    idx = child.expect(
        [
            pexpect.EOF,
            r"(?i)password:\s*",
            r"(?i)are you sure you want to continue connecting.*\?",
            pexpect.TIMEOUT,
        ],
        timeout=60,
    )
    if idx == 0:
        break
    if idx == 1:
        child.sendline(password)
        continue
    if idx == 2:
        child.sendline("yes")
        continue
    # TIMEOUT: keep waiting while rsync transfers
    continue

child.close()
sys.exit(child.exitstatus if child.exitstatus is not None else 1)
PY
}

if command -v sshpass >/dev/null 2>&1; then
  run_with_sshpass
else
  echo "sshpass not found; using pexpect fallback..."
  run_with_pexpect
fi

echo
echo "Done."
