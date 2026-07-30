#!/usr/bin/env bash
# Sync shine_shoes_fr3 dataset from remote teleop machine.
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-casbot}"
REMOTE_HOST="${REMOTE_HOST:-10.42.0.2}"
REMOTE_PATH="${REMOTE_PATH:-/home/casbot/teleop_project/datasets/shine_shoes_fr3/}"
LOCAL_PATH="${LOCAL_PATH:-/home/casbotskill/ct/va/shine_shoes_fr3/}"
SSH_PASS="${SSH_PASS:-123456}"

mkdir -p "$LOCAL_PATH"

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync not found. Install with: sudo apt install rsync" >&2
  exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no"
SRC="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

echo "==> Syncing"
echo "    ${SRC}"
echo " -> ${LOCAL_PATH}"
echo

if command -v sshpass >/dev/null 2>&1; then
  sshpass -p "$SSH_PASS" rsync -avh --progress \
    -e "ssh ${SSH_OPTS}" \
    "$SRC" \
    "$LOCAL_PATH"
elif command -v expect >/dev/null 2>&1; then
  export SSH_PASS SRC LOCAL_PATH SSH_OPTS
  expect <<'EOF'
set timeout -1
spawn rsync -avh --progress -e "ssh $::env(SSH_OPTS)" $::env(SRC) $::env(LOCAL_PATH)
expect {
  -re "(?i)password:" {
    send "$::env(SSH_PASS)\r"
    exp_continue
  }
  eof
}
catch wait result
exit [lindex $result 3]
EOF
else
  ASKPASS="$(mktemp)"
  trap 'rm -f "$ASKPASS"' EXIT
  cat >"$ASKPASS" <<EOF
#!/usr/bin/env bash
printf '%s\n' '$SSH_PASS'
EOF
  chmod 700 "$ASKPASS"
  DISPLAY="${DISPLAY:-:0}" \
  SSH_ASKPASS="$ASKPASS" \
  SSH_ASKPASS_REQUIRE=force \
    setsid -w rsync -avh --progress \
      -e "ssh ${SSH_OPTS}" \
      "$SRC" \
      "$LOCAL_PATH"
fi

echo
echo "==> Done."
