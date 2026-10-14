#!/usr/bin/env bash
# Start one runner stage on the VM(s), in tmux so SSH drops don't kill it. Writes
# machine-<stage>-<instance>.json first (via cloud/remote_stage.sh on the VM).
#
#   cloud/run_stage.sh pilot|224|448 [arm64|x86_64|all]
#
# Stages: pilot (matrices/phase2-pilot.json), then 224 and 448 (matrices/phase2.json
# --stage input=N, iteration caps from the pilot). Fetch and review between stages.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

stage=${1:?usage: cloud/run_stage.sh pilot|224|448 [arm64|x86_64|all]}
case $stage in pilot | 224 | 448) ;; *) die "stage must be pilot, 224 or 448" ;; esac
TARGETS=$(targets "${2:-all}")
require_aws

for arch in $TARGETS; do
    ip=$(instance_ip "$arch")
    log "$arch ($ip): starting stage $stage"
    vm_ssh "$ip" "bash $REMOTE_DIR/cloud/remote_stage.sh $stage"
done
log "watch: ssh -i $KEY_FILE $SSH_USER@<ip> then 'tmux attach -t odb' (detach: Ctrl-b d)"
