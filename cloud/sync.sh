#!/usr/bin/env bash
# Copy the repo and the exact model files to the VM(s), then verify the model sha256s
# on the remote side against cloud/models.sha256 (the Mac's files).
#
#   cloud/sync.sh [arm64|x86_64|all]
#
# Code is mirrored (rsync --delete). results/ is never mirrored: the VM owns its
# results. The only results ever pushed are results/phase2/<arch>/, and only to a VM
# that has none yet (a replacement instance resuming a stage).
set -euo pipefail
source "$(dirname "$0")/lib.sh"

TARGETS=$(targets "${1:-all}")
require_aws
cd "$ROOT"

log "local models vs cloud/models.sha256"
(cd models && shasum -a 256 -c ../cloud/models.sha256) || die "local model files differ from cloud/models.sha256"
[ -z "$(git status --porcelain --untracked-files=no)" ] ||
    echo "WARNING: uncommitted changes; machine-*.json will record the commit as dirty" >&2

for arch in $TARGETS; do
    ip=$(instance_ip "$arch")
    log "$arch ($ip): code + models"
    rsync -az --delete -e "ssh $SSH_OPTS" \
        --exclude .venv/ --exclude results/ --exclude cloud/.state/ \
        --exclude __pycache__/ --exclude .DS_Store \
        ./ "$SSH_USER@$ip:$REMOTE_DIR/"

    dir=results/phase2/$(remote_arch "$arch")
    if [ -d "$dir" ] && ! vm_ssh "$ip" "test -e $REMOTE_DIR/$dir"; then
        log "$arch: VM has no $dir; pushing the local copy so the runner resumes from it"
        vm_ssh "$ip" "mkdir -p $REMOTE_DIR/$dir"
        rsync -az -e "ssh $SSH_OPTS" "$dir/" "$SSH_USER@$ip:$REMOTE_DIR/$dir/"
    fi

    log "$arch: verifying model sha256 on the VM"
    vm_ssh "$ip" "cd $REMOTE_DIR/models && sha256sum -c ../cloud/models.sha256" ||
        die "$arch: model files on the VM do not match cloud/models.sha256"
done
