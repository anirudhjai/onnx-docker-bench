#!/usr/bin/env bash
# Copy results back from the VM(s) into results/phase2/<arch>/ (aarch64 or x86_64).
#
#   cloud/fetch.sh [arm64|x86_64|all]
#
# Results are fetched into a staging directory first; a local CSV is only replaced
# if the fetched copy contains every run it has (cloud/check_superset.py).
set -euo pipefail
source "$(dirname "$0")/lib.sh"

TARGETS=$(targets "${1:-all}")
require_aws
cd "$ROOT"

for arch in $TARGETS; do
    ip=$(instance_ip "$arch")
    dir=results/phase2/$(remote_arch "$arch")
    staging=$(mktemp -d)
    log "$arch ($ip): fetching $dir"
    rsync -az -e "ssh $SSH_OPTS" "$SSH_USER@$ip:$REMOTE_DIR/$dir/" "$staging/"
    mkdir -p "$dir"
    python3 cloud/check_superset.py "$staging" "$dir" ||
        die "$arch: the VM's copy lacks runs that $dir has; not overwriting (staged in $staging)"
    rsync -a "$staging/" "$dir/"
    rm -rf "$staging"
    raw=$(find "$dir/raw" -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')
    log "$arch: $dir now has $raw raw files and: $(find "$dir" -maxdepth 1 -name '*.csv' -exec basename {} \; | tr '\n' ' ')"
done
