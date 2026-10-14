#!/usr/bin/env bash
# Terminate every instance tagged project=onnx-docker-bench, delete the key pair and
# security group, then list anything with the project tag that still exists.
#
#   cloud/teardown.sh [--dry-run] [--yes] [--force]
#
# Before terminating a running instance it checks that its results are already in the
# local results/phase2/<arch>/ (run cloud/fetch.sh first); --force skips that check.
# --dry-run asks AWS to validate the deletes without doing them.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

DRY_RUN=0 YES=0 FORCE=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
        --yes) YES=1 ;;
        --force) FORCE=1 ;;
        *) die "unknown option $arg" ;;
    esac
done
require_aws
cd "$ROOT"

dry() {
    local out
    if out=$("$@" --dry-run 2>&1); then echo "    unexpected success: $out"; return 1; fi
    if echo "$out" | grep -q DryRunOperation; then echo "    OK (would succeed): $1 $2 $3"
    else echo "    FAILED: $1 $2 $3: $out"; return 1; fi
}

IDS=$(instance_ids)
log "region $REGION; instances tagged project=$PROJECT: ${IDS:-none}"

# Refuse to destroy results that haven't been fetched.
if [ -n "$IDS" ] && [ $FORCE = 0 ]; then
    for arch in arm64 x86_64; do
        [ -n "$(instance_ids "$arch")" ] || continue
        ip=$(instance_ip "$arch") || die "$arch instance is not running, so its results can't be checked; use --force to skip"
        dir=results/phase2/$(remote_arch "$arch")
        staging=$(mktemp -d)
        rsync -az -e "ssh $SSH_OPTS" --include '*.csv' --exclude '*' "$SSH_USER@$ip:$REMOTE_DIR/$dir/" "$staging/" 2>/dev/null || true
        log "$arch: are the VM's results all in $dir?"
        mkdir -p "$dir"
        python3 cloud/check_superset.py "$dir" "$staging" ||
            die "$arch: the VM has runs that $dir lacks; run cloud/fetch.sh $arch first (or --force)"
        rm -rf "$staging"
    done
fi

if [ $DRY_RUN = 1 ]; then
    # shellcheck disable=SC2086
    [ -n "$IDS" ] && dry aws ec2 terminate-instances --instance-ids $IDS
    dry aws ec2 delete-key-pair --key-name "$KEY_NAME" || true
    sg=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)
    [ "$sg" != None ] && { dry aws ec2 delete-security-group --group-id "$sg" || true; }
    log "dry run: nothing was deleted"
    exit 0
fi

if [ $YES = 0 ]; then
    printf 'Terminate %s and delete key pair %s and security group %s? [y/N] ' "${IDS:-no instances}" "$KEY_NAME" "$SG_NAME"
    read -r answer
    [ "$answer" = y ] || die "aborted"
fi

if [ -n "$IDS" ]; then
    log "terminating $IDS"
    # shellcheck disable=SC2086
    aws ec2 terminate-instances --instance-ids $IDS --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
    # shellcheck disable=SC2086
    aws ec2 wait instance-terminated --instance-ids $IDS
fi

if [ "$(aws ec2 describe-key-pairs --filters "Name=key-name,Values=$KEY_NAME" --query 'length(KeyPairs)' --output text)" != 0 ]; then
    log "deleting key pair $KEY_NAME"
    aws ec2 delete-key-pair --key-name "$KEY_NAME"
fi
rm -f "$KEY_FILE" "$KNOWN_HOSTS"

sg=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)
if [ "$sg" != None ]; then
    log "deleting security group $sg"
    for _ in 1 2 3 4 5 6; do   # network interfaces can take a moment to release it
        aws ec2 delete-security-group --group-id "$sg" 2>/dev/null && break
        sleep 10
    done
fi

log "anything still tagged project=$PROJECT in $REGION:"
aws ec2 describe-instances --filters "Name=tag:project,Values=$PROJECT" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text | sed 's/^/    instance /'
aws ec2 describe-volumes --filters "Name=tag:project,Values=$PROJECT" --query 'Volumes[].[VolumeId,State]' --output text | sed 's/^/    volume /'
aws ec2 describe-security-groups --filters "Name=tag:project,Values=$PROJECT" --query 'SecurityGroups[].GroupId' --output text | sed 's/^/    security group /'
aws ec2 describe-key-pairs --filters "Name=tag:project,Values=$PROJECT" --query 'KeyPairs[].KeyName' --output text | sed 's/^/    key pair /'
aws resourcegroupstaggingapi get-resources --tag-filters "Key=project,Values=$PROJECT" \
    --query 'ResourceTagMappingList[].ResourceARN' --output text | tr '\t' '\n' | sed '/^$/d; s/^/    tagged resource /'
echo "    (terminated instances stay listed for about an hour; anything else above still exists)"
