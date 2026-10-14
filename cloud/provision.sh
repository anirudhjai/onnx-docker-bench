#!/usr/bin/env bash
# Launch the benchmark instances: c8g.2xlarge (arm64) and c8a.2xlarge (x86_64),
# Ubuntu 24.04 LTS, a dedicated key pair and an SSH-only-from-your-IP security group.
#
#   cloud/provision.sh [--dry-run] [arm64|x86_64|all]
#
# --dry-run creates nothing: it resolves the AMIs and asks AWS to validate each
# create call (EC2 --dry-run), reporting "would succeed" or the error.
#
# Safety: instances terminate when shut down from inside, and user-data arms a
# poweroff SHUTDOWN_HOURS (default 24) after first boot, re-armed on every boot.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

DRY_RUN=0
if [ "${1:-}" = --dry-run ]; then DRY_RUN=1; shift; fi
TARGETS=$(targets "${1:-all}")
require_aws
mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR"

# EC2 --dry-run answers with an error: DryRunOperation means "would have succeeded".
dry() {
    local out
    if out=$("$@" --dry-run 2>&1); then
        echo "    unexpected success: $out"
        return 1
    fi
    if echo "$out" | grep -q DryRunOperation; then
        echo "    OK (would succeed): $1 $2 $3"
    else
        echo "    FAILED: $1 $2 $3: $out"
        return 1
    fi
}

log "region $REGION; targets: $TARGETS; auto-shutdown ${SHUTDOWN_HOURS}h after first boot; ${VOLUME_GB} GB gp3"

MY_IP=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
[[ $MY_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "could not determine your public IPv4 address (got '$MY_IP')"
log "SSH will be allowed from $MY_IP/32 only"

VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
[ "$VPC_ID" != None ] || die "no default VPC in $REGION; create one (aws ec2 create-default-vpc) or adapt this script"

# --- key pair (private key kept in cloud/.state/, never synced) -------------------
KEY_EXISTS=$(aws ec2 describe-key-pairs --filters "Name=key-name,Values=$KEY_NAME" --query 'length(KeyPairs)' --output text)
if [ "$KEY_EXISTS" = 0 ]; then
    if [ $DRY_RUN = 1 ]; then
        dry aws ec2 create-key-pair --key-name "$KEY_NAME" --key-type ed25519
    else
        [ ! -e "$KEY_FILE" ] || die "$KEY_FILE exists but no key pair '$KEY_NAME' in AWS; remove the stale file first"
        log "creating key pair $KEY_NAME -> $KEY_FILE"
        (umask 077 && aws ec2 create-key-pair --key-name "$KEY_NAME" --key-type ed25519 \
            --tag-specifications "ResourceType=key-pair,Tags=[{Key=project,Value=$PROJECT}]" \
            --query KeyMaterial --output text > "$KEY_FILE")
        KEY_EXISTS=1
    fi
else
    [ -s "$KEY_FILE" ] || die "key pair '$KEY_NAME' exists in AWS but $KEY_FILE is missing; run cloud/teardown.sh first"
    log "key pair $KEY_NAME exists"
fi

# --- security group: SSH from MY_IP only -------------------------------------------
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG_ID" = None ]; then
    if [ $DRY_RUN = 1 ]; then
        dry aws ec2 create-security-group --group-name "$SG_NAME" --description "$PROJECT SSH" --vpc-id "$VPC_ID"
    else
        log "creating security group $SG_NAME"
        SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "$PROJECT: SSH from one IP" \
            --vpc-id "$VPC_ID" --tag-specifications "ResourceType=security-group,Tags=[{Key=project,Value=$PROJECT}]" \
            --query GroupId --output text)
    fi
fi
if [ "$SG_ID" != None ]; then
    RULE="IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MY_IP/32,Description=$PROJECT-ssh}]"
    if [ $DRY_RUN = 1 ]; then
        dry aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions "$RULE" || true
    elif ! out=$(aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions "$RULE" 2>&1); then
        echo "$out" | grep -q InvalidPermission.Duplicate || die "$out"
    fi
    log "security group $SG_ID allows 22/tcp from $MY_IP/32"
fi

# --- user-data: auto-poweroff at a fixed deadline counted from first boot ----------
USER_DATA=$STATE_DIR/user-data.yaml
cat > "$USER_DATA" <<EOF
#cloud-config
# Safety net for forgotten instances: power off (which terminates them, see
# --instance-initiated-shutdown-behavior) ${SHUTDOWN_HOURS}h after first boot. The deadline
# is stored on first boot and re-armed on every boot, so a reboot cannot extend it.
write_files:
  - path: /usr/local/sbin/odb-deadline
    permissions: '0755'
    content: |
      #!/bin/sh
      f=/var/lib/onnx-docker-bench-deadline
      [ -f "\$f" ] || echo \$(( \$(date +%s) + ${SHUTDOWN_HOURS} * 3600 )) > "\$f"
      left=\$(( (\$(cat "\$f") - \$(date +%s)) / 60 ))
      [ "\$left" -ge 1 ] || left=1
      exec /usr/sbin/shutdown -h +"\$left" "onnx-docker-bench auto-shutdown deadline"
  - path: /etc/systemd/system/odb-deadline.service
    content: |
      [Unit]
      Description=onnx-docker-bench auto-shutdown deadline
      After=systemd-logind.service
      [Service]
      Type=oneshot
      ExecStart=/usr/local/sbin/odb-deadline
      [Install]
      WantedBy=multi-user.target
runcmd:
  - [systemctl, daemon-reload]
  - [systemctl, enable, --now, odb-deadline.service]
EOF

# --- instances --------------------------------------------------------------------
for arch in $TARGETS; do
    type=$(instance_type "$arch")
    ami=$(aws ssm get-parameter --name "$(ami_parameter "$arch")" --query Parameter.Value --output text)
    root_dev=$(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].RootDeviceName' --output text)
    ami_name=$(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].Name' --output text)
    log "$arch: $type, $ami ($ami_name), root $root_dev"

    existing=$(instance_ids "$arch")
    if [ -n "$existing" ]; then
        log "$arch: already have $existing, not launching another"
        continue
    fi

    set -- --image-id "$ami" --instance-type "$type" --count 1 \
        --block-device-mappings "DeviceName=$root_dev,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
        --instance-initiated-shutdown-behavior terminate \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --user-data "file://$USER_DATA" \
        --tag-specifications \
        "ResourceType=instance,Tags=[{Key=project,Value=$PROJECT},{Key=arch,Value=$arch},{Key=Name,Value=$PROJECT-$arch}]" \
        "ResourceType=volume,Tags=[{Key=project,Value=$PROJECT},{Key=arch,Value=$arch}]"
    # In a dry run the key pair / group may not exist yet, so validate without them.
    [ "$KEY_EXISTS" != 0 ] && set -- "$@" --key-name "$KEY_NAME"
    [ "$SG_ID" != None ] && set -- "$@" --security-group-ids "$SG_ID"

    if [ $DRY_RUN = 1 ]; then
        dry aws ec2 run-instances "$@"
        continue
    fi
    id=$(aws ec2 run-instances "$@" --query 'Instances[0].InstanceId' --output text)
    log "$arch: launched $id; waiting for it to run"
    aws ec2 wait instance-running --instance-ids "$id"
    ip=$(instance_ip "$arch")
    log "$arch: $id running at $ip (SSH comes up ~1 min later)"
done

if [ $DRY_RUN = 1 ]; then
    log "dry run: nothing was created"
else
    log "next: cloud/sync.sh, then cloud/setup.sh over ssh (see cloud/README.md)"
fi
