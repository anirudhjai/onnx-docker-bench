# shellcheck shell=bash disable=SC2034  # settings below are used by the scripts that source this file
# Shared settings and helpers for the cloud/*.sh scripts that run on your machine.
# Sourced, not executed. Bash 3.2 compatible (macOS /bin/bash).
# Every setting can be overridden from the environment, e.g. SHUTDOWN_HOURS=36.

CLOUD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$CLOUD_DIR/.." && pwd)

PROJECT=onnx-docker-bench
SHUTDOWN_HOURS=${SHUTDOWN_HOURS:-24}   # each instance powers off (= terminates) this long after first boot
VOLUME_GB=${VOLUME_GB:-20}             # gp3 root volume
KEY_NAME=${KEY_NAME:-$PROJECT}
SG_NAME=${SG_NAME:-$PROJECT-ssh}
SSH_USER=ubuntu
REMOTE_DIR=onnx-docker-bench           # relative to the VM user's home
STATE_DIR=$CLOUD_DIR/.state            # gitignored: private key, known_hosts, user-data
KEY_FILE=$STATE_DIR/$KEY_NAME.pem
KNOWN_HOSTS=$STATE_DIR/known_hosts

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

# Targets are named by the instance tag "arch": arm64 or x86_64.
targets() {
    case ${1:-all} in
        all) echo "arm64 x86_64" ;;
        arm64 | x86_64) echo "$1" ;;
        *) die "target must be arm64, x86_64 or all (got '$1')" ;;
    esac
}
instance_type() {
    case $1 in
        arm64) echo "${ARM64_TYPE:-c8g.2xlarge}" ;;
        x86_64) echo "${X86_64_TYPE:-c8a.2xlarge}" ;;
    esac
}
# What `uname -m` reports on the VM, i.e. the results/phase2/<dir> the runner writes.
remote_arch() {
    case $1 in
        arm64) echo aarch64 ;;
        x86_64) echo x86_64 ;;
    esac
}
# Canonical's public SSM parameter for the current Ubuntu 24.04 LTS AMI.
ami_parameter() {
    local a
    case $1 in arm64) a=arm64 ;; x86_64) a=amd64 ;; esac
    echo "/aws/service/canonical/ubuntu/server/24.04/stable/current/$a/hvm/ebs-gp3/ami-id"
}

require_aws() {
    command -v aws >/dev/null || die "AWS CLI not found; see cloud/README.md (Prerequisites)"
    REGION=${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region 2>/dev/null || true)}}
    [ -n "$REGION" ] || die "no AWS region: run 'aws configure' or export AWS_REGION"
    export AWS_DEFAULT_REGION=$REGION
    aws sts get-caller-identity >/dev/null || die "AWS credentials are not working (aws sts get-caller-identity)"
}

# Non-terminated instances with our project tag, optionally for one arch.
instance_ids() {
    local filters="Name=tag:project,Values=$PROJECT Name=instance-state-name,Values=pending,running,stopping,stopped"
    [ -n "${1:-}" ] && filters="$filters Name=tag:arch,Values=$1"
    # shellcheck disable=SC2086  # $filters is deliberately split into separate filters
    aws ec2 describe-instances --filters $filters --query 'Reservations[].Instances[].InstanceId' --output text
}

instance_ip() {
    local ip
    ip=$(aws ec2 describe-instances \
        --filters "Name=tag:project,Values=$PROJECT" "Name=tag:arch,Values=$1" "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    [ -n "$ip" ] && [ "$ip" != None ] || die "no running $1 instance tagged project=$PROJECT"
    echo "$ip"
}

SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS -o ServerAliveInterval=30"
vm_ssh() {
    local ip=$1
    shift
    # SSH_OPTS holds several options; callers build the remote command on purpose.
    # shellcheck disable=SC2086,SC2029
    ssh $SSH_OPTS "$SSH_USER@$ip" "$@"
}
