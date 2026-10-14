#!/usr/bin/env bash
# SSH to a benchmark VM by arch, optionally running one command.
#   cloud/ssh.sh arm64|x86_64 [command...]
set -euo pipefail
source "$(dirname "$0")/lib.sh"

arch=${1:?usage: cloud/ssh.sh arm64|x86_64 [command...]}
shift
case $arch in arm64 | x86_64) ;; *) die "arch must be arm64 or x86_64" ;; esac
require_aws
vm_ssh "$(instance_ip "$arch")" "$@"
