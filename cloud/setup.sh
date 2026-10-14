#!/usr/bin/env bash
# Runs ON THE VM (Ubuntu 24.04), from the synced repo:
#   ssh ... 'bash onnx-docker-bench/cloud/setup.sh'
# Installs Docker CE from Docker's apt repository, quiets background maintenance,
# confirms cgroup v2, builds the benchmark image and verifies it. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

EXPECTED_ORT=1.31.0   # frozen, see docs/EXPERIMENT_PLAN.md
IMAGE=onnx-docker-bench:latest
SUDO=sudo
[ "$(id -u)" -eq 0 ] && SUDO=

# shellcheck disable=SC1091
. /etc/os-release
[ "$ID" = ubuntu ] && [ "$VERSION_ID" = 24.04 ] || { echo "expected Ubuntu 24.04, got $PRETTY_NAME" >&2; exit 1; }

echo "==> Docker CE from download.docker.com (https://docs.docker.com/engine/install/ubuntu/)"
if ! command -v docker >/dev/null; then
    $SUDO apt-get update -q
    $SUDO apt-get install -y -q ca-certificates curl
    $SUDO install -m 0755 -d /etc/apt/keyrings
    $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    $SUDO chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" |
        $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
    $SUDO apt-get update -q
    $SUDO apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi
$SUDO apt-get install -y -q tmux rsync git python3 >/dev/null
$SUDO usermod -aG docker "$(id -un)"   # takes effect at the next login
DOCKER="$SUDO docker"

echo "==> quiet machine: no unattended upgrades or snap refreshes during runs"
$SUDO systemctl disable --now unattended-upgrades.service apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
command -v snap >/dev/null && $SUDO snap refresh --hold >/dev/null 2>&1 || true

echo "==> cgroup v2"
fs=$(stat -fc %T /sys/fs/cgroup)
[ "$fs" = cgroup2fs ] || { echo "host is not on cgroup v2 (/sys/fs/cgroup is $fs)" >&2; exit 1; }
[ "$($DOCKER info --format '{{.CgroupVersion}}')" = 2 ] || { echo "docker does not report cgroup v2" >&2; exit 1; }
echo "    cgroup v2; docker cgroup driver: $($DOCKER info --format '{{.CgroupDriver}}')"

echo "==> models match cloud/models.sha256"
(cd models && sha256sum -c ../cloud/models.sha256)

echo "==> build $IMAGE"
$DOCKER build -q -t "$IMAGE" . >/dev/null
image_id=$($DOCKER image inspect --format '{{.Id}}' "$IMAGE")
probe='import platform, onnxruntime; print(onnxruntime.__version__, platform.machine())'
read -r ort machine <<<"$($DOCKER run --rm --entrypoint python "$IMAGE" -c "$probe")"
[ "$ort" = "$EXPECTED_ORT" ] || { echo "image has onnxruntime $ort, expected $EXPECTED_ORT" >&2; exit 1; }
[ "$machine" = "$(uname -m)" ] || { echo "image runs as $machine on a $(uname -m) host (emulated?)" >&2; exit 1; }
echo "    image $image_id: onnxruntime $ort, $machine, docker $($DOCKER version --format '{{.Server.Version}}')"

deadline=$(cat /var/lib/onnx-docker-bench-deadline 2>/dev/null || true)
if [ -n "$deadline" ]; then
    echo "==> auto-shutdown at $(date -u -d "@$deadline" '+%Y-%m-%d %H:%M UTC') ($(( (deadline - $(date +%s)) / 3600 ))h from now)"
else
    echo "==> WARNING: no auto-shutdown deadline found (/var/lib/onnx-docker-bench-deadline)" >&2
fi
echo "==> setup done; log out and back in (or start a new ssh session) for docker group membership"
