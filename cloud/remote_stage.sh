#!/usr/bin/env bash
# Runs ON THE VM (started by cloud/run_stage.sh): writes machine.json for the stage,
# checks the stage fits before the auto-shutdown deadline, then starts the runner in
# tmux session "odb" so it survives SSH disconnects. One stage at a time.
#   bash onnx-docker-bench/cloud/remote_stage.sh pilot|224|448
set -euo pipefail
cd "$(dirname "$0")/.."

stage=${1:?usage: remote_stage.sh pilot|224|448}
dir=results/phase2/$(uname -m)
case $stage in
    pilot) name=pilot; args="matrices/phase2-pilot.json" ;;
    224 | 448)
        name=input-$stage
        [ -f "$dir/pilot.csv" ] || { echo "run the pilot first: $dir/pilot.csv is missing" >&2; exit 1; }
        args="matrices/phase2.json --stage input=$stage --estimates $dir/pilot.csv"
        ;;
    *) echo "unknown stage '$stage' (pilot, 224 or 448)" >&2; exit 1 ;;
esac

if tmux has-session -t odb 2>/dev/null; then
    echo "a stage is already running (tmux session 'odb'); attach with: tmux attach -t odb" >&2
    exit 1
fi
if [ -n "$(docker ps -q --filter label=onnx-docker-bench.matrix)" ]; then
    echo "a benchmark container is running; not starting a second stage" >&2
    exit 1
fi

mkdir -p "$dir"
python3 scripts/record_machine.py --stage "$name" --out-dir "$dir"

# shellcheck disable=SC2086  # $args is a list of runner arguments
summary=$(python3 scripts/run_matrix.py $args --dry-run | tail -1)
echo "$summary"
deadline=$(cat "${ODB_DEADLINE_FILE:-/var/lib/onnx-docker-bench-deadline}" 2>/dev/null || echo 0)
python3 - "$deadline" "$summary" <<'EOF'
import re, sys, time
deadline, summary = int(sys.argv[1]), sys.argv[2]
left_h = (deadline - time.time()) / 3600
m = re.search(r"estimated (?:(\d+)h(\d+)m|(\d+)m(\d+)s)", summary)
eta_h = (int(m[1]) + int(m[2]) / 60 if m[1] else (int(m[3]) + int(m[4]) / 60) / 60) if m else None
left = f"in {left_h:.1f} h" if deadline else "not armed (no deadline file)"
print(f"auto-shutdown {left}; stage estimate {'unknown' if eta_h is None else f'{eta_h:.1f} h'}")
if deadline and eta_h is not None and eta_h + 0.5 > left_h:
    sys.exit("stage would not finish before the auto-shutdown; extend the deadline first (cloud/README.md)")
EOF

tmux new-session -d -s odb "python3 scripts/run_matrix.py $args 2>&1 | tee -a $dir/console-$name.log"
echo "started stage '$name' in tmux session 'odb' on $(hostname)"
echo "progress: tail -f $dir/$( [ "$stage" = pilot ] && echo pilot || echo results ).log"
