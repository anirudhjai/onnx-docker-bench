#!/usr/bin/env bash
# Build the benchmark image and run exactly one container: --cpus=2, ORT default
# intra-op threads and spinning, appending one row to results/scratch/run_one.csv.
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE=onnx-docker-bench:latest

if ! docker info >/dev/null 2>&1; then
    echo "error: Docker is not running (docker info failed)" >&2
    exit 1
fi
if [[ ! -f models/resnet50.onnx ]]; then
    echo "error: models/resnet50.onnx not found; run scripts/export_model.py first" >&2
    exit 1
fi
mkdir -p results

docker build -t "$IMAGE" .
docker run --rm \
    --cpus=2 \
    -v "$PWD/models:/app/models:ro" \
    -v "$PWD/results:/app/results" \
    "$IMAGE" \
    --intra-threads 0 \
    --spin default \
    --label docker-cpus2-default \
    --out results/scratch/run_one.csv
