# onnx-docker-bench

How does ONNX Runtime CPU inference latency change inside Docker containers? This repo
measures a single model (torchvision ResNet-50) under different container CPU limits
(`docker run --cpus`), ORT thread settings, input sizes and quantization, and records
latency percentiles alongside the environment they were measured in (ORT/Python
versions, CPU count, affinity, cgroup CPU quota) so that every number can be traced
back to its conditions.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # runtime + export deps (torch, torchvision, onnx)
```

`requirements.txt` holds only the benchmark's runtime deps (onnxruntime, numpy).

## Export the model

```bash
.venv/bin/python scripts/export_model.py
```

Writes `models/resnet50.onnx` (opset 17, dynamic batch/height/width, random weights:
latency doesn't depend on weight values), validates it with `onnx.checker`, and runs
test inferences through onnxruntime. `models/` is gitignored.

For the int8 variant (ORT dynamic quantization, plus an fp32-vs-int8 argmax
sanity check on fixed random inputs):

```bash
.venv/bin/python scripts/quantize_model.py   # writes models/resnet50.int8.onnx
```

## Run one benchmark

On the host:

```bash
.venv/bin/python bench/run.py --label host --out results/phase1.csv
```

In Docker (builds the image, runs one container with `--cpus=2` and ORT's default
thread count):

```bash
scripts/run_one.sh
```

Both append one row to the CSV. See `python bench/run.py --help` for all options
(intra-op threads, thread spinning, optimization level, input size, batch,
warmup/iterations). ORT runs in its default sequential execution mode.

`results/` holds real experiment output and is tracked; put smoke tests and
throwaway runs in `results/scratch/` (gitignored). Each row has a `run_id`; with
`--save-raw`, per-iteration latencies go to `raw/<run_id>.csv` next to the `--out`
file (so `results/raw/` for official runs, `results/scratch/raw/` for scratch).

## Results

TODO
