"""Produce an int8 ResNet-50 with ONNX Runtime dynamic quantization, and sanity-check it.

Usage:
    .venv/bin/python scripts/quantize_model.py [--src models/resnet50.onnx] [--out models/resnet50.int8.onnx]
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=ROOT / "models" / "resnet50.onnx")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "resnet50.int8.onnx")
    parser.add_argument("--check-inputs", type=int, default=32, help="fixed random inputs for the agreement check")
    parser.add_argument("--min-agreement", type=float, default=0.9, help="fail below this argmax agreement")
    args = parser.parse_args()

    model = onnx.load(str(args.src))
    # Workaround: the dynamo exporter writes shape annotations (graph.value_info) for
    # every value, weights included (fc.weight is annotated [1000, 2048]).
    # quantize_dynamic rewrites the classifier's Gemm(transB=1) as a MatMul with the
    # weight transposed to [2048, 1000] but keeps the old annotation, so its own
    # shape-inference pass fails with "Inferred shape and existing shape differ in
    # dimension 0: (2048) vs (1000)". The annotations are optional (shape inference
    # recomputes them), so drop them before quantizing.
    del model.graph.value_info[:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(model, args.out, weight_type=QuantType.QInt8)
    ops = collections.Counter(n.op_type for n in onnx.load(str(args.out)).graph.node)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  ops: {dict(ops)}")

    # Agreement check. The weights are random (see export_model.py), so this is a
    # sanity check that quantization didn't break the graph, not an accuracy result.
    # One input per run (batch 1, as benchmarked): DynamicQuantizeLinear picks its
    # scale from the whole tensor, so batching inputs would change the int8 result.
    opts = ort.SessionOptions()
    fp32 = ort.InferenceSession(str(args.src), opts, providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(args.out), opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    agree, worst_rel = 0, 0.0
    for _ in range(args.check_inputs):
        x = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
        (a,) = fp32.run(None, {"input": x})
        (b,) = int8.run(None, {"input": x})
        agree += int(a.argmax() == b.argmax())
        worst_rel = max(worst_rel, float(np.abs(a - b).max() / np.abs(a).max()))
    rate = agree / args.check_inputs
    print(
        f"  fp32 vs int8 on {args.check_inputs} fixed random inputs: argmax agreement {agree}/{args.check_inputs} "
        f"({rate:.0%}), worst max|diff|/max|fp32| = {worst_rel:.3f}"
    )
    if rate < args.min_agreement:
        raise SystemExit(f"argmax agreement {rate:.0%} below --min-agreement {args.min_agreement:.0%}")


if __name__ == "__main__":
    main()
