"""Export torchvision ResNet-50 to ONNX with dynamic batch/height/width (opset 17).

Usage:
    .venv/bin/python scripts/export_model.py [--out models/resnet50.onnx]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

OPSET = 17
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "resnet50.onnx")
    args = parser.parse_args()

    # weights=None: inference latency does not depend on the weight values, and this
    # avoids downloading ~100 MB of pretrained weights. Seeded so that re-exports
    # produce the same random weights.
    torch.manual_seed(0)
    model = torchvision.models.resnet50(weights=None).eval()
    dummy = torch.randn(1, 3, 224, 224)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False selects the TorchScript-based exporter, which targets opset 17
    # directly and takes dynamic_axes. The torch.export-based default (dynamo=True)
    # needs onnxscript and only starts at opset 18. The TorchScript path is
    # deprecated since torch 2.9 and emits a DeprecationWarning.
    torch.onnx.export(
        model,
        (dummy,),
        str(args.out),
        dynamo=False,
        opset_version=OPSET,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
        },
    )
    print(f"exported {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    onnx.checker.check_model(str(args.out), full_check=True)
    proto = onnx.load(str(args.out))
    opsets = {o.domain or "ai.onnx": o.version for o in proto.opset_import}
    print(f"onnx.checker: OK (full_check), opset_import={opsets}")
    for kind, values in (("input", proto.graph.input), ("output", proto.graph.output)):
        for v in values:
            dims = [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
            print(f"  {kind} {v.name}: {dims}")

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    print(f"onnxruntime {ort.__version__}, providers={sess.get_providers()}")

    # One inference at the export shape, compared against PyTorch.
    with torch.no_grad():
        ref = model(dummy).numpy()
    (out,) = sess.run(None, {"input": dummy.numpy()})
    max_abs = float(np.max(np.abs(out - ref)))
    rel = max_abs / float(np.max(np.abs(ref)))
    print(
        f"  run {tuple(dummy.shape)} -> {out.shape}, finite={np.isfinite(out).all()}, "
        f"range=[{out.min():.4g}, {out.max():.4g}], max|ort-torch|={max_abs:.3g} (rel {rel:.2g})"
    )
    if out.shape != (1, 1000) or not np.isfinite(out).all() or rel > 1e-3:
        raise SystemExit("test inference failed")

    # One inference at a different batch/height/width to confirm the axes are dynamic.
    x = np.random.default_rng(0).standard_normal((2, 3, 160, 192), dtype=np.float32)
    (out,) = sess.run(None, {"input": x})
    print(f"  run {x.shape} -> {out.shape}, finite={np.isfinite(out).all()}")
    if out.shape != (2, 1000) or not np.isfinite(out).all():
        raise SystemExit("dynamic-shape inference failed")


if __name__ == "__main__":
    main()
