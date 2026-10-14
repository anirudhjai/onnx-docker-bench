"""Export torchvision ResNet-50 to ONNX with dynamic batch/height/width (opset 18).

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

# The opset torch's dynamo exporter natively implements (torchlib); lower opsets are
# produced by down-converting from it.
OPSET = 18
ROOT = Path(__file__).resolve().parent.parent


def check_against_torch(sess: ort.InferenceSession, model: torch.nn.Module, x: torch.Tensor) -> None:
    with torch.no_grad():
        ref = model(x).numpy()
    (out,) = sess.run(None, {"input": x.numpy()})
    max_abs = float(np.max(np.abs(out - ref)))
    rel = max_abs / float(np.max(np.abs(ref)))
    print(
        f"  run {tuple(x.shape)} -> {out.shape}, finite={np.isfinite(out).all()}, "
        f"range=[{out.min():.4g}, {out.max():.4g}], max|ort-torch|={max_abs:.3g} (rel {rel:.2g})"
    )
    if out.shape != (x.shape[0], 1000) or not np.isfinite(out).all() or rel > 1e-3:
        raise SystemExit(f"test inference failed for input shape {tuple(x.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "resnet50.onnx")
    args = parser.parse_args()

    # weights=None: inference latency does not depend on the weight values, and this
    # avoids downloading ~100 MB of pretrained weights. Seeded so that re-exports
    # produce the same random weights.
    torch.manual_seed(0)
    model = torchvision.models.resnet50(weights=None).eval()
    # Batch 2, not 1: torch.export specializes size-1 dims to a constant.
    dummy = torch.randn(2, 3, 224, 224)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    batch, height, width = (torch.export.Dim(n) for n in ("batch", "height", "width"))
    torch.onnx.export(
        model,
        (dummy,),
        str(args.out),
        dynamo=True,
        opset_version=OPSET,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes=({0: batch, 2: height, 3: width},),
        external_data=False,  # one self-contained file (~100 MB, well under the 2 GB limit)
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

    # One inference at the export shape, then one at a different batch/height/width
    # to confirm the axes are dynamic; both compared against PyTorch.
    check_against_torch(sess, model, dummy)
    check_against_torch(sess, model, torch.randn(1, 3, 160, 192))


if __name__ == "__main__":
    main()
