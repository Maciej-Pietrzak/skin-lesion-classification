"""Exports the trained checkpoint to ONNX so the architecture (not just the
weights) can be inspected visually, e.g. in Netron (netron.app).

The checkpoint itself is a raw state_dict -- just a flat bag of weight
tensors, with no record of how they're wired together. ONNX captures the
actual computation graph, which is what a graph viewer needs.
"""

import torch

from ml.src.model import input_size
from ml.src.registry import DEFAULT_VERSION, find_checkpoint, load_trained_model
from ml.src.train import DEVICE


def export(version: str = DEFAULT_VERSION):
    # Same name as the checkpoint, just .onnx (+ .onnx.data with the weights).
    path = find_checkpoint(version)
    output_path = path.with_suffix(".onnx")

    model, arch = load_trained_model(path, DEVICE)

    # Shape must match the architecture's input size (224x224 for B0, 260x260
    # for B2) and channel count (RGB) -- torch.onnx.export traces the graph by
    # actually running this dummy input through the model, so it only needs to
    # be shape-valid.
    dummy_input = torch.randn(1, 3, *input_size(arch), device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
    )
    print(f"Saved {output_path}")
    print("Open it at https://netron.app (drag & drop -- runs entirely in your browser, nothing is uploaded)")


if __name__ == "__main__":
    export()
