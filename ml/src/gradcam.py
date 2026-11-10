"""Grad-CAM visualizations for the trained HAM10000 classifier.

Highlights which pixels the model actually relied on for each prediction —
important for a medical-facing tool, so clinicians can sanity-check that the
model is looking at the lesion itself rather than, say, a ruler or hair.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from ml.src.dataloaders import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from ml.src.dataset import CLASSES, HAM10000Dataset
from ml.src.model import build_model
from ml.src.registry import DEFAULT_VERSION, checkpoint_path
from ml.src.train import DEVICE

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "gradcam"


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverses dataloaders.py's ImageNet normalization, producing a 0-1 RGB
    array (HWC) for show_cam_on_image to overlay the heatmap onto.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img.astype(np.float32)


def pick_one_index_per_class(test_ds: HAM10000Dataset) -> list[int]:
    """First test-set row for each of the 7 classes, so the output set covers
    every diagnosis instead of whatever happens to be first in splits.csv
    (which is grouped by lesion_id, not shuffled).
    """
    seen = {}
    for i, dx in enumerate(test_ds.df["dx"]):
        if dx not in seen:
            seen[dx] = i
    return [seen[cls] for cls in CLASSES if cls in seen]


def generate(version: str = DEFAULT_VERSION):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path(version), map_location=DEVICE))
    model.eval()

    # model.features[-1] is EfficientNet-B0's last conv block, the deepest
    # layer that still has spatial resolution (before global pooling)
    # -- the standard Grad-CAM target for this architecture.
    cam = GradCAM(model=model, target_layers=[model.features[-1]])

    test_ds = HAM10000Dataset("test", transform=build_transforms(train=False))

    for i in pick_one_index_per_class(test_ds):
        img_tensor, label = test_ds[i]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        rgb_img = denormalize(img_tensor)
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        with torch.no_grad():
            pred = model(input_tensor).argmax(dim=1).item()

        out_path = (
            OUTPUT_DIR / f"sample_{i}_true-{CLASSES[label]}_pred-{CLASSES[pred]}.jpg"
        )
        Image.fromarray(overlay).save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    generate()
