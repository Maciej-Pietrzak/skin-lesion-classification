"""Loads the trained EfficientNet checkpoint once (singleton) and runs prediction on an image."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.core.config import get_settings
from ml.src.dataloaders import build_transforms
from ml.src.dataset import CLASSES
from ml.src.model import input_size
from ml.src.registry import arch_from_checkpoint, load_trained_model


@lru_cache
def get_model() -> torch.nn.Module:
    # Cached so the checkpoint loads once per process (first call), not on
    # every prediction -- mirrors get_settings()'s @lru_cache in config.py.
    # The architecture (B0/B2) is read from the checkpoint's file name.
    settings = get_settings()
    model, _ = load_trained_model(settings.model_checkpoint_path, settings.device)
    return model


@lru_cache
def get_transform():
    """Preprocessing at the input size the loaded checkpoint was trained at
    (224x224 for B0, 260x260 for B2) -- shared by predict() and Grad-CAM."""
    arch = arch_from_checkpoint(get_settings().model_checkpoint_path)
    return build_transforms(train=False, input_size=input_size(arch))


def predict(image: Image.Image) -> dict[str, float]:
    """Runs one PIL image through the model, returns {class_code: probability} for all 7 CLASSES."""
    settings = get_settings()
    model = get_model()
    transform = get_transform()

    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(settings.device)
    with torch.no_grad():
        probabilities = F.softmax(model(input_tensor), dim=1)[0]

    return {cls: probabilities[i].item() for i, cls in enumerate(CLASSES)}
